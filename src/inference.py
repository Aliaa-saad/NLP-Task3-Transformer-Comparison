"""Inference adapters for the trained Yelp Model A and Model B checkpoints.

The adapters intentionally reproduce the training-time 256-token head+tail policy:
first 192 token IDs + last 64 token IDs, preserving [CLS] and [SEP].
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from src.model_a_arch import ID2LABEL, ModelAConfig, ScratchTransformerClassifier

LABELS = ("negative", "neutral", "positive")
MAX_LENGTH = 256
HEAD_TOKENS = 192
TAIL_TOKENS = 64


def head_tail_ids(tokenizer: Any, text: str, max_length: int = MAX_LENGTH,
                  head_tokens: int = HEAD_TOKENS, tail_tokens: int = TAIL_TOKENS) -> list[int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a non-empty review.")
    ids = tokenizer(text, add_special_tokens=True, truncation=False, padding=False,
                    verbose=False)["input_ids"]
    if len(ids) > max_length:
        ids = ids[:head_tokens] + (ids[-tail_tokens:] if tail_tokens else [])
    return list(ids[:max_length])


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _probability_payload(logits: torch.Tensor) -> dict[str, Any]:
    probs = torch.softmax(logits.float(), dim=-1)[0].detach().cpu().tolist()
    pred_id = max(range(len(probs)), key=probs.__getitem__)
    return {
        "label": LABELS[pred_id],
        "label_id": int(pred_id),
        "confidence": float(probs[pred_id]),
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(3)},
    }


def _resolve_existing(candidates: list[Path], is_dir: bool = False) -> Path | None:
    for path in candidates:
        if (path.is_dir() if is_dir else path.is_file()):
            return path
    return None


def _load_model_a_metadata(candidates: list[Path]) -> tuple[list[float], bool, dict[str, Any]]:
    biases = [0.0, 0.0, 0.0]
    baked_in = False
    inf_cfg: dict[str, Any] = {}
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "logit_biases" in data and isinstance(data["logit_biases"], list) and len(data["logit_biases"]) == 3:
            biases = [float(x) for x in data["logit_biases"]]
            baked_in = bool(data.get("logit_biases_baked_in", False))
            inf_cfg = data
            break
        candidate = data.get("logit_bias_tuning", {}).get("biases")
        if isinstance(candidate, list) and len(candidate) == 3:
            biases = [float(x) for x in candidate]
            baked_in = False
    return biases, baked_in, inf_cfg


def load_model_a(artifacts_root: Path, device: torch.device):
    model_dir = artifacts_root / "model_a"
    ckpt_path = model_dir / "best_model.pt"
    onnx_path = _resolve_existing([
        model_dir / "model_a.onnx",
        model_dir / "onnx" / "model_a.onnx",
        artifacts_root.parent / "model_a.onnx",
    ])
    tokenizer_dir = _resolve_existing(
        [
            d for d in [
                model_dir / "tokenizer",
                model_dir / "onnx",
                onnx_path.parent if onnx_path else None,
            ]
            if d is not None and ((d / "vocab.txt").is_file() or (d / "tokenizer.json").is_file())
        ],
        is_dir=True,
    )
    if tokenizer_dir is None:
        raise FileNotFoundError(f"Missing Model A tokenizer folder under: {model_dir / 'tokenizer'}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    meta_candidates = [
        model_dir / "inference_config.json",
        model_dir / "onnx" / "inference_config.json",
        *( [onnx_path.parent / "inference_config.json"] if onnx_path else [] ),
        model_dir / "metrics.json",
    ]
    biases, biases_baked_in, inf_cfg = _load_model_a_metadata(meta_candidates)

    if ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_cfg" not in payload or "model_state_dict" not in payload:
            raise ValueError("Model A checkpoint must contain model_cfg and model_state_dict.")
        cfg = ModelAConfig(**payload["model_cfg"])
        model = ScratchTransformerClassifier(cfg)
        state = payload["model_state_dict"]
        state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                 for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return {"backend": "pytorch", "model": model, "tokenizer": tokenizer, "config": cfg,
                "biases": biases, "biases_baked_in": False,
                "parameters": sum(p.numel() for p in model.parameters())}

    if onnx_path is None:
        raise FileNotFoundError(f"Missing Model A checkpoint or ONNX model: {ckpt_path}")

    import onnxruntime as ort

    cfg_kwargs = {
        k: int(inf_cfg[k])
        for k in ("max_seq_len", "head_tokens", "tail_tokens", "pad_token_id")
        if k in inf_cfg
    }
    cfg = ModelAConfig(**cfg_kwargs)
    available = set(ort.get_available_providers())
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    return {"backend": "onnx", "session": session, "onnx_path": str(onnx_path),
            "tokenizer": tokenizer, "config": cfg,
            "biases": biases, "biases_baked_in": biases_baked_in}


def load_model_b(artifacts_root: Path, device: torch.device):
    model_dir = artifacts_root / "model_b" / "model"
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing Model B config: {model_dir / 'config.json'}")
    if not ((model_dir / "model.safetensors").is_file() or
            (model_dir / "pytorch_model.bin").is_file()):
        raise FileNotFoundError(f"Missing Model B weights under: {model_dir}")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    actual = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if actual != {0: "negative", 1: "neutral", 2: "positive"}:
        raise ValueError(f"Model B id2label mismatch: {actual}")
    model.to(device).eval()
    return {"model": model, "tokenizer": tokenizer,
            "parameters": sum(p.numel() for p in model.parameters())}


def verify_tokenizer_compatibility(model_a: dict[str, Any], model_b: dict[str, Any]) -> None:
    """Fail closed if local inference tokenizers do not map text to identical IDs."""
    if model_a["tokenizer"].get_vocab() != model_b["tokenizer"].get_vocab():
        raise ValueError("Model A and Model B tokenizer vocabularies differ.")
    probes = [
        "The food was excellent, and the staff were friendly!",
        "I wouldn't recommend this place. The service was VERY slow.",
        "Café crème, punctuation... and mixed-case TEXT.",
    ]
    for text in probes:
        ids_a = head_tail_ids(model_a["tokenizer"], text)
        ids_b = head_tail_ids(model_b["tokenizer"], text)
        if ids_a != ids_b:
            raise ValueError(f"Tokenizer output mismatch for probe: {text!r}")


def predict_model_a(bundle: dict[str, Any], text: str, device: torch.device,
                    apply_validation_biases: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = bundle["config"]
    ids = head_tail_ids(bundle["tokenizer"], text, cfg.max_seq_len, cfg.head_tokens, cfg.tail_tokens)
    biases_baked_in = bool(bundle.get("biases_baked_in", False))

    if bundle.get("backend") == "onnx":
        import numpy as np

        np_ids = np.array([ids], dtype=np.int64)
        np_mask = (np_ids != cfg.pad_token_id).astype(np.int64)
        forward_start = time.perf_counter()
        raw_logits = bundle["session"].run(
            ["logits"],
            {"input_ids": np_ids, "attention_mask": np_mask},
        )[0]
        forward_ms = (time.perf_counter() - forward_start) * 1000.0
        logits = torch.from_numpy(raw_logits).float()
        if biases_baked_in and not apply_validation_biases:
            logits = logits - torch.tensor(bundle["biases"], dtype=torch.float32)
        elif not biases_baked_in and apply_validation_biases:
            logits = logits + torch.tensor(bundle["biases"], dtype=torch.float32)
    else:
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = input_ids.ne(cfg.pad_token_id)
        _device_sync(device)
        forward_start = time.perf_counter()
        with torch.inference_mode(), _autocast_context(device):
            logits = bundle["model"](input_ids, attention_mask)
        if apply_validation_biases:
            logits = logits.float() + torch.tensor(bundle["biases"], device=device)
        _device_sync(device)
        forward_ms = (time.perf_counter() - forward_start) * 1000.0

    result = _probability_payload(logits)
    result.update({"model": "Model A — Scratch Transformer", "model_latency_ms": forward_ms,
                   "end_to_end_latency_ms": (time.perf_counter() - started) * 1000.0,
                   "token_count": len(ids), "truncated": len(ids) >= cfg.max_seq_len,
                   "validation_biases_applied": bool(apply_validation_biases)})
    return result


def predict_model_b(bundle: dict[str, Any], text: str, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    ids = head_tail_ids(bundle["tokenizer"], text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    _device_sync(device)
    forward_start = time.perf_counter()
    with torch.inference_mode(), _autocast_context(device):
        logits = bundle["model"](input_ids=input_ids, attention_mask=attention_mask).logits
    _device_sync(device)
    forward_ms = (time.perf_counter() - forward_start) * 1000.0
    result = _probability_payload(logits)
    result.update({"model": "Model B — Pretrained DistilBERT", "model_latency_ms": forward_ms,
                   "end_to_end_latency_ms": (time.perf_counter() - started) * 1000.0,
                   "token_count": len(ids), "truncated": len(ids) >= MAX_LENGTH,
                   "validation_biases_applied": False})
    return result
