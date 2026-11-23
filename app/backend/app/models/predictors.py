"""Model A / Model B predictors behind a common interface, plus a fault-tolerant registry."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch

from app.config import Settings
from app.models.model_a_arch import ID2LABEL, ModelAConfig, ScratchTransformerClassifier
from app.models.preprocessing import encode_model_a, pad_batch
from app.schemas import LABELS, Prediction, Probs

log = logging.getLogger(__name__)
T = TypeVar("T")

MODEL_A_KEY = "model_a"
MODEL_B_KEY = "model_b"
MODEL_A_NAME = "Model A — Scratch Transformer (40.8M)"
MODEL_B_NAME = "Model B — DistilBERT fine-tuned (67.0M)"


class SentimentPredictor(Protocol):
    key: str
    name: str

    def predict(self, texts: list[str], apply_biases: bool = True) -> list[Prediction]: ...


# --------------------------------------------------------------------------- helpers
def resolve_device(setting: str) -> torch.device:
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(setting)
    if dev.type == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return dev


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn: Callable[[], T], device: torch.device) -> tuple[T, float]:
    """Run fn and return (result, elapsed ms), synchronising CUDA before and after."""
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, (time.perf_counter() - t0) * 1000.0


def strip_orig_mod(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefix = "_orig_mod."
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _probs(row: list[float]) -> Probs:
    return Probs(**{name: float(row[i]) for i, name in enumerate(LABELS)})


# --------------------------------------------------------------------------- Model A
class ModelAPredictor:
    key = MODEL_A_KEY
    name = MODEL_A_NAME
    tokenizer_desc = "bert-base-uncased (WordPiece, 30,522)"
    truncation_desc = "head+tail: first 192 + last 64 tokens (max 256)"

    def __init__(self, settings: Settings, device: torch.device) -> None:
        self.settings = settings
        self.device = device
        self.demo_mode = False
        self.bias_warning: str | None = None
        self._lock = threading.Lock()

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_a_tokenizer)

        ckpt_path = settings.model_a_dir / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            self.cfg = ModelAConfig(**ckpt["model_cfg"])
            model = ScratchTransformerClassifier(self.cfg)
            model.load_state_dict(strip_orig_mod(ckpt["model_state_dict"]), strict=True)
        elif settings.demo_random_weights:
            log.warning("Model A checkpoint missing; DEMO_RANDOM_WEIGHTS=1 -> random init")
            self.cfg = ModelAConfig()
            torch.manual_seed(0)
            model = ScratchTransformerClassifier(self.cfg)
            self.demo_mode = True
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. Copy best_model.pt there "
                "(or set DEMO_RANDOM_WEIGHTS=1 for development)."
            )

        self.model = model.to(device).eval()
        self.parameters = count_params(self.model)
        self.biases = self._load_biases(settings.model_a_dir / "metrics.json")

    def _load_biases(self, path: Path) -> list[float]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            biases = [float(b) for b in data["logit_bias_tuning"]["biases"]]
            if len(biases) != self.cfg.num_classes:
                raise ValueError(f"expected {self.cfg.num_classes} biases, got {len(biases)}")
            self.biases_available = True
            return biases
        except Exception as exc:  # noqa: BLE001 - any problem -> zero biases + warning
            self.biases_available = False
            self.bias_warning = f"Logit biases unavailable ({exc}); using zeros."
            log.warning(self.bias_warning)
            return [0.0] * self.cfg.num_classes

    @torch.inference_mode()
    def predict(self, texts: list[str], apply_biases: bool = True) -> list[Prediction]:
        with self._lock:
            def _tok() -> list[Any]:
                return [
                    encode_model_a(
                        self.tokenizer,
                        t,
                        max_len=self.cfg.max_seq_len,
                        head_tokens=self.cfg.head_tokens,
                        tail_tokens=self.cfg.tail_tokens,
                    )
                    for t in texts
                ]

            encoded, tok_ms = timed(_tok, torch.device("cpu"))
            ids = torch.tensor(
                pad_batch([e.input_ids for e in encoded], self.cfg.pad_token_id),
                dtype=torch.long,
                device=self.device,
            )
            mask = ids != self.cfg.pad_token_id
            logits, fwd_ms = timed(lambda: self.model(ids, mask), self.device)

            logits = logits.float()
            if apply_biases:
                logits = logits + torch.tensor(self.biases, device=logits.device)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        warnings = []
        if self.demo_mode:
            warnings.append("DEMO MODE: random weights, predictions are meaningless.")
        if apply_biases and self.bias_warning:
            warnings.append(self.bias_warning)

        out = []
        for enc, row in zip(encoded, probs):
            label_id = int(max(range(len(row)), key=row.__getitem__))
            out.append(
                Prediction(
                    model_key=self.key,
                    model_name=self.name,
                    label=ID2LABEL[label_id],  # type: ignore[arg-type]
                    label_id=label_id,
                    probs=_probs(row),
                    latency_ms=round(fwd_ms, 3),
                    tokenize_ms=round(tok_ms, 3),
                    num_tokens=enc.num_tokens,
                    original_tokens=enc.original_tokens,
                    truncated=enc.truncated,
                    biases_applied=apply_biases,
                    demo_mode=self.demo_mode,
                    warning=" ".join(warnings) or None,
                )
            )
        return out


# --------------------------------------------------------------------------- Model B
class ModelBPredictor:
    key = MODEL_B_KEY
    name = MODEL_B_NAME
    tokenizer_desc = "distilbert-base-uncased (WordPiece, 30,522)"
    truncation_desc = "right truncation (max 256)"
    biases_available = False
    biases: list[float] | None = None
    max_length = 256

    def __init__(self, settings: Settings, device: torch.device) -> None:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.settings = settings
        self.device = device
        self.demo_mode = False
        self._lock = threading.Lock()

        model_dir = settings.model_b_dir
        has_weights = (model_dir / "config.json").exists() and (
            (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
        )
        if has_weights:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        elif settings.demo_random_weights:
            log.warning("Model B weights missing; DEMO_RANDOM_WEIGHTS=1 -> random init")
            cfg = AutoConfig.from_pretrained(
                settings.model_b_base,
                num_labels=3,
                id2label=dict(ID2LABEL),
                label2id={v: k for k, v in ID2LABEL.items()},
            )
            torch.manual_seed(0)
            model = AutoModelForSequenceClassification.from_config(cfg)
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_b_base)
            self.demo_mode = True
        else:
            raise FileNotFoundError(
                f"Model folder not found or incomplete: {model_dir} "
                "(needs config.json + model.safetensors + tokenizer files)."
            )

        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        if id2label != ID2LABEL:
            raise ValueError(f"Unexpected id2label {model.config.id2label}; expected {ID2LABEL}")

        self.model = model.to(device).eval()
        self.parameters = count_params(self.model)

    @torch.inference_mode()
    def predict(self, texts: list[str], apply_biases: bool = True) -> list[Prediction]:
        del apply_biases  # Model B uses plain softmax
        with self._lock:
            def _tok() -> tuple[Any, list[int]]:
                enc = self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                full = self.tokenizer(texts, truncation=False, verbose=False)["input_ids"]
                return enc, [len(f) for f in full]

            (enc, full_lens), tok_ms = timed(_tok, torch.device("cpu"))
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out, fwd_ms = timed(lambda: self.model(**enc), self.device)
            probs = torch.softmax(out.logits.float(), dim=-1).cpu().tolist()
            kept = enc["attention_mask"].sum(dim=1).cpu().tolist()

        results = []
        for row, n_kept, n_full in zip(probs, kept, full_lens):
            label_id = int(max(range(len(row)), key=row.__getitem__))
            results.append(
                Prediction(
                    model_key=self.key,
                    model_name=self.name,
                    label=ID2LABEL[label_id],  # type: ignore[arg-type]
                    label_id=label_id,
                    probs=_probs(row),
                    latency_ms=round(fwd_ms, 3),
                    tokenize_ms=round(tok_ms, 3),
                    num_tokens=int(n_kept),
                    original_tokens=int(n_full),
                    truncated=n_full > n_kept,
                    biases_applied=False,
                    demo_mode=self.demo_mode,
                    warning="DEMO MODE: random weights, predictions are meaningless."
                    if self.demo_mode
                    else None,
                )
            )
        return results


# --------------------------------------------------------------------------- registry
class ModelRegistry:
    """Loads each model independently; a failure in one never blocks the other."""

    specs: list[tuple[str, str, type]] = [
        (MODEL_A_KEY, MODEL_A_NAME, ModelAPredictor),
        (MODEL_B_KEY, MODEL_B_NAME, ModelBPredictor),
    ]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.predictors: dict[str, Any] = {}
        self.errors: dict[str, str] = {}

    @property
    def names(self) -> dict[str, str]:
        return {key: name for key, name, _ in self.specs}

    def load_all(self) -> None:
        for key, _name, cls in self.specs:
            try:
                t0 = time.perf_counter()
                predictor = cls(self.settings, self.device)
                if not self.settings.skip_warmup:
                    predictor.predict(["Warm-up pass for latency measurement."])
                self.predictors[key] = predictor
                log.info("Loaded %s in %.1fs", key, time.perf_counter() - t0)
            except Exception as exc:  # noqa: BLE001 - keep the server alive
                log.exception("Failed to load %s", key)
                self.errors[key] = f"{type(exc).__name__}: {exc}"

    def predict_all(self, text: str, apply_biases: bool) -> list[Prediction]:
        results: list[Prediction] = []
        for key, name, _ in self.specs:
            predictor = self.predictors.get(key)
            if predictor is None:
                results.append(
                    Prediction(
                        model_key=key,
                        model_name=name,
                        error=self.errors.get(key, "Model not loaded."),
                    )
                )
                continue
            try:
                results.append(predictor.predict([text], apply_biases=apply_biases)[0])
            except Exception as exc:  # noqa: BLE001
                log.exception("Prediction failed for %s", key)
                results.append(
                    Prediction(model_key=key, model_name=name, error=f"Inference failed: {exc}")
                )
        return results
