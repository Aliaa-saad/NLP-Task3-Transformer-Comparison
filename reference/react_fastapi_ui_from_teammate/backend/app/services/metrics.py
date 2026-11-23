"""Read and normalise the offline artifacts used by the Compare dashboard."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import REPO_DIR, Settings
from app.schemas import ComparisonResults, ErrorRow, HistoryRow, SetupResponse, TrainingSummary


class ArtifactMissing(FileNotFoundError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactMissing(f"Missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def load_comparison(settings: Settings) -> ComparisonResults:
    return ComparisonResults.model_validate(_read_json(settings.comparison_path))


def _first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


MODEL_B_NOTEBOOK = REPO_DIR / "notebooks" / "01_pretrained_transformer.ipynb"


def _model_b_results_path(settings: Settings) -> Path | None:
    return _first_existing(
        [
            settings.model_b_dir / "pretrained_results.json",
            settings.artifacts_dir / "comparison" / "pretrained_results.json",
            REPO_DIR / "results" / "pretrained_results.json",
        ]
    )


def _gpu_from_notebook(path: Path) -> str | None:
    """Read the GPU name printed in a training notebook's saved outputs (e.g. 'GPU: Tesla T4')."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", "")) or "".join(out.get("data", {}).get("text/plain", ""))
            match = re.search(r"^GPU(?: Name)?:\s*(.+?)\s*$", text, re.M)
            if match:
                return match.group(1)
    return None


def _read_model_a_metrics(settings: Settings) -> dict[str, Any] | None:
    try:
        return _read_json(settings.model_a_dir / "metrics.json")
    except (ArtifactMissing, json.JSONDecodeError):
        return None


def _read_model_b_results(settings: Settings) -> dict[str, Any] | None:
    path = _model_b_results_path(settings)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _model_b_gpu(b: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if b and (b.get("gpu_name") or b.get("gpu")):
        return b.get("gpu_name") or b.get("gpu"), "pretrained_results.json"
    gpu = _gpu_from_notebook(MODEL_B_NOTEBOOK)
    return (gpu, "notebooks/01_pretrained_transformer.ipynb (saved output)") if gpu else (None, None)


def load_training_summaries(settings: Settings) -> dict[str, TrainingSummary]:
    out: dict[str, TrainingSummary] = {
        "model_a": TrainingSummary(regime="scratch training"),
        "model_b": TrainingSummary(regime="fine-tuning"),
    }

    m = _read_model_a_metrics(settings)
    if m is not None:
        training = m.get("training", {})
        split = m.get("split_manifest", {})
        gpu = m.get("hardware", {}).get("gpu_name")
        out["model_a"] = TrainingSummary(
            parameters=m.get("parameters", {}).get("total"),
            training_time_sec=training.get("elapsed_train_sec"),
            epochs=training.get("epochs"),
            best_val_macro_f1=training.get("best_val_macro_f1"),
            train_size=(split.get("splits", {}).get("train", {}) or {}).get("size"),
            regime="scratch training",
            gpu_name=gpu,
            gpu_source="model_a/metrics.json" if gpu else None,
        )

    b = _read_model_b_results(settings)
    if b is not None:
        gpu, src = _model_b_gpu(b)
        out["model_b"] = TrainingSummary(
            parameters=b.get("total_parameters"),
            training_time_sec=b.get("training_time_seconds"),
            epochs=b.get("epochs"),
            best_val_macro_f1=b.get("best_validation_f1"),
            train_size=b.get("train_size"),
            regime="fine-tuning",
            gpu_name=gpu,
            gpu_source=src,
        )
    return out


def load_setup(settings: Settings) -> SetupResponse:
    m = _read_model_a_metrics(settings) or {}
    b = _read_model_b_results(settings) or {}
    split = m.get("split_manifest", {})
    counts = split.get("counts", {})
    ds = m.get("dataset", {})
    try:
        lat = load_comparison(settings).latency_same_machine
    except (ArtifactMissing, ValueError):
        lat = None
    gpu_b, gpu_b_src = _model_b_gpu(b)
    return SetupResponse(
        dataset={
            "name": "Yelp Open Dataset",
            "slug": ds.get("slug"),
            "file": ds.get("file"),
            "url": ds.get("url"),
            "raw_reviews": counts.get("raw"),
            "duplicates_removed": counts.get("dropped_duplicate"),
            "deduped_reviews": counts.get("deduped"),
        },
        label_mapping=[
            {"id": 0, "label": "negative", "stars": "1–2"},
            {"id": 1, "label": "neutral", "stars": "3"},
            {"id": 2, "label": "positive", "stars": "4–5"},
        ],
        model_a={
            "splits": {k: v.get("size") for k, v in split.get("splits", {}).items()},
            "split_strategy": "deduplicated, stratified",
            "seed": split.get("seed"),
            "tokenizer": split.get("tokenizer_name"),
            "pretrained_weights": False,
            "max_length": split.get("max_seq_len"),
            "head_tokens": split.get("head_tokens"),
            "tail_tokens": split.get("tail_tokens"),
            "gpu": m.get("hardware", {}).get("gpu_name"),
        },
        model_b={
            "splits": {"train": b.get("train_size"), "val": b.get("validation_size"), "test": b.get("test_size")},
            "split_strategy": "balanced sample",
            "seed": b.get("random_seed"),
            "tokenizer": b.get("model_name"),
            "pretrained_weights": True,
            "max_length": b.get("max_length"),
            "gpu": gpu_b,
            "gpu_source": gpu_b_src,
        },
        comparison={
            "device": lat.device if lat else None,
            "gpu": lat.gpu_name if lat else None,
            "protocol": lat.protocol if lat else None,
        },
    )


def load_history(settings: Settings) -> list[HistoryRow]:
    path = settings.model_a_dir / "training_history.csv"
    if not path.exists():
        raise ArtifactMissing(f"Missing artifact: {path}")
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce")
    rows = [{k: _clean(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]
    return [HistoryRow.model_validate(r) for r in rows]


def load_errors(settings: Settings, limit: int) -> list[ErrorRow]:
    path = settings.model_a_dir / "error_analysis_top50.csv"
    if not path.exists():
        raise ArtifactMissing(f"Missing artifact: {path}")
    df = pd.read_csv(path, keep_default_na=False)
    rows: list[ErrorRow] = []
    for rec in df.head(limit).to_dict(orient="records"):
        rows.append(
            ErrorRow(
                rank=int(rec.get("rank", len(rows) + 1)),
                text=str(rec.get("text_or_index", rec.get("text", ""))),
                true_label=str(rec.get("true_label", "")),
                pred_label=str(rec.get("pred_label", "")),
                confidence=float(rec.get("confidence", 0.0) or 0.0),
                prob_negative=_to_float(rec.get("prob_negative")),
                prob_neutral=_to_float(rec.get("prob_neutral")),
                prob_positive=_to_float(rec.get("prob_positive")),
            )
        )
    return rows


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
