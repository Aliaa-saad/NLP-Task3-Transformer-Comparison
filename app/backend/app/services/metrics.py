"""Read and normalise the offline artifacts used by the Compare dashboard."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import REPO_DIR, Settings
from app.schemas import ComparisonResults, ErrorRow, HistoryRow, TrainingSummary


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


def load_training_summaries(settings: Settings) -> dict[str, TrainingSummary]:
    out: dict[str, TrainingSummary] = {"model_a": TrainingSummary(), "model_b": TrainingSummary()}

    try:
        m = _read_json(settings.model_a_dir / "metrics.json")
        training = m.get("training", {})
        split = m.get("split_manifest", {})
        out["model_a"] = TrainingSummary(
            parameters=m.get("parameters", {}).get("total"),
            training_time_sec=training.get("elapsed_train_sec"),
            epochs=training.get("epochs"),
            best_val_macro_f1=training.get("best_val_macro_f1"),
            train_size=(split.get("splits", {}).get("train", {}) or {}).get("size"),
        )
    except (ArtifactMissing, json.JSONDecodeError):
        pass

    b_path = _first_existing(
        [
            settings.model_b_dir / "pretrained_results.json",
            settings.artifacts_dir / "comparison" / "pretrained_results.json",
            REPO_DIR / "results" / "pretrained_results.json",
        ]
    )
    if b_path is not None:
        try:
            b = json.loads(b_path.read_text(encoding="utf-8"))
            out["model_b"] = TrainingSummary(
                parameters=b.get("total_parameters"),
                training_time_sec=b.get("training_time_seconds"),
                epochs=b.get("epochs"),
                best_val_macro_f1=b.get("best_validation_f1"),
                train_size=b.get("train_size"),
            )
        except json.JSONDecodeError:
            pass
    return out


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
