"""Build a leakage-safe, paired Model A vs Model B evaluation from test predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABELS = ["negative", "neutral", "positive"]
LABEL_IDS = [0, 1, 2]


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=LABEL_IDS, target_names=LABELS,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            label: {"precision": float(report[label]["precision"]),
                    "recall": float(report[label]["recall"]),
                    "f1": float(report[label]["f1-score"]),
                    "support": int(report[label]["support"])}
            for label in LABELS
        },
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
    }


def _paired_bootstrap_f1_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray,
                              seed: int = 42, n_bootstrap: int = 300) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    n = len(y)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = f1_score(y[idx], a[idx], labels=LABEL_IDS, average="macro", zero_division=0) - \
                   f1_score(y[idx], b[idx], labels=LABEL_IDS, average="macro", zero_division=0)
    return {"a_minus_b_mean": float(diffs.mean()),
            "ci95_low": float(np.quantile(diffs, 0.025)),
            "ci95_high": float(np.quantile(diffs, 0.975)),
            "bootstrap_samples": int(n_bootstrap)}


def _balanced_positions(frame: pd.DataFrame, seed: int = 42) -> np.ndarray:
    counts = frame["true_label_id"].value_counts().to_dict()
    n_each = min(int(counts.get(i, 0)) for i in LABEL_IDS)
    if n_each <= 0:
        raise ValueError("The held-out test set must contain all three sentiment classes.")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label in LABEL_IDS:
        positions = np.flatnonzero(frame["true_label_id"].to_numpy() == label)
        chosen.extend(rng.choice(positions, size=n_each, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def _view_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["true_label_id"].to_numpy(dtype=np.int64)
    a_raw = frame["model_a_raw_pred"].to_numpy(dtype=np.int64)
    a_adj = frame["model_a_adjusted_pred"].to_numpy(dtype=np.int64)
    b = frame["model_b_pred"].to_numpy(dtype=np.int64)
    return {
        "size": int(len(frame)),
        "class_counts": {LABELS[i]: int(np.sum(y == i)) for i in LABEL_IDS},
        "model_a_raw": metric_block(y, a_raw),
        "model_a_logit_adjusted": metric_block(y, a_adj),
        "model_b": metric_block(y, b),
        "bootstrap_a_raw_minus_b": _paired_bootstrap_f1_diff(y, a_raw, b),
        "bootstrap_a_adjusted_minus_b": _paired_bootstrap_f1_diff(y, a_adj, b),
        "agreement_a_raw_vs_b": float(np.mean(a_raw == b)),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison(artifacts_dir: Path, output_dir: Path, seed: int = 42,
                     strict_full_data: bool = True) -> dict[str, Any]:
    a_dir = artifacts_dir / "model_a"
    b_dir = artifacts_dir / "model_b"
    a_pred_path = a_dir / "test_predictions.csv"
    b_pred_path = b_dir / "test_predictions.csv"
    a_metrics_path = a_dir / "metrics.json"
    b_metrics_path = b_dir / "metrics.json"
    required = [a_pred_path, b_pred_path, a_metrics_path, b_metrics_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing final comparison inputs:\n" + "\n".join(missing))

    a = pd.read_csv(a_pred_path)
    b = pd.read_csv(b_pred_path)
    a_required = {"record_index", "true_label_id", "model_a_raw_pred", "model_a_adjusted_pred"}
    b_required = {"record_index", "true_label_id", "predicted_label_id"}
    if not a_required.issubset(a.columns):
        raise ValueError(f"Model A predictions missing columns: {sorted(a_required - set(a.columns))}")
    if not b_required.issubset(b.columns):
        raise ValueError(f"Model B predictions missing columns: {sorted(b_required - set(b.columns))}")
    if a["record_index"].duplicated().any() or b["record_index"].duplicated().any():
        raise ValueError("Duplicate record_index values found in a prediction file.")

    merged = a[list(a_required)].merge(
        b[["record_index", "true_label_id", "predicted_label_id"]],
        on="record_index", how="outer", suffixes=("_a", "_b"), indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Model A and Model B test indices differ; refusing an unpaired comparison.")
    if not np.array_equal(merged["true_label_id_a"].to_numpy(), merged["true_label_id_b"].to_numpy()):
        raise ValueError("Ground-truth labels differ for the same test record indices.")
    merged = merged.rename(columns={
        "true_label_id_a": "true_label_id",
        "predicted_label_id": "model_b_pred",
    }).drop(columns=["true_label_id_b", "_merge"])
    merged = merged.sort_values("record_index").reset_index(drop=True)
    for col in ["true_label_id", "model_a_raw_pred", "model_a_adjusted_pred", "model_b_pred"]:
        values = set(merged[col].astype(int).unique())
        if not values.issubset(set(LABEL_IDS)):
            raise ValueError(f"Unexpected class IDs in {col}: {values}")

    a_metrics = _read_json(a_metrics_path)
    b_metrics = _read_json(b_metrics_path)
    a_splits = a_metrics.get("split_manifest", {}).get("splits", {})
    b_splits = b_metrics.get("splits", {})
    for split_name in ("train", "val", "test"):
        a_split_meta = a_splits.get(split_name, {})
        b_split_meta = b_splits.get(split_name, {})
        a_split_hash = a_split_meta.get("sha256")
        b_split_hash = b_split_meta.get("sha256")
        if a_split_hash and b_split_hash and a_split_hash != b_split_hash:
            raise ValueError(f"Model A and Model B {split_name} split SHA-256 fingerprints differ.")

    a_split = a_splits.get("test", {})
    b_split = b_splits.get("test", {})
    a_hash, b_hash = a_split.get("sha256"), b_split.get("sha256")
    if a_split.get("size") and int(a_split["size"]) != len(merged):
        raise ValueError("Merged predictions do not match Model A test split size.")
    if b_split.get("size") and int(b_split["size"]) != len(merged):
        raise ValueError("Merged predictions do not match Model B test split size.")

    if strict_full_data:
        expected_sizes = {"train": 6_913_573, "val": 20_000, "test": 40_000}
        for split_name, expected_size in expected_sizes.items():
            if int(a_splits.get(split_name, {}).get("size", -1)) != expected_size:
                raise ValueError(f"Model A {split_name} size is not the required full-data size {expected_size:,}.")
            if int(b_splits.get(split_name, {}).get("size", -1)) != expected_size:
                raise ValueError(f"Model B {split_name} size is not the required full-data size {expected_size:,}.")
        if len(merged) != 40_000:
            raise ValueError(f"Expected 40,000 paired full-data test predictions, found {len(merged):,}.")
        expected_test_counts = {"negative": 9_231, "neutral": 3_962, "positive": 26_807}
        actual_test_counts = {LABELS[i]: int((merged["true_label_id"] == i).sum()) for i in LABEL_IDS}
        if actual_test_counts != expected_test_counts:
            raise ValueError(f"Full-data test class counts mismatch: {actual_test_counts}")

    balanced = merged.iloc[_balanced_positions(merged, seed=seed)].reset_index(drop=True)
    views = {"natural": _view_metrics(merged), "balanced": _view_metrics(balanced)}

    a_training = a_metrics.get("training", {})
    b_training = b_metrics.get("training", {})
    a_hardware = a_metrics.get("hardware", {})
    b_hardware = b_metrics.get("hardware", {})
    a_latency = a_metrics.get("gpu_latency_bf16", {}).get("bs1", {})
    b_latency = b_metrics.get("inference_latency_gpu", {}).get("bs1", {})
    models = {
        "model_a": {
            "name": "Scratch Transformer",
            "parameters": a_metrics.get("parameters", {}).get("total"),
            "training_time_seconds": a_training.get("elapsed_train_sec"),
            "epochs": a_training.get("epochs"),
            "train_size": a_split.get("size"),
            "gpu_name": a_hardware.get("gpu_name"),
            "test_raw_macro_f1": views["natural"]["model_a_raw"]["f1_macro"],
            "test_adjusted_macro_f1": views["natural"]["model_a_logit_adjusted"]["f1_macro"],
        },
        "model_b": {
            "name": "Pretrained DistilBERT",
            "parameters": b_metrics.get("model", {}).get("total_parameters"),
            "training_time_seconds": b_training.get("fine_tuning_time_seconds"),
            "epochs": b_training.get("epochs"),
            "train_size": b_split.get("size"),
            "gpu_name": b_hardware.get("gpu"),
            "test_macro_f1": views["natural"]["model_b"]["f1_macro"],
        },
    }
    same_gpu = bool(models["model_a"]["gpu_name"] and
                    models["model_a"]["gpu_name"] == models["model_b"]["gpu_name"])
    latency = {
        "same_gpu": same_gpu,
        "model_a": {"mean_ms": a_latency.get("mean_ms_per_sample"),
                    "p50_ms": a_latency.get("p50_ms_per_batch"),
                    "p95_ms": a_latency.get("p95_ms_per_batch")},
        "model_b": {"mean_ms": b_latency.get("mean_ms_per_sample"),
                    "p50_ms": b_latency.get("p50_ms_per_batch"),
                    "p95_ms": b_latency.get("p95_ms_per_batch")},
        "protocol": "Notebook GPU forward-pass benchmark, batch size 1; use only as a direct speed comparison when GPU hardware matches.",
    }
    payload = {
        "project": "Yelp 3-class sentiment: scratch Transformer vs pretrained DistilBERT",
        "primary_view": "natural",
        "description": "Primary metrics use every held-out test review. Balanced view is a deterministic, equal-per-class diagnostic subset drawn only from that same held-out test set.",
        "dataset": {"raw_reviews": 6_990_280, "deduplicated_reviews": 6_973_573,
                    "train_size": 6_913_573, "validation_size": 20_000,
                    "test_size": int(len(merged)), "seed": 42,
                    "test_split_sha256": a_hash or b_hash},
        "models": models,
        "views": views,
        "latency": latency,
        "caveats": [
            "Model A raw argmax vs Model B is the primary quality comparison. Model A validation-selected logit-bias results are shown separately.",
            "Model A was trained for 3 epochs; Model B's full-data notebook is configured for 1 epoch. Disclose the different training budgets.",
            "Training time is measured on each model's recorded hardware and is not a controlled hardware comparison.",
            "Latency is directly comparable only if both recorded GPU names match; otherwise report each measured value with its hardware.",
            "The old 150K Model B results in the teammate's original repository are archived as legacy and are not used here.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    merged.to_csv(output_dir / "paired_test_predictions.csv", index=False)

    rows = []
    for view_name, view in views.items():
        for key, title in [("model_a_raw", "Model A — raw"),
                           ("model_a_logit_adjusted", "Model A — validation-adjusted"),
                           ("model_b", "Model B — DistilBERT")]:
            m = view[key]
            rows.append({"view": view_name, "model": title, "n": view["size"],
                         "accuracy": m["accuracy"], "macro_precision": m["precision_macro"],
                         "macro_recall": m["recall_macro"], "macro_f1": m["f1_macro"],
                         "weighted_f1": m["f1_weighted"]})
    pd.DataFrame(rows).to_csv(output_dir / "model_comparison.csv", index=False)
    return payload
