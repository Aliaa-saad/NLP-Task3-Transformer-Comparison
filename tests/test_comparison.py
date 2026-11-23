import json
import pandas as pd
import pytest
from src.comparison import metric_block, build_comparison


def test_metric_block_has_required_classification_metrics():
    result = metric_block(pd.Series([0, 1, 2, 0, 1, 2]).to_numpy(),
                          pd.Series([0, 1, 2, 1, 1, 2]).to_numpy())
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert "f1_macro" in result
    assert set(result["per_class"]) == {"negative", "neutral", "positive"}
    assert len(result["confusion_matrix"]["matrix"]) == 3


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_comparison_pairs_exact_indices(tmp_path):
    a_dir = tmp_path / "model_a"
    b_dir = tmp_path / "model_b"
    out = tmp_path / "comparison"
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    a = pd.DataFrame({"record_index": range(9), "true_label_id": labels,
                      "model_a_raw_pred": labels, "model_a_adjusted_pred": labels})
    b = pd.DataFrame({"record_index": range(9), "true_label_id": labels,
                      "predicted_label_id": labels})
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)
    a.to_csv(a_dir / "test_predictions.csv", index=False)
    b.to_csv(b_dir / "test_predictions.csv", index=False)
    _write_json(a_dir / "metrics.json", {
        "split_manifest": {"splits": {"test": {"size": 9, "sha256": "same"}}},
        "parameters": {"total": 100}, "training": {"epochs": 3, "elapsed_train_sec": 10},
        "hardware": {"gpu_name": "A100"},
        "gpu_latency_bf16": {"bs1": {"mean_ms_per_sample": 1.0}},
    })
    _write_json(b_dir / "metrics.json", {
        "splits": {"test": {"size": 9, "sha256": "same"}},
        "model": {"total_parameters": 200}, "training": {"epochs": 1, "fine_tuning_time_seconds": 5},
        "hardware": {"gpu": "A100"},
        "inference_latency_gpu": {"bs1": {"mean_ms_per_sample": 2.0}},
    })
    result = build_comparison(tmp_path, out, seed=42, strict_full_data=False)
    assert result["dataset"]["test_size"] == 9
    assert result["views"]["natural"]["model_a_raw"]["accuracy"] == 1.0
    assert result["views"]["natural"]["model_b"]["f1_macro"] == 1.0
    assert result["latency"]["same_gpu"] is True
    assert (out / "comparison_results.json").is_file()


def test_build_comparison_rejects_different_test_indices(tmp_path):
    a_dir = tmp_path / "model_a"
    b_dir = tmp_path / "model_b"
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)
    pd.DataFrame({"record_index": [1, 2, 3], "true_label_id": [0, 1, 2],
                  "model_a_raw_pred": [0, 1, 2], "model_a_adjusted_pred": [0, 1, 2]}).to_csv(a_dir / "test_predictions.csv", index=False)
    pd.DataFrame({"record_index": [1, 2, 4], "true_label_id": [0, 1, 2],
                  "predicted_label_id": [0, 1, 2]}).to_csv(b_dir / "test_predictions.csv", index=False)
    _write_json(a_dir / "metrics.json", {"split_manifest": {"splits": {"test": {"size": 3}}}})
    _write_json(b_dir / "metrics.json", {"splits": {"test": {"size": 3}}})
    with pytest.raises(ValueError, match="test indices differ"):
        build_comparison(tmp_path, tmp_path / "out", strict_full_data=False)
