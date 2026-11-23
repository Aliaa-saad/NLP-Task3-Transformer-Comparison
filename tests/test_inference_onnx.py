import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference import load_model_a, predict_model_a


class FakeSession:
    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)

    def run(self, output_names, feed_dict):
        return [self.output]


def test_predict_model_a_onnx_unbakes_and_applies_biases():
    class StubTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [101, 2000, 102]}

    from src.model_a_arch import ModelAConfig

    baked_logits = [[1.0 + 0.5, 2.0 + 1.0, 3.0 + 0.0]]
    bundle = {
        "backend": "onnx",
        "session": FakeSession(baked_logits),
        "tokenizer": StubTokenizer(),
        "config": ModelAConfig(),
        "biases": [0.5, 1.0, 0.0],
        "biases_baked_in": True,
    }
    raw_res = predict_model_a(bundle, "test review", torch.device("cpu"), apply_validation_biases=False)
    adj_res = predict_model_a(bundle, "test review", torch.device("cpu"), apply_validation_biases=True)

    assert raw_res["validation_biases_applied"] is False
    assert adj_res["validation_biases_applied"] is True
    assert raw_res["probabilities"]["positive"] > adj_res["probabilities"]["positive"]


def test_load_model_a_missing_raises(tmp_path):
    (tmp_path / "model_a").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_model_a(tmp_path, torch.device("cpu"))


def test_load_model_a_from_onnx_when_present():
    root = Path(__file__).resolve().parents[1]
    if not (root / "model_a.onnx").is_file():
        pytest.skip("model_a.onnx not present in workspace root")
    bundle = load_model_a(root / "artifacts", torch.device("cpu"))
    assert bundle["backend"] == "onnx"
    assert bundle["biases_baked_in"] is True
    assert len(bundle["biases"]) == 3
    res = predict_model_a(bundle, "The food was amazing and the staff were friendly!", torch.device("cpu"), False)
    assert res["label"] == "positive"
    assert res["confidence"] > 0.8
