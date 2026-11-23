from __future__ import annotations

import pytest
import torch

from app.models.model_a_arch import ModelAConfig, ScratchTransformerClassifier
from app.models.predictors import ModelAPredictor, ModelRegistry
from tests.conftest import make_settings, write_metrics

CPU = torch.device("cpu")


def test_missing_checkpoint_raises_without_demo(tmp_settings):
    with pytest.raises(FileNotFoundError):
        ModelAPredictor(tmp_settings, CPU)


def test_demo_random_weights_predicts(tmp_path):
    settings = make_settings(tmp_path, demo_random_weights=True)
    write_metrics(settings)
    p = ModelAPredictor(settings, CPU)
    assert p.demo_mode and p.biases_available
    [pred] = p.predict(["The food was great!"])
    probs = pred.probs.model_dump()
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert pred.label in probs and pred.label == max(probs, key=probs.get)
    assert pred.latency_ms > 0 and pred.tokenize_ms >= 0
    assert pred.demo_mode and "DEMO" in (pred.warning or "")
    assert not pred.truncated


def test_long_text_is_head_tail_truncated(tmp_path):
    settings = make_settings(tmp_path, demo_random_weights=True)
    write_metrics(settings)
    p = ModelAPredictor(settings, CPU)
    [pred] = p.predict(["word " * 600])
    assert pred.truncated and pred.num_tokens == 256 and pred.original_tokens > 256


def test_biases_change_probabilities(tmp_path):
    settings = make_settings(tmp_path, demo_random_weights=True)
    write_metrics(settings, biases=(3.0, -3.0, 0.0))
    p = ModelAPredictor(settings, CPU)
    raw = p.predict(["okay"], apply_biases=False)[0].probs
    adj = p.predict(["okay"], apply_biases=True)[0].probs
    assert adj.negative > raw.negative and adj.neutral < raw.neutral


def test_missing_biases_fall_back_to_zero_with_warning(tmp_path):
    settings = make_settings(tmp_path, demo_random_weights=True)
    p = ModelAPredictor(settings, CPU)
    assert p.biases == [0.0, 0.0, 0.0] and not p.biases_available
    assert "Logit biases unavailable" in (p.predict(["hi"])[0].warning or "")


def test_checkpoint_roundtrip_strict_load(tmp_path):
    """A checkpoint saved like the notebook (incl. a stray _orig_mod. prefix) loads strictly."""
    settings = make_settings(tmp_path)
    write_metrics(settings)
    cfg = ModelAConfig()
    torch.manual_seed(1)
    ref = ScratchTransformerClassifier(cfg).eval()
    state = {f"_orig_mod.{k}": v for k, v in ref.state_dict().items()}
    from dataclasses import asdict

    torch.save(
        {"model_state_dict": state, "model_cfg": asdict(cfg), "global_step": 1, "best_val_macro_f1": 0.5},
        settings.model_a_dir / "best_model.pt",
    )
    p = ModelAPredictor(settings, CPU)
    assert not p.demo_mode
    ids = torch.tensor([[101, 2307, 102]])
    with torch.inference_mode():
        expected = torch.softmax(ref(ids, ids != 0), dim=-1)[0].tolist()
    got = p.predict(["great"], apply_biases=False)[0].probs.model_dump()
    # "great" -> [CLS] great [SEP] == [101, 2307, 102] in bert-base-uncased
    assert [round(v, 4) for v in got.values()] == [round(v, 4) for v in expected]


def test_registry_degrades_gracefully(tmp_settings):
    reg = ModelRegistry(tmp_settings)
    reg.load_all()
    assert set(reg.errors) == {"model_a", "model_b"}
    results = reg.predict_all("hello", apply_biases=True)
    assert all(r.error for r in results)
