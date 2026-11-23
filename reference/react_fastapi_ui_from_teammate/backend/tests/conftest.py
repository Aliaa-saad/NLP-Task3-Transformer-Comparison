from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings


def make_settings(tmp_path: Path, **kw) -> Settings:
    artifacts = tmp_path / "artifacts"
    (artifacts / "model_a").mkdir(parents=True, exist_ok=True)
    (artifacts / "model_b").mkdir(parents=True, exist_ok=True)
    (artifacts / "comparison").mkdir(parents=True, exist_ok=True)
    return Settings(
        artifacts_dir=artifacts,
        frontend_dist=tmp_path / "no-dist",
        device="cpu",
        skip_warmup=True,
        _env_file=None,
        **kw,
    )


def write_metrics(settings: Settings, biases=(0.4, -0.2, 0.0)) -> None:
    (settings.model_a_dir / "metrics.json").write_text(
        json.dumps({"logit_bias_tuning": {"biases": list(biases)}}), encoding="utf-8"
    )


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)
