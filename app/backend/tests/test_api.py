from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_settings, write_metrics


def _metrics_obj(acc: float) -> dict:
    per = {k: {"precision": acc, "recall": acc, "f1": acc, "support": 10} for k in ("negative", "neutral", "positive")}
    return {
        "accuracy": acc, "precision_macro": acc, "recall_macro": acc, "f1_macro": acc, "f1_weighted": acc,
        "per_class": per,
        "confusion_matrix": {"labels": ["negative", "neutral", "positive"], "matrix": [[8, 1, 1], [1, 8, 1], [1, 1, 8]]},
    }


def _comparison() -> dict:
    view = {
        "size": 30,
        "model_a_raw": _metrics_obj(0.7),
        "model_a_logit_adjusted": _metrics_obj(0.72),
        "model_b": _metrics_obj(0.8),
        "bootstrap_a_adjusted_vs_b": {"macro_f1_diff_a_minus_b": -0.08, "ci95_low": -0.1, "ci95_high": -0.06, "p_a_better": 0.0},
        "agreement_rate_a_adjusted_vs_b": 0.85,
    }
    return {
        "description": "test", "caveat": "caveat text", "eval_slice": {"natural_size": 30, "balanced_size": 30},
        "models": {"model_a": {"name": "A"}, "model_b": {"name": "B"}},
        "views": {"natural": view, "balanced": view},
        "latency_same_machine": {"device": "cpu", "model_a": {"mean_ms": 5, "median_ms": 4, "p95_ms": 8, "n": 10}},
    }


def test_server_starts_without_artifacts(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        h = client.get("/api/health").json()
        assert h["status"] == "degraded"
        assert {m["status"] for m in h["models"]} == {"error"}

        r = client.post("/api/predict", json={"text": "Great food"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 2 and all(x["error"] for x in body["results"])
        assert body["agree"] is None

        assert client.get("/api/metrics").status_code == 404
        assert client.get("/api/history/model_a").status_code == 404
        assert client.get("/api/errors/model_a").status_code == 404
        assert len(client.get("/api/models").json()) == 2


def test_validation(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.post("/api/predict", json={"text": "   "}).status_code == 422
        r = client.post("/api/predict", json={"text": "a" * 5001})
        assert r.status_code == 422 and "5,000" in r.json()["detail"]
        assert client.post("/api/predict", json={}).status_code == 422


def test_demo_mode_model_a_and_metrics(tmp_path):
    settings = make_settings(tmp_path, demo_random_weights=True)
    write_metrics(settings)
    settings.comparison_path.write_text(json.dumps(_comparison()), encoding="utf-8")
    (settings.model_a_dir / "training_history.csv").write_text(
        "step,epoch,train_loss,val_loss,val_macro_f1,val_accuracy,lr\n2000,1,0.9,0.8,0.6,0.7,0.0005\n4000,1,0.7,,0.65,0.72,0.0004\n",
        encoding="utf-8",
    )
    (settings.model_a_dir / "error_analysis_top50.csv").write_text(
        "rank,text_or_index,true_label_id,true_label,pred_label_id,pred_label,confidence,prob_negative,prob_neutral,prob_positive\n"
        '1,"Loved it, ""not""",0,negative,2,positive,0.99,0.005,0.005,0.99\n',
        encoding="utf-8",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        h = client.get("/api/health").json()
        a = next(m for m in h["models"] if m["model_key"] == "model_a")
        assert a["status"] == "loaded" and a["demo_mode"]

        body = client.post("/api/predict", json={"text": "Amazing!", "apply_biases": False}).json()
        res_a = next(x for x in body["results"] if x["model_key"] == "model_a")
        assert res_a["demo_mode"] and res_a["biases_applied"] is False and res_a["latency_ms"] > 0

        m = client.get("/api/metrics").json()
        assert m["comparison"]["views"]["balanced"]["model_b"]["f1_macro"] == 0.8
        hist = client.get("/api/history/model_a").json()
        assert len(hist) == 2 and hist[1]["val_loss"] is None
        errs = client.get("/api/errors/model_a?limit=5").json()
        assert errs[0]["text"] == 'Loved it, "not"' and errs[0]["pred_label"] == "positive"
