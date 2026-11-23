from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.comparison import build_comparison


def pct(value):
    return f"{value:.4%}"


def build_report(payload):
    natural = payload["views"]["natural"]
    models = payload["models"]
    latency = payload["latency"]
    rows = [
        ("Model A — Scratch (raw)", natural["model_a_raw"]),
        ("Model A — Scratch (validation-adjusted)", natural["model_a_logit_adjusted"]),
        ("Model B — DistilBERT", natural["model_b"]),
    ]
    lines = [
        "# Final Measured Comparison — Yelp Sentiment",
        "",
        f"**Primary evaluation:** all {payload['dataset']['test_size']:,} held-out test reviews.",
        f"**Test split SHA-256:** `{payload['dataset'].get('test_split_sha256')}`",
        "",
        "| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metric in rows:
        lines.append("| " + name + " | " + " | ".join([
            pct(metric["accuracy"]), pct(metric["precision_macro"]),
            pct(metric["recall_macro"]), pct(metric["f1_macro"]),
            pct(metric["f1_weighted"])
        ]) + " |")
    lines += [
        "",
        "## Experiment cost",
        "",
        "| Model | Parameters | Training time (min) | Epochs | GPU |",
        "|---|---:|---:|---:|---|",
    ]
    for key, title in [("model_a", "Model A — Scratch"), ("model_b", "Model B — DistilBERT")]:
        m = models[key]
        t = m.get("training_time_seconds")
        params = f"{int(m['parameters']):,}" if m.get("parameters") is not None else "not recorded"
        duration = f"{t/60:.2f}" if t is not None else "not recorded"
        lines.append(f"| {title} | {params} | {duration} | {m.get('epochs')} | {m.get('gpu_name')} |")
    lines += [
        "",
        "## Single-review inference latency",
        "",
        f"- Model A mean: {latency['model_a'].get('mean_ms')} ms",
        f"- Model B mean: {latency['model_b'].get('mean_ms')} ms",
        f"- Same GPU recorded: **{latency.get('same_gpu')}**",
        f"- Protocol: {latency.get('protocol')}",
        "",
        "## Measured interpretation",
        "",
        (f"On the natural held-out test set, Model B Macro-F1 is {natural['model_b']['f1_macro']:.4f} "
         f"and Model A raw Macro-F1 is {natural['model_a_raw']['f1_macro']:.4f} "
         f"(B minus A = {natural['model_b']['f1_macro'] - natural['model_a_raw']['f1_macro']:+.4f})."),
        (f"Model B accuracy is {natural['model_b']['accuracy']:.4f}; Model A raw accuracy is "
         f"{natural['model_a_raw']['accuracy']:.4f}. These are measured outcomes for the supplied runs, "
         "not a claim that pretraining alone caused the difference."),
        "",
        "## Paired uncertainty",
        "",
        f"Model A raw minus Model B Macro-F1: {natural['bootstrap_a_raw_minus_b']['a_minus_b_mean']:+.5f}; "
        f"95% bootstrap interval [{natural['bootstrap_a_raw_minus_b']['ci95_low']:+.5f}, "
        f"{natural['bootstrap_a_raw_minus_b']['ci95_high']:+.5f}].",
        "",
        "## Balanced diagnostic view",
        "",
        f"The balanced diagnostic view contains {payload['views']['balanced']['size']:,} held-out test reviews "
        f"({payload['views']['balanced']['class_counts']}). It is secondary; the natural full test set above is primary.",
        "",
        "## Caveats and limitations",
        "",
    ]
    lines += [f"- {c}" for c in payload.get("caveats", [])]
    lines += [
        "- Interpret all reported scores as measurements from the exported checkpoints and prediction files, not expected or illustrative values.",
        "- Per-class precision, recall, F1, and confusion matrices are available in `artifacts/comparison/comparison_results.json` and the Streamlit dashboard.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    payload = build_comparison(ROOT / "artifacts", ROOT / "artifacts" / "comparison")
    natural = payload["views"]["natural"]
    report = build_report(payload)
    report_path = ROOT / "reports" / "final_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print("Paired held-out test comparison created.")
    print("Test examples:", payload["dataset"]["test_size"])
    print("Model A raw Macro-F1:", round(natural["model_a_raw"]["f1_macro"], 5))
    print("Model A adjusted Macro-F1:", round(natural["model_a_logit_adjusted"]["f1_macro"], 5))
    print("Model B Macro-F1:", round(natural["model_b"]["f1_macro"], 5))
    print("Report:", report_path)
