from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.inference import load_model_a, load_model_b, predict_model_a, predict_model_b, verify_tokenizer_compatibility

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
COMPARISON_PATH = ARTIFACTS / "comparison" / "comparison_results.json"
LABEL_ORDER = ["negative", "neutral", "positive"]
EXAMPLES = {
    "Positive review": "The food was delicious, the staff were friendly, and the service was excellent. I would definitely come back!",
    "Neutral review": "The place was okay. The food was average and the service was neither good nor bad.",
    "Negative review": "The service was extremely slow, the food arrived cold, and nobody helped us. I would not recommend it.",
}

st.set_page_config(page_title="Yelp Transformer Comparison", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1450px;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid rgba(120,130,150,.25); border-radius: 18px;
       background: linear-gradient(120deg, rgba(35,45,65,.10), rgba(30,190,120,.08)); margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size:2.35rem; letter-spacing:-.04em;}
.hero p {margin:.5rem 0 0 0; opacity:.78;}
.model-card {padding: 1rem 1.1rem; border:1px solid rgba(120,130,150,.25); border-radius:14px;}
.small-muted {font-size:.86rem; opacity:.72;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the trained models…")
def load_runtime():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    errors = {}
    try:
        models["model_a"] = load_model_a(ARTIFACTS, device)
    except Exception as exc:  # show missing artifacts without crashing the dashboard
        errors["model_a"] = f"{type(exc).__name__}: {exc}"
    try:
        models["model_b"] = load_model_b(ARTIFACTS, device)
    except Exception as exc:
        errors["model_b"] = f"{type(exc).__name__}: {exc}"
    if "model_a" in models and "model_b" in models:
        try:
            verify_tokenizer_compatibility(models["model_a"], models["model_b"])
        except Exception as exc:
            errors["tokenizer_compatibility"] = str(exc)
    return device, models, errors


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def metric_card(label: str, value: str, help_text: str | None = None):
    st.metric(label, value, help=help_text)


def show_prediction(col, result: dict, model_color: str):
    with col:
        st.markdown(f"<div class='model-card'><b>{result['model']}</b></div>", unsafe_allow_html=True)
        st.markdown(f"### :{model_color}[{result['label'].title()}]")
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.caption(
            f"Model forward pass: {result['model_latency_ms']:.2f} ms · "
            f"End-to-end: {result['end_to_end_latency_ms']:.2f} ms · "
            f"Tokens: {result['token_count']}"
        )
        probs = pd.DataFrame({"Sentiment": LABEL_ORDER,
                              "Probability": [result["probabilities"][x] for x in LABEL_ORDER]})
        fig = px.bar(probs, x="Sentiment", y="Probability", range_y=[0, 1],
                     color="Sentiment", color_discrete_map={
                         "negative": "#e36b70", "neutral": "#e4b34e", "positive": "#36b78a"})
        fig.update_layout(height=230, margin=dict(l=5, r=5, t=8, b=5), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def show_confusion_matrix(block: dict, title: str):
    cm = block["confusion_matrix"]
    labels = cm["labels"]
    matrix = cm["matrix"]
    fig = go.Figure(data=go.Heatmap(z=matrix, x=labels, y=labels,
                                    colorscale="Blues", text=matrix, texttemplate="%{text}",
                                    hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Predicted label", yaxis_title="True label",
                      height=360, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


st.markdown("""
<div class="hero">
  <h1>Two Transformers. One Yelp review.</h1>
  <p>Three-class sentiment analysis · Scratch Transformer vs. pretrained DistilBERT · Same full Yelp experiment.</p>
</div>
""", unsafe_allow_html=True)

device, models, errors = load_runtime()
comparison = read_json(COMPARISON_PATH)

with st.sidebar:
    st.title("Project Navigation")
    st.caption("NLP Task 3 · Transformer Comparison")
    st.markdown("---")
    st.write(f"**Inference device:** `{device}`")
    for key, label in [("model_a", "Model A — Scratch"), ("model_b", "Model B — DistilBERT")]:
        if key in models:
            st.success(f"{label}: loaded")
        else:
            st.error(f"{label}: not loaded")
    if errors.get("tokenizer_compatibility"):
        st.error("Tokenizer compatibility check failed")
    st.markdown("---")
    st.caption("Only measured experiment artifacts are shown. Missing training outputs are never replaced with demo metrics.")

tab_analyze, tab_compare, tab_setup = st.tabs(["🔎 Analyze a Review", "📊 Test-set Comparison", "🧪 Experiment Setup"])

with tab_analyze:
    st.subheader("Run both models on the same review")
    st.write("Enter a Yelp review or choose an example. Each model returns its sentiment, class probabilities, and measured local inference time.")
    example_choice = st.selectbox("Example reviews", ["Choose an example…", *EXAMPLES.keys()])
    initial_text = EXAMPLES.get(example_choice, "")
    review_text = st.text_area("Review text", value=initial_text, height=150,
                               placeholder="Paste a customer review here…", max_chars=5000)
    apply_biases = st.checkbox("Apply Model A's validation-selected logit bias (secondary inference option)", value=False,
                               help="Off by default so the main live comparison uses raw argmax for both models.")
    run = st.button("Compare sentiment", type="primary", use_container_width=True)
    if run:
        if not review_text.strip():
            st.warning("Please enter a review before running inference.")
        elif len(review_text.strip()) > 5000:
            st.warning("The review exceeds the 5,000-character limit.")
        elif errors.get("tokenizer_compatibility"):
            st.error(f"The two tokenizers failed compatibility verification: {errors['tokenizer_compatibility']}")
        else:
            results = {}
            if "model_a" in models:
                try:
                    results["model_a"] = predict_model_a(models["model_a"], review_text, device, apply_biases)
                except Exception as exc:
                    st.error(f"Model A inference failed: {exc}")
            if "model_b" in models:
                try:
                    results["model_b"] = predict_model_b(models["model_b"], review_text, device)
                except Exception as exc:
                    st.error(f"Model B inference failed: {exc}")
            if results:
                cols = st.columns(2)
                if "model_a" in results:
                    show_prediction(cols[0], results["model_a"], "blue")
                else:
                    cols[0].error(f"Model A unavailable: {errors.get('model_a', 'not loaded')}")
                if "model_b" in results:
                    show_prediction(cols[1], results["model_b"], "green")
                else:
                    cols[1].error(f"Model B unavailable: {errors.get('model_b', 'not loaded')}")
                if len(results) == 2:
                    st.info("Both models agree." if results["model_a"]["label"] == results["model_b"]["label"]
                            else "The models disagree on this review. Compare the probability distributions above.")
    if errors:
        with st.expander("Model loading details", expanded=not models):
            for key, value in errors.items():
                st.error(f"{key}: {value}")

with tab_compare:
    st.subheader("Measured held-out test results")
    if comparison is None:
        st.warning("The final paired comparison has not been generated yet. Finish Model B training, export both prediction files, then run `python scripts/build_comparison.py`.")
        st.code("python scripts/check_artifacts.py\npython scripts/build_comparison.py", language="powershell")
    else:
        natural = comparison["views"]["natural"]
        balanced = comparison["views"]["balanced"]
        view_choice = st.radio("Evaluation view", ["Natural full test set", "Balanced diagnostic subset"], horizontal=True)
        view = natural if view_choice.startswith("Natural") else balanced
        st.caption(comparison.get("description", ""))
        st.caption(f"Evaluated examples: {view['size']:,} · class counts: {view['class_counts']}")
        a = view["model_a_raw"]
        b = view["model_b"]
        cols = st.columns(4)
        for col, label, value in [
            (cols[0], "Model A accuracy", f"{a['accuracy']:.2%}"),
            (cols[1], "Model B accuracy", f"{b['accuracy']:.2%}"),
            (cols[2], "Model A Macro-F1", f"{a['f1_macro']:.4f}"),
            (cols[3], "Model B Macro-F1", f"{b['f1_macro']:.4f}"),
        ]:
            with col:
                st.metric(label, value)

        rows = []
        for key, name in [("model_a_raw", "Model A — raw"),
                          ("model_a_logit_adjusted", "Model A — validation-adjusted"),
                          ("model_b", "Model B — DistilBERT")]:
            m = view[key]
            rows.append({"Model": name, "Accuracy": m["accuracy"],
                         "Macro Precision": m["precision_macro"], "Macro Recall": m["recall_macro"],
                         "Macro-F1": m["f1_macro"], "Weighted-F1": m["f1_weighted"]})
        metric_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(metric_df.style.format("{:.4f}"), use_container_width=True)
        chart_df = metric_df.reset_index().melt(id_vars="Model", value_vars=["Accuracy", "Macro-F1", "Weighted-F1"],
                                                var_name="Metric", value_name="Score")
        chart = px.bar(chart_df, x="Metric", y="Score", color="Model", barmode="group", range_y=[0, 1],
                       color_discrete_map={"Model A — raw": "#4f8bd6", "Model A — validation-adjusted": "#8c7ad8", "Model B — DistilBERT": "#35ae86"})
        chart.update_layout(height=330, margin=dict(l=5, r=5, t=25, b=5))
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
        with st.expander("Per-class precision, recall and F1"):
            per_class_rows = []
            for model_key, model_title in [("model_a_raw", "Model A — raw"),
                                           ("model_a_logit_adjusted", "Model A — validation-adjusted"),
                                           ("model_b", "Model B — DistilBERT")]:
                for sentiment, values in view[model_key]["per_class"].items():
                    per_class_rows.append({"Model": model_title, "Class": sentiment,
                                           "Precision": values["precision"], "Recall": values["recall"],
                                           "F1": values["f1"], "Support": values["support"]})
            st.dataframe(pd.DataFrame(per_class_rows).style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
        cm1, cm2 = st.columns(2)
        with cm1:
            show_confusion_matrix(view["model_a_raw"], "Model A — raw confusion matrix")
        with cm2:
            show_confusion_matrix(view["model_b"], "Model B — DistilBERT confusion matrix")
        st.markdown("#### Paired bootstrap: Macro-F1 difference")
        boot = view["bootstrap_a_raw_minus_b"]
        st.write(f"Model A raw minus Model B: **{boot['a_minus_b_mean']:+.4f}** · 95% CI [{boot['ci95_low']:+.4f}, {boot['ci95_high']:+.4f}]")
        delta_b_minus_a = b["f1_macro"] - a["f1_macro"]
        if delta_b_minus_a > 0:
            st.info(f"In this measured run, Model B's Macro-F1 is higher by {delta_b_minus_a:.4f} on the natural test set.")
        elif delta_b_minus_a < 0:
            st.info(f"In this measured run, Model A raw Macro-F1 is higher by {-delta_b_minus_a:.4f} on the natural test set.")
        else:
            st.info("In this measured run, both models have equal Macro-F1 on the natural test set.")
        st.caption("This is a measured outcome, not proof that pretraining alone caused the difference. Positive bootstrap values favor Model A; negative values favor Model B.")
        st.markdown("#### Efficiency and training details")
        model_info = comparison["models"]
        latency = comparison["latency"]
        efficiency_rows = []
        for key, label in [("model_a", "Model A — Scratch"), ("model_b", "Model B — DistilBERT")]:
            info = model_info[key]
            efficiency_rows.append({"Model": label, "Parameters": info.get("parameters"),
                                   "Training time (minutes)": (info.get("training_time_seconds") or 0) / 60,
                                   "Epochs": info.get("epochs"), "Training GPU": info.get("gpu_name")})
        st.dataframe(pd.DataFrame(efficiency_rows).set_index("Model"), use_container_width=True)
        st.write("**Single-review forward-pass latency:**")
        st.json(latency)
        if not latency.get("same_gpu"):
            st.warning("Latency runs used different GPU models. Report the measured values with their hardware; do not claim a controlled speed winner.")
        for caveat in comparison.get("caveats", []):
            st.caption(f"• {caveat}")

with tab_setup:
    st.subheader("Dataset and experiment protocol")
    st.markdown("""
    **Task:** Yelp review sentiment classification (negative / neutral / positive).

    | Yelp stars | Label ID | Sentiment |
    |---|---:|---|
    | 1–2 | 0 | Negative |
    | 3 | 1 | Neutral |
    | 4–5 | 2 | Positive |

    **Shared full-data experiment:** 6,990,280 raw reviews; 6,973,573 valid unique reviews after removing 16,707 duplicates. Split seed 42: 6,913,573 train, 20,000 validation, and 40,000 test.

    **Model A:** 8-layer Pre-LN encoder-only Transformer, manually implemented attention and feed-forward blocks, random initialization, 512 hidden size, 8 heads, 2,048 FFN size, 256 tokens (192 head + 64 tail), 3 epochs.

    **Model B:** Hugging Face `distilbert-base-uncased` with a 3-class classification head, fine-tuned on the same train split. The current notebook is configured for one full epoch. This training-budget difference must be disclosed.

    **Batching note:** Model A uses fixed-size batches with `drop_last=True`, omitting 37 training rows from each epoch schedule; Model B keeps its final partial batch. The split membership and held-out test set remain identical.
    """)
    st.markdown("#### Artifact readiness")
    model_a_pt = ARTIFACTS / "model_a/best_model.pt"
    model_a_onnx = next((p for p in [ARTIFACTS / "model_a/model_a.onnx", ARTIFACTS / "model_a/onnx/model_a.onnx", ROOT / "model_a.onnx"] if p.is_file()), None)
    model_a_tok = next((p for p in [ARTIFACTS / "model_a/tokenizer/vocab.txt", ARTIFACTS / "model_a/tokenizer/tokenizer.json", ROOT / "tokenizer.json"] if p.is_file()), None)
    expected = [
        ("Model A weights", "Ready (PyTorch)" if model_a_pt.is_file() else ("Ready (ONNX)" if model_a_onnx else "Missing"), str((model_a_pt if model_a_pt.is_file() else (model_a_onnx or model_a_pt)).relative_to(ROOT))),
        ("Model A metrics", "Ready" if (ARTIFACTS / "model_a/metrics.json").is_file() else "Missing", "artifacts/model_a/metrics.json"),
        ("Model A tokenizer", "Ready" if model_a_tok else "Missing", str((model_a_tok or (ARTIFACTS / "model_a/tokenizer/vocab.txt")).relative_to(ROOT))),
        ("Model B checkpoint", "Ready" if (ARTIFACTS / "model_b/model/model.safetensors").is_file() else "Missing", "artifacts/model_b/model/model.safetensors"),
        ("Model B metrics", "Ready" if (ARTIFACTS / "model_b/metrics.json").is_file() else "Missing", "artifacts/model_b/metrics.json"),
        ("Paired comparison", "Ready" if COMPARISON_PATH.is_file() else "Missing", str(COMPARISON_PATH.relative_to(ROOT))),
    ]
    st.table(pd.DataFrame([{"Artifact": k, "Status": status, "Path": path} for k, status, path in expected]))
    st.markdown("#### Reproducibility")
    if comparison:
        st.code(f"Test split SHA-256: {comparison['dataset'].get('test_split_sha256')}\nRandom seed: {comparison['dataset'].get('seed')}")
    else:
        st.info("The comparison JSON will include the verified test-split SHA-256 after both model prediction exports are present.")

st.markdown("---")
st.caption("NLP Task 3 · Yelp Dataset · Model A: scratch Transformer · Model B: pretrained DistilBERT")
