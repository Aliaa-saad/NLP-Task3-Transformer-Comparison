# NLP Task 3 — Yelp Sentiment: Scratch Transformer vs. Pretrained DistilBERT

This is the consolidated final-project workspace for the revised course brief. It keeps the two full-Yelp training notebooks, adds a paired test-set comparison pipeline, and provides the required local Streamlit application.

> **Important status:** the supplied Model A notebook contains completed training/evaluation outputs. The supplied Model B full-data notebook is a training workflow, but its uploaded copy has no completed full-data training outputs. The old 150K Model B results from the teammate's original repository are preserved under `reference/legacy_150k_results/` only and are **not** used as final results. Do not report Model B performance until the full-data notebook finishes and exports its artifacts.

## Final experiment design

| Item | Model A | Model B |
|---|---|---|
| Approach | Encoder-only Transformer built from scratch | Hugging Face `distilbert-base-uncased` fine-tuning |
| Initialization | Random weights | Pretrained weights + new 3-class head |
| Architecture | 8 layers, 8 heads, `d_model=512`, `d_ff=2048` | DistilBERT encoder |
| Max sequence length | 256 | 256 |
| Truncation | First 192 + last 64 tokens | First 192 + last 64 tokens |
| Training split | Same full-data split | Same full-data split |
| Training budget | 3 epochs | 1 epoch by default |

The different epoch counts are intentional settings in the supplied notebooks and must be disclosed. Model A uses `drop_last=True` for fixed-size training micro-batches, so 37 examples from the 6,913,573-row train split are omitted from each epoch schedule; Model B keeps the final partial batch. Both still share the exact same train/validation/test split membership and the exact same held-out test set. The comparison script verifies that both prediction files contain the same unique `record_index` values and true labels. It also compares the test-split SHA-256 fingerprints in the metrics files when available.

## Dataset

- Dataset: [Yelp Open Dataset on Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)
- File: `yelp_academic_dataset_review.json`
- Raw reviews counted by Model A: **6,990,280**
- Invalid records removed: **0**
- Normalized-text duplicates removed: **16,707**
- Valid unique reviews: **6,973,573**
- Seed: **42**
- Train: **6,913,573**; validation: **20,000**; test: **40,000**

Labels: 1–2 stars = negative (0), 3 stars = neutral (1), 4–5 stars = positive (2).

The Yelp dataset is subject to the Yelp Dataset Agreement. Review current terms and use the dataset only for permitted purposes. Do not upload raw reviews, the multi-GB tokenized cache, Kaggle credentials, or review-text prediction exports to a public repository.

## Project layout

```text
notebooks/
  01_model_a_scratch_full_yelp.ipynb
  02_model_b_distilbert_full_yelp.ipynb
  03_export_final_artifacts.ipynb
src/
  model_a_arch.py
  inference.py
  comparison.py
scripts/
  check_artifacts.py
  install_artifacts.py
  build_comparison.py
app.py                         # required Streamlit UI
artifacts/                     # local-only model weights and measured outputs
reports/
reference/                     # original teammate React/FastAPI UI and legacy 150K work
```

## Step 1 — Finish the training artifacts in Google Colab

1. Open `notebooks/01_model_a_scratch_full_yelp.ipynb` in Colab and connect the Google Drive that contains `yelp_model_a/`. The notebook already shows completed A training outputs in the supplied copy. If the checkpoint is already present, do not train again.
2. Run the final evaluation cell (section 5) and the new **paired test prediction export** cell immediately after it. It creates `yelp_model_a/artifacts_v2/test_predictions.csv` without storing review text.
3. Open `notebooks/02_model_b_distilbert_full_yelp.ipynb`. The shared Model A cache must be accessible at `MyDrive/yelp_model_a/tokenized/`. If the folder belongs to a teammate, use Drive → Shared with me → right-click the shared `yelp_model_a` folder → Organize → Add shortcut → My Drive, keeping the name `yelp_model_a`.
4. Select a CUDA GPU. A100 + High-RAM is recommended. If the tokenized cache is missing, this notebook intentionally stops on low system RAM rather than trying to rebuild the 6.9M-review cache unsafely. Do not substitute the old 150K balanced splits.
5. Run Model B cells in order. Confirm all three Train/Validation/Test SHA-256 checks say `PASS`, let the full training epoch complete, and run final test evaluation/export cells. Checkpointing supports resume after a Colab disconnect.
6. Run `notebooks/03_export_final_artifacts.ipynb`. It bundles model weights, tokenizers, metrics, histories, and text-free test predictions into `NLP_Task3_Final_Artifacts.zip`. It deliberately excludes raw data, the tokenized cache, and Kaggle credentials.

## Step 2 — Install artifacts locally (Windows)

1. Download and extract this project ZIP.
2. Download `NLP_Task3_Final_Artifacts.zip` created by the export notebook and put it beside this README.
3. Open PowerShell in the project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/install_artifacts.py NLP_Task3_Final_Artifacts.zip
python scripts/check_artifacts.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again. Install the PyTorch build appropriate for your machine if you want CUDA inference; CPU inference is supported but slower.

## Step 3 — Build the final paired comparison

```powershell
python scripts/build_comparison.py
```

This command fails closed if the paired prediction files, metrics, labels, or test indices do not match. It writes:

- `artifacts/comparison/comparison_results.json`
- `artifacts/comparison/paired_test_predictions.csv`
- `artifacts/comparison/model_comparison.csv`
- `reports/final_comparison_report.md`

Primary quality comparison: Model A raw argmax vs. Model B on all 40,000 held-out test reviews. Model A validation-selected logit-bias metrics are shown separately as a secondary result. A balanced equal-per-class view is also reported as a diagnostic subset drawn from the same held-out test set.

## Step 4 — Run the required Streamlit app

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal (normally `http://localhost:8501`). The app provides:

- **Analyze a Review:** sends one review to both models and displays predicted sentiment, class probabilities, and measured local latency.
- **Test-set Comparison:** displays the measured side-by-side metrics, confusion matrices, paired bootstrap interval, parameter counts, training time, and latency.
- **Experiment Setup:** displays the dataset, split sizes, label mapping, and artifact readiness.

If a model artifact is missing, the app reports which file is missing; it never silently substitutes random weights or legacy 150K metrics. After adding artifacts, restart Streamlit to clear its model cache.

## Required evaluation and reporting

The comparison reports accuracy, macro precision, macro recall, macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrices, parameter counts, training time, and recorded inference latency. Latency is a direct speed comparison only when both notebooks recorded the same GPU type and benchmark protocol. Training time is measured on each run's hardware and is not a controlled hardware comparison.

The final report must use measured Model B results from the completed full-data run. Do not copy the old 150K Model B metrics in `reference/legacy_150k_results/` into the final report.

## Troubleshooting

- **Model A cache not found in Model B:** the Drive shortcut/share is missing or not mounted at the expected `MyDrive/yelp_model_a/tokenized` path. Do not rebuild the cache on a 12.7 GB RAM runtime.
- **Model B GPU error:** select Runtime → Change runtime type → GPU. The notebook requires a CUDA GPU with at least 12 GB VRAM; A100 is recommended.
- **Missing Model A test predictions:** rerun Model A evaluation cell and the paired test prediction export cell; no retraining is needed if the completed checkpoint is present.
- **Comparison refuses to run:** inspect the first error. It is designed to stop if the two prediction files are not paired on identical test indices and labels.
- **App says model unavailable:** install the artifact bundle and rerun `python scripts/check_artifacts.py`.

## References

- Course brief: `docs/Transformer_Comparison_Project_Requirements_Revised.pdf`
- Transformer lecture: `docs/Transformer.pdf`
- Original teammate React/FastAPI UI and earlier 150K experiment: `reference/` (preserved for reference, not the final result source).
