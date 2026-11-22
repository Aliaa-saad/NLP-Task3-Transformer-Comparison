# NLP Task 3 — Yelp Sentiment Analysis: Transformer Comparison

This repository contains the shared Yelp dataset preparation notebook and the pretrained DistilBERT implementation. The scratch Transformer implementation and Streamlit comparison app are to be added by the other teammates.

## Notebooks

1. `notebooks/00_dataset_preparation.ipynb` — shared Yelp data preparation pipeline.
2. `notebooks/01_pretrained_transformer.ipynb` — pretrained DistilBERT fine-tuning, evaluation, and prediction.

## Shared dataset

Source: [Yelp Open Dataset on Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)

Recorded preparation setup:
- Balanced sample: 150,000 reviews (50,000 per class)
- Negative = 0; Neutral = 1; Positive = 2
- Train = 120,000; Validation = 15,000; Test = 15,000
- Random seed = 42

Both model implementations must use the same three saved splits from the team's shared Google Drive folder:

`MyDrive/NLP_Task3/dataset/processed/`

See `docs/DATASET_AND_RUN_GUIDE.md`.

## Pretrained model

Checkpoint: `distilbert-base-uncased`, fine-tuned for three-class sentiment classification.

Recorded run metrics:
- Test accuracy: 0.8365
- Macro precision: 0.8375
- Macro recall: 0.8365
- Macro-F1: 0.8369
- Average single-review inference latency: 6.34 ms
- Median single-review inference latency: 6.06 ms
- Total parameters: 66,955,779
- Training time: 32.56 minutes on Tesla T4

These are recorded measurements from the notebook run and may vary if training is repeated.

See `docs/PRETRAINED_HANDOFF.md` for the completed work and the next tasks for the teammate.

## Run instructions

- Use Google Colab with access to the shared Drive folder.
- If the processed split files already exist in Drive, do not rerun the full data preparation pipeline.
- To use the saved pretrained model, load the checkpoint from the shared Drive folder. Do not run the clean initialization and fine-tuning cells unless a new training run is intended.
- The scratch model must use the exact same train/validation/test split files.

## Security and large files

Do not commit:
- `kaggle.json`, API tokens, `.env`, or Streamlit secrets
- Raw Yelp JSON files or processed CSV/CSV.GZ splits
- Model weights such as `.safetensors` or `pytorch_model.bin`
- `pretrained_test_predictions.csv.gz` (contains review text)

Keep those files in the team's shared Google Drive folder. Public GitHub should contain code, notebooks, documentation, and safe summary results only.
