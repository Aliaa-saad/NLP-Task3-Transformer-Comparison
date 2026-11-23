# Dataset and Run Guide

## Shared dataset pipeline

`notebooks/00_dataset_preparation.ipynb` contains the team's data preparation workflow:

1. Mount Google Drive and create the project folders.
2. Authenticate to Kaggle inside the Colab runtime when downloading is required.
3. Download and extract `yelp_academic_dataset_review.json`.
4. Inspect the source records.
5. Scan the reviews and select a balanced sample using deterministic hash-based sampling.
6. Map star ratings to sentiment labels.
7. Create stratified train/validation/test splits.
8. Save the split files and metadata to Google Drive.
9. Reload the saved files and validate columns, labels, and split non-overlap.

### Label mapping

| Yelp stars | Sentiment | Label |
|---|---|---:|
| 1–2 | Negative | 0 |
| 3 | Neutral | 1 |
| 4–5 | Positive | 2 |

### Split sizes

| Split | Total | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
| Train | 120,000 | 40,000 | 40,000 | 40,000 |
| Validation | 15,000 | 5,000 | 5,000 | 5,000 |
| Test | 15,000 | 5,000 | 5,000 | 5,000 |

The saved split files are expected under:

`MyDrive/NLP_Task3/dataset/processed/`

- `yelp_train_150k_seed42.csv.gz`
- `yelp_validation_150k_seed42.csv.gz`
- `yelp_test_150k_seed42.csv.gz`
- `dataset_manifest.json`
- distribution CSV files

## What each teammate should do

- If the processed split files already exist, use them directly; do not rerun the full preparation notebook.
- Both scratch and pretrained models must use these exact split files and the same label mapping.
- Do not train, tune, or select checkpoints using the test split.

## Kaggle credentials

`kaggle.json` is a private API credential. Never upload it to GitHub or share it with teammates. If the dataset must be downloaded again, each teammate should generate their own Kaggle API token and upload it directly to their own Colab runtime using the authentication cell.

The raw Yelp JSON is several GB. Regenerating the dataset requires downloading and scanning the full source file; loading the prepared split files is much faster.

## Google Drive access

The teammate needs permission to the shared `NLP_Task3` Drive folder to read the prepared dataset and saved checkpoint. If Drive shows "Request access", send the access request to the folder owner and wait for approval. Do not make the folder public just to avoid the request.
