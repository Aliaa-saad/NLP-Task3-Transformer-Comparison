# GitHub upload checklist

Upload these files/folders to the root of the existing repository:

## Root
- `README.md`
- `.gitignore`
- `requirements-pretrained.txt`
- `GITHUB_UPLOAD_CHECKLIST.md`

## notebooks/
- `00_dataset_preparation.ipynb`
- `01_pretrained_transformer.ipynb`

## docs/
- `DATASET_AND_RUN_GUIDE.md`
- `PRETRAINED_HANDOFF.md`

## Folder README files
- `dataset/README.md`
- `pretrained_model/README.md`
- `scratch_model/README.md`
- `results/README.md`

Do NOT upload:
- `kaggle.json` or any credentials
- Raw Yelp JSON or ZIP files
- Processed CSV/CSV.GZ split files
- DistilBERT weights (`model.safetensors`, `pytorch_model.bin`)
- `pretrained_test_predictions.csv.gz`
- `.env` or Streamlit secrets

Share `MyDrive/NLP_Task3/` with teammates separately. If Drive displays "Request access", request access from the folder owner and wait for approval.

Important:
- `00_dataset_preparation.ipynb` is shared. Do not rerun it if the prepared splits already exist.
- `01_pretrained_transformer.ipynb` has training cells. Do not run the clean initialization and training cells just to use the saved checkpoint.
