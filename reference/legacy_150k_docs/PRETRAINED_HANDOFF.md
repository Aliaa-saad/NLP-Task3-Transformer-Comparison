# Pretrained Model Handoff — Yelp Sentiment Analysis

## Completed implementation

The pretrained branch is in `notebooks/01_pretrained_transformer.ipynb`.

Completed work:
1. Loaded the shared train, validation, and test splits.
2. Validated required columns and the three-class label mapping.
3. Loaded Hugging Face checkpoint `distilbert-base-uncased`.
4. Added a three-class sequence-classification head.
5. Fine-tuned for 2 epochs on a Tesla T4 GPU.
6. Selected the best checkpoint using validation Macro-F1.
7. Evaluated the selected checkpoint on the held-out test set.
8. Generated classification report and confusion matrix.
9. Measured inference latency and parameter count.
10. Saved the model, tokenizer, metrics, history, predictions, and confusion matrix to Google Drive.
11. Implemented `predict_sentiment(text)` for later app integration.

## Recorded results

| Metric | Result |
|---|---:|
| Test accuracy | 0.8365 |
| Macro precision | 0.8375 |
| Macro recall | 0.8365 |
| Macro-F1 | 0.8369 |
| Average inference latency | 6.34 ms |
| Median inference latency | 6.06 ms |
| Total parameters | 66,955,779 |
| Training time | 32.56 minutes |
| GPU | Tesla T4 |

These are the recorded measurements from the completed run, not guaranteed results for a future rerun.

## Google Drive artifacts

Project folder: `MyDrive/NLP_Task3/`

- Dataset splits: `dataset/processed/`
- Checkpoint and tokenizer: `pretrained_model/distilbert_yelp_sentiment_clean/`
- Metrics: `results/pretrained_results.json`
- Training history: `results/pretrained_training_history.csv`
- Classification report: `results/pretrained_classification_report.csv`
- Confusion matrix: `results/pretrained_confusion_matrix.png`
- Test predictions: `results/pretrained_test_predictions.csv.gz`

The test-predictions export contains original review text. Keep it in shared Drive, not in the public repository.

## What the teammate should continue

1. Analyze classification errors, especially Neutral reviews confused with Negative or Positive.
2. Create a reusable inference module such as `src/pretrained_inference.py`.
3. Load the saved checkpoint and tokenizer from shared Drive; do not retrain just to use the model.
4. Return sentiment, confidence, per-class probabilities, and measured latency.
5. Test empty, short, long, and normal review inputs.
6. Integrate the pretrained model into Streamlit.
7. Coordinate a common prediction interface with the scratch-model teammates.
8. Compare both models on the exact same held-out test examples.

Do not tune or select the model using the test set. Do not run the clean initialization and training cells unless a new training run is explicitly intended.
