# Final Report — Yelp Sentiment Transformer Comparison

## Task and dataset
Three-class sentiment analysis using Yelp Open Dataset. Include the source URL, dataset agreement/license note, exact preprocessing, duplicate policy, label mapping, split counts, seed, and test split SHA-256.

## Model A — Scratch Transformer
Describe the manual Q/K/V attention, multi-head attention, sinusoidal positional encoding, FFN, residual connections, LayerNorm, mask, architecture dimensions, optimizer, training epochs, hardware, measured training time, and parameter count.

## Model B — Pretrained DistilBERT
Describe checkpoint `distilbert-base-uncased`, classification head, tokenizer compatibility verification, same 192+64 head-tail token policy, optimizer, learning rate, batch/gradient accumulation, epochs, hardware, measured fine-tuning time, and parameter count.

## Evaluation protocol
State that both prediction files were paired by identical test record indices and true labels, and report the shared test SHA-256. Test data was not used for checkpoint selection or tuning. Model A raw argmax is the primary comparison; validation-selected bias results are secondary.

## Measured results
Paste the generated `reports/final_comparison_report.md` table. Include accuracy, macro precision/recall/F1, weighted-F1, per-class metrics, confusion matrices, inference latency, parameters, and training time.

## Discussion and limitations
Explain the observed gap using measured results only. Disclose the different training budgets (Model A: 3 epochs; Model B: 1 epoch unless deliberately changed), class imbalance in the natural Yelp test distribution, hardware differences in timing, and any inference truncation/latency limitations.

## Interactive demo
Describe the Streamlit app, include screenshots from the final run, and document the local run command.
