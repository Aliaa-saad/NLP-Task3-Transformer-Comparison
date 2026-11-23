# Experiment protocol and interpretation rules

- Dataset records are deduplicated by lowercasing and collapsing whitespace before hashing; the shared tokenized cache and split manifest are produced by Model A's builder.
- Label IDs: negative=0 (1–2 stars), neutral=1 (3 stars), positive=2 (4–5 stars).
- The train/validation/test partitions are fixed with seed 42 and SHA-256 fingerprints embedded in the notebooks.
- Model B verifies exact split fingerprints and class counts before it starts training. It also checks that DistilBERT and BERT uncased tokenizers yield identical vocabulary and representative token IDs.
- Both models use 256-token inputs with the same head+tail policy (first 192 + last 64 tokens).
- Model A uses fixed-size training batches with `drop_last=True` (37 train examples are omitted from each epoch schedule); Model B retains its final partial batch. Both use the same split membership, and the held-out test examples are identical.
- Model A checkpoint selection uses validation Macro-F1; logit-bias tuning uses validation logits. Model B checkpoint selection uses validation Macro-F1. Neither model may tune on the test set.
- Primary comparison: Model A raw argmax vs Model B over the full 40,000-review held-out test set. Model A's validation-adjusted result is secondary and explicitly labeled.
- Balanced diagnostic view is sampled from the held-out test set only; it does not replace the primary natural-distribution test results.
- Latency is directly comparable only when the recorded GPU and benchmark protocol match. Training duration must be reported with each model's hardware and epoch budget.
