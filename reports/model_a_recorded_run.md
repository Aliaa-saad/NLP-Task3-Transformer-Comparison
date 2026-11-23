# Model A — recorded run from the supplied notebook

These values are copied from the saved outputs of `01_model_a_scratch_full_yelp.ipynb`. They describe Model A only and are not a Model A-vs-Model B comparison.

- Raw Yelp reviews: 6,990,280
- Valid unique reviews after deduplication: 6,973,573
- Train / validation / test: 6,913,573 / 20,000 / 40,000
- Parameters: 40,848,899
- Training epochs: 3
- Active training time: 81.85 minutes
- Recorded GPU: NVIDIA A100-SXM4-80GB
- Test accuracy (raw argmax): 0.8959
- Test Macro-F1 (raw argmax): 0.7714
- Test accuracy (validation-selected logit adjustment): 0.8870
- Test Macro-F1 (validation-selected logit adjustment): 0.7902
- GPU bs=1 mean model-forward latency: 8.285 ms per sample
- GPU bs=32 mean model-forward latency: 0.335 ms per sample

The primary final comparison uses raw argmax for both models. The validation-adjusted Model A result is shown separately. Model B full-data results are not yet available in the uploaded notebook, so no final comparative conclusion is made here.
