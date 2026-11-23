# Artifacts (not in git)

Model weights and evaluation outputs live here. Everything in this folder except this README is git-ignored.
The server starts even if files are missing: the affected model or dashboard section shows an error instead.

```
artifacts/
├── model_a/                       ← Drive: <project folder>/yelp_model_a/
│   ├── best_model.pt              ← checkpoints_v2/best_model.pt   (~160 MB; torch.save dict)
│   ├── metrics.json               ← artifacts_v2/metrics.json      (logit biases, own test metrics)
│   ├── training_history.csv       ← artifacts_v2/training_history.csv
│   ├── error_analysis_top50.csv   ← artifacts_v2/error_analysis_top50.csv
│   └── config.json                ← artifacts_v2/config.json       (optional)
├── model_b/                       ← Drive: <project folder>/pretrained_model/distilbert_yelp_sentiment/  (NOT *_clean)
│   ├── config.json
│   ├── model.safetensors          (~268 MB)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.txt
│   └── pretrained_results.json    ← <project folder>/results/pretrained_results.json (optional, About/table)
└── comparison/
    └── comparison_results.json    ← <project folder>/results/comparison/comparison_results.json
```

`best_model.pt` must be a dict with `model_state_dict`, `model_cfg` (`dataclasses.asdict(ModelAConfig)`),
`global_step` and `best_val_macro_f1`. It is loaded with `strict=True`.

If only the ONNX export is available (`artifacts_v2/onnx/model_a.onnx`), rebuild an equivalent checkpoint with:

```powershell
cd app\backend
.\.venv\Scripts\python -m pip install onnx
.\.venv\Scripts\python ..\tools\onnx_to_checkpoint.py path\to\model_a.onnx artifacts\model_a\best_model.pt
```
