# GitHub Upload Checklist

Commit source code, notebooks, docs, tests, and safe summary templates. Do **not** commit raw Yelp JSON, tokenized caches, Kaggle credentials, model weights, private prediction text, or the local artifact bundle.

- [ ] `notebooks/01_model_a_scratch_full_yelp.ipynb` included.
- [ ] `notebooks/02_model_b_distilbert_full_yelp.ipynb` included.
- [ ] `notebooks/03_export_final_artifacts.ipynb` included.
- [ ] `app.py`, `src/`, `scripts/`, and `tests/` included.
- [ ] `requirements.txt` and README setup commands included.
- [ ] Final measured report generated only after the full-data Model B run.
- [ ] No `kaggle.json`, `.env`, raw dataset, Drive tokenized cache, model weights, or test review text committed.
- [ ] `reference/legacy_150k_results/` clearly remains reference-only; do not present it as final results.
