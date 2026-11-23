# Two Transformers. One review.

A locally runnable web app that compares the two Yelp sentiment classifiers from this project side by side:

| | Model A | Model B |
|---|---|---|
| What | Pre-LN encoder-only Transformer trained **from scratch** | `distilbert-base-uncased` **fine-tuned** |
| Params | 40.8M | 67.0M |
| Tokenizer | bert-base-uncased | distilbert-base-uncased |
| Truncation | head+tail (first 192 + last 64 tokens) | right truncation (256) |
| Post-processing | tuned additive logit biases (toggle) | none |

Labels: `0 negative` (1–2★), `1 neutral` (3★), `2 positive` (4–5★).

**Stack:** FastAPI + PyTorch (backend), React 18 + Vite + TypeScript + Tailwind + Recharts + framer-motion + TanStack Query (frontend).

---

## Quick start (Windows PowerShell)

Run all commands from the **repository root**.

```powershell
# 1. one-time setup: venv + CPU PyTorch + backend deps + npm packages
.\app\setup.ps1            # add -Cuda for a CUDA 12.1 PyTorch build

# 2. put the artifacts in app\backend\artifacts\  (see "Artifacts" below)

# 3a. single-process production mode: builds the UI, serves everything on http://localhost:8000
.\app\run_prod.ps1

# 3b. or dev mode: API with --reload (new window) + Vite on http://localhost:5173
.\app\run_dev.ps1          # add -Demo to use random weights when checkpoints are missing
```

If script execution is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Manual commands

```powershell
# backend
cd app\backend
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --reload       # http://127.0.0.1:8000/docs
.\.venv\Scripts\python -m pytest                              # tests

# frontend (second terminal)
cd app\frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api to :8000)
npm run build      # production build -> dist/ (FastAPI serves it at / when present)
npm run lint
```

The first start downloads the `bert-base-uncased` tokenizer (~1 MB) from Hugging Face.
The server **always starts**, even with missing artifacts: the affected model card shows the load error, and the dashboard shows what is missing.

---

## Artifacts

Weights are not in git. Copy them into `app/backend/artifacts/`. The full layout is in
[`backend/artifacts/README.md`](backend/artifacts/README.md):

```
artifacts/model_a/     best_model.pt, metrics.json, training_history.csv, error_analysis_top50.csv
artifacts/model_b/     config.json, model.safetensors, tokenizer.json, tokenizer_config.json,
                       special_tokens_map.json, vocab.txt  (+ optional pretrained_results.json)
artifacts/comparison/  comparison_results.json
```

---

## Features

- **Analyze:** a textarea (validated: not empty, ≤ 5,000 chars) and one **Run both models** button. Each model gets a card with the
  predicted class pill, confidence, animated probability bars, and **server-measured forward-pass latency**
  (`perf_counter`, CUDA-synchronised, one warm-up pass at startup). Tokenization time, token count and truncation are shown
  separately. The card also shows browser round-trip time, labelled separately. An agreement banner, 5 example chips, and the
  Model A logit-bias toggle complete the page.
- **Compare:** Balanced / Natural view toggle; KPI tiles with Δ Macro-F1 (B − A) and the bootstrap 95% CI; grouped
  Accuracy/Macro-F1/Weighted-F1 chart; per-class F1 chart; a metrics table with the best value per row highlighted;
  annotated confusion-matrix heatmaps (counts + row %); a latency card; the eval-slice caveat; a raw/logit-adjusted toggle for
  Model A; Model A training curves; and its highest-confidence errors. Head-to-head numbers come **only** from
  `comparison_results.json` (the same held-out reviews for both models).
- **About:** task description, label mapping, and spec cards for both models.
- Dark/light mode, responsive down to 360 px, keyboard accessible, `prefers-reduced-motion` respected.
- Degrades gracefully: if the backend is offline, a banner shows the start command. If one model fails, the other still works.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | device + per-model status (`loaded` / `error` + message) |
| GET | `/api/models` | name, description, parameter count, biases availability |
| POST | `/api/predict` | `{ "text": str, "apply_biases": true }` → `{ results: [Prediction, Prediction], agree }` |
| GET | `/api/metrics` | validated `comparison_results.json` + training summaries |
| GET | `/api/history/model_a` | training history rows |
| GET | `/api/errors/model_a?limit=20` | high-confidence errors |

Interactive docs: http://localhost:8000/docs

## Configuration

Environment variables (or `app/backend/.env`, see `.env.example`):
`DEVICE` (`auto`|`cpu`|`cuda`), `MAX_CHARS` (5000), `CORS_ORIGINS`, `ARTIFACTS_DIR`,
`DEMO_RANDOM_WEIGHTS=1` (**dev only**: random weights when checkpoints are missing; flagged in API + UI).

## Project layout

```
app/
  backend/   FastAPI app (app/), tests/, artifacts/ (git-ignored), requirements.txt
  frontend/  React app (src/api, components, pages, hooks, types, lib)
  tools/     e2e_screenshots.py (Playwright check), onnx_to_checkpoint.py
  screenshots/
  setup.ps1  run_dev.ps1  run_prod.ps1
```

`backend/app/models/model_a_arch.py` is copied verbatim from `notebooks/train_model_a_colab_(updatedd).ipynb`
(only the `try: from src.config import ...` block is removed), and the checkpoint is loaded with `strict=True`.

## Screenshots

| | |
|---|---|
| Analyze (desktop) | `screenshots/real_desktop_analyze_result.png` |
| Long review, truncated | `screenshots/real_desktop_analyze_long_truncated.png` |
| Compare (desktop) | `screenshots/real_desktop_compare_balanced.png` |
| About | `screenshots/real_desktop_about.png` |
| Mobile | `screenshots/real_mobile_analyze_result.png` |
| Light mode | `screenshots/real_desktop_analyze_light.png` |

Re-take all screenshots with `app\backend\.venv\Scripts\python app\tools\e2e_screenshots.py --url http://localhost:8000 --with-comparison`.
