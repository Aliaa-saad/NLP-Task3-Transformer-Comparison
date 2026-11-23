from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
checks = {
    "Model A checkpoint": ROOT / "artifacts/model_a/best_model.pt",
    "Model A metrics": ROOT / "artifacts/model_a/metrics.json",
    "Model A test predictions": ROOT / "artifacts/model_a/test_predictions.csv",
    "Model A tokenizer": ROOT / "artifacts/model_a/tokenizer/vocab.txt",
    "Model B config": ROOT / "artifacts/model_b/model/config.json",
    "Model B weights": ROOT / "artifacts/model_b/model/model.safetensors",
    "Model B metrics": ROOT / "artifacts/model_b/metrics.json",
    "Model B test predictions": ROOT / "artifacts/model_b/test_predictions.csv",
}
for label, path in checks.items():
    status = "FOUND" if path.is_file() else "MISSING"
    size = f" ({path.stat().st_size/1024/1024:.1f} MB)" if path.is_file() else ""
    print(f"{status:7} {label}: {path.relative_to(ROOT)}{size}")
if all(p.is_file() for p in checks.values()):
    print("\nCore artifact files are present. You can build the paired comparison.")
else:
    print("\nArtifacts are incomplete. Finish Model B and export from the shared Drive first.")
