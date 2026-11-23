"""Safely install the Colab-exported artifact bundle into ./artifacts."""
from __future__ import annotations
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    "artifacts/model_a/best_model.pt",
    "artifacts/model_a/metrics.json",
    "artifacts/model_a/test_predictions.csv",
    "artifacts/model_a/tokenizer/vocab.txt",
    "artifacts/model_b/model/config.json",
    "artifacts/model_b/model/model.safetensors",
    "artifacts/model_b/metrics.json",
    "artifacts/model_b/test_predictions.csv",
]

def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    for info in zf.infolist():
        name = PurePosixPath(info.filename)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe archive path: {info.filename}")
        target = (dest / Path(*name.parts)).resolve()
        if dest.resolve() not in target.parents and target != dest.resolve():
            raise ValueError(f"Archive path escapes destination: {info.filename}")
    zf.extractall(dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", nargs="?", default="NLP_Task3_Final_Artifacts.zip")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    if not bundle.is_file():
        raise FileNotFoundError(f"Artifact bundle not found: {bundle}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(bundle) as zf:
            safe_extract(zf, tmp_path)
        extracted = tmp_path / "artifacts"
        missing = [name for name in EXPECTED if not (tmp_path / name).is_file()]
        if missing:
            raise FileNotFoundError("Artifact bundle is incomplete:\n" + "\n".join(missing))
        destination = ROOT / "artifacts"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(extracted, destination)
    print("Artifacts installed successfully into:", ROOT / "artifacts")
    print("Next: python scripts/build_comparison.py")
