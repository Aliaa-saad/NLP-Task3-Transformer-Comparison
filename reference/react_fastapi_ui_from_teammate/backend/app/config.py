"""Application settings (pydantic-settings). Override via environment variables or `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BACKEND_DIR.parent
REPO_DIR = APP_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    artifacts_dir: Path = Field(default=BACKEND_DIR / "artifacts")
    frontend_dist: Path = Field(default=APP_DIR / "frontend" / "dist")

    # "auto" picks CUDA when available, otherwise CPU. Use "cpu" or "cuda" / "cuda:0" to force.
    device: str = "auto"
    max_chars: int = 5000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_a_tokenizer: str = "bert-base-uncased"
    model_b_base: str = "distilbert-base-uncased"  # only used by DEMO_RANDOM_WEIGHTS

    # Development only: run with randomly initialised weights when checkpoints are missing.
    demo_random_weights: bool = False
    # Skip the startup warm-up forward pass (useful for fast tests).
    skip_warmup: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip().startswith("["):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def model_a_dir(self) -> Path:
        return self.artifacts_dir / "model_a"

    @property
    def model_b_dir(self) -> Path:
        return self.artifacts_dir / "model_b"

    @property
    def comparison_path(self) -> Path:
        return self.artifacts_dir / "comparison" / "comparison_results.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
