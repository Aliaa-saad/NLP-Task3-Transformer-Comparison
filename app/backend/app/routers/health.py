from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.predictors import ModelRegistry
from app.schemas import HealthResponse, ModelInfo, ModelStatus

router = APIRouter(prefix="/api", tags=["health"])

DESCRIPTIONS = {
    "model_a": (
        "Pre-LN encoder-only Transformer (8 layers, d_model 512, 8 heads) trained from scratch "
        "on deduplicated Yelp reviews, with masked mean pooling."
    ),
    "model_b": "distilbert-base-uncased fine-tuned for 3-class sentiment on a balanced 120k Yelp sample.",
}
SHORT = {"model_a": "Model A", "model_b": "Model B"}
TOKENIZER = {
    "model_a": "bert-base-uncased (WordPiece, 30,522)",
    "model_b": "distilbert-base-uncased (WordPiece, 30,522)",
}
TRUNCATION = {
    "model_a": "head+tail: first 192 + last 64 tokens (max 256)",
    "model_b": "right truncation (max 256)",
}


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    reg = _registry(request)
    statuses = []
    for key, name in reg.names.items():
        p = reg.predictors.get(key)
        if p is not None:
            statuses.append(
                ModelStatus(model_key=key, name=name, status="loaded", demo_mode=p.demo_mode,
                            message="Random weights (DEMO_RANDOM_WEIGHTS=1)" if p.demo_mode else None)
            )
        else:
            statuses.append(
                ModelStatus(model_key=key, name=name, status="error",
                            message=reg.errors.get(key, "Model not loaded."))
            )
    ok = all(s.status == "loaded" for s in statuses)
    return HealthResponse(
        status="ok" if ok else "degraded",
        device=str(reg.device),
        models=statuses,
        demo_random_weights=reg.settings.demo_random_weights,
    )


@router.get("/models", response_model=list[ModelInfo])
def models(request: Request) -> list[ModelInfo]:
    reg = _registry(request)
    out = []
    for key, name in reg.names.items():
        p = reg.predictors.get(key)
        out.append(
            ModelInfo(
                model_key=key,
                name=name,
                short_name=SHORT[key],
                description=DESCRIPTIONS[key],
                parameters=getattr(p, "parameters", None),
                biases_available=bool(getattr(p, "biases_available", False)),
                biases=getattr(p, "biases", None) if key == "model_a" else None,
                tokenizer=TOKENIZER[key],
                truncation=TRUNCATION[key],
                loaded=p is not None,
                demo_mode=bool(getattr(p, "demo_mode", False)),
                error=None if p is not None else reg.errors.get(key),
            )
        )
    return out
