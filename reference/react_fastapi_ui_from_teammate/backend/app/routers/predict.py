from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models.predictors import ModelRegistry
from app.schemas import PredictRequest, PredictResponse

router = APIRouter(prefix="/api", tags=["predict"])


# A plain `def` endpoint runs in FastAPI's threadpool, so inference never blocks the event loop.
@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    reg: ModelRegistry = request.app.state.registry
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Please enter a review: the text is empty.")
    max_chars = reg.settings.max_chars
    if len(text) > max_chars:
        raise HTTPException(
            status_code=422,
            detail=f"Review is too long: {len(text):,} characters (max {max_chars:,}).",
        )

    results = reg.predict_all(text, apply_biases=body.apply_biases)
    labels = [r.label for r in results if r.error is None and r.label is not None]
    agree = (len(set(labels)) == 1) if len(labels) == len(results) else None
    return PredictResponse(results=results, agree=agree, device=str(reg.device))
