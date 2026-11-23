from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import ValidationError

from app.schemas import ErrorRow, HistoryRow, MetricsResponse, SetupResponse
from app.services import metrics as svc

router = APIRouter(prefix="/api", tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: Request) -> MetricsResponse:
    settings = request.app.state.settings
    try:
        comparison = svc.load_comparison(settings)
    except svc.ArtifactMissing as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=500, detail=f"comparison_results.json has an unexpected schema: {exc}"
        ) from exc
    return MetricsResponse(comparison=comparison, training=svc.load_training_summaries(settings))


@router.get("/history/model_a", response_model=list[HistoryRow])
def history_model_a(request: Request) -> list[HistoryRow]:
    try:
        return svc.load_history(request.app.state.settings)
    except svc.ArtifactMissing as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/errors/model_a", response_model=list[ErrorRow])
def errors_model_a(request: Request, limit: int = Query(20, ge=1, le=50)) -> list[ErrorRow]:
    try:
        return svc.load_errors(request.app.state.settings, limit)
    except svc.ArtifactMissing as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/setup", response_model=SetupResponse)
def setup(request: Request) -> SetupResponse:
    """Dataset, splits, seed, tokenizers and hardware, read from the training artifacts."""
    return svc.load_setup(request.app.state.settings)
