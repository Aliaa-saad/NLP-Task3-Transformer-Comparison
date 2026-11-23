"""Pydantic v2 request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LabelName = Literal["negative", "neutral", "positive"]
LABELS: tuple[LabelName, ...] = ("negative", "neutral", "positive")


# ---------------------------------------------------------------- inference
class PredictRequest(BaseModel):
    text: str = Field(..., description="Review text (1-5000 characters).")
    apply_biases: bool = Field(True, description="Apply Model A's tuned logit biases.")


class Probs(BaseModel):
    negative: float
    neutral: float
    positive: float


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_key: str
    model_name: str
    label: LabelName | None = None
    label_id: int | None = None
    probs: Probs | None = None
    latency_ms: float | None = None
    tokenize_ms: float | None = None
    num_tokens: int | None = None
    original_tokens: int | None = None
    truncated: bool | None = None
    biases_applied: bool | None = None
    demo_mode: bool = False
    warning: str | None = None
    error: str | None = None


class PredictResponse(BaseModel):
    results: list[Prediction]
    agree: bool | None
    device: str


# ---------------------------------------------------------------- health / models
class ModelStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_key: str
    name: str
    status: Literal["loaded", "error"]
    message: str | None = None
    demo_mode: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    device: str
    models: list[ModelStatus]
    demo_random_weights: bool


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_key: str
    name: str
    short_name: str
    description: str
    parameters: int | None
    biases_available: bool
    biases: list[float] | None = None
    tokenizer: str
    truncation: str
    loaded: bool
    demo_mode: bool = False
    error: str | None = None


# ---------------------------------------------------------------- comparison metrics
class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class ClassMetrics(_Loose):
    precision: float
    recall: float
    f1: float
    support: int | float | None = None


class ConfusionMatrixData(_Loose):
    labels: list[str]
    matrix: list[list[int | float]]


class EvalMetrics(_Loose):
    accuracy: float
    precision_macro: float | None = None
    recall_macro: float | None = None
    f1_macro: float
    f1_weighted: float | None = None
    per_class: dict[str, ClassMetrics]
    confusion_matrix: ConfusionMatrixData


class Bootstrap(_Loose):
    macro_f1_diff_a_minus_b: float
    ci95_low: float
    ci95_high: float
    p_a_better: float | None = None


class ComparisonView(_Loose):
    size: int | None = None
    model_a_raw: EvalMetrics
    model_a_logit_adjusted: EvalMetrics
    model_b: EvalMetrics
    bootstrap_a_adjusted_vs_b: Bootstrap | None = None
    agreement_rate_a_adjusted_vs_b: float | None = None


class LatencyStats(_Loose):
    mean_ms: float | None = None
    median_ms: float | None = None
    p95_ms: float | None = None
    n: int | None = None


class LatencySameMachine(_Loose):
    device: str | None = None
    gpu_name: str | None = None
    protocol: str | None = None
    model_a: LatencyStats | None = None
    model_b: LatencyStats | None = None


class ComparisonResults(_Loose):
    description: str | None = None
    caveat: str | None = None
    eval_slice: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    views: dict[str, ComparisonView]
    latency_same_machine: LatencySameMachine | None = None


class TrainingSummary(BaseModel):
    """Extra per-model facts for the dashboard table (from each model's own artifacts)."""

    parameters: int | None = None
    training_time_sec: float | None = None
    epochs: int | None = None
    best_val_macro_f1: float | None = None
    train_size: int | None = None
    regime: str | None = None  # "scratch training" | "fine-tuning"
    gpu_name: str | None = None
    gpu_source: str | None = None  # which artifact the GPU name was read from


class MetricsResponse(BaseModel):
    comparison: ComparisonResults
    training: dict[str, TrainingSummary]


class SetupResponse(_Loose):
    """Dataset + experimental setup, read from the training artifacts."""

    dataset: dict[str, Any]
    label_mapping: list[dict[str, Any]]
    model_a: dict[str, Any]
    model_b: dict[str, Any]
    comparison: dict[str, Any]


class HistoryRow(_Loose):
    step: float | None = None
    epoch: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    val_macro_f1: float | None = None
    val_accuracy: float | None = None
    lr: float | None = None


class ErrorRow(_Loose):
    rank: int
    text: str
    true_label: str
    pred_label: str
    confidence: float
    prob_negative: float | None = None
    prob_neutral: float | None = None
    prob_positive: float | None = None
