// Types mirroring app/backend/app/schemas.py

export const LABELS = ["negative", "neutral", "positive"] as const;
export type Label = (typeof LABELS)[number];
export type ModelKey = "model_a" | "model_b";

export type Probs = Record<Label, number>;

export interface Prediction {
  model_key: ModelKey;
  model_name: string;
  label: Label | null;
  label_id: number | null;
  probs: Probs | null;
  latency_ms: number | null;
  tokenize_ms: number | null;
  num_tokens: number | null;
  original_tokens: number | null;
  truncated: boolean | null;
  biases_applied: boolean | null;
  demo_mode: boolean;
  warning: string | null;
  error: string | null;
}

export interface PredictRequest {
  text: string;
  apply_biases: boolean;
}

export interface PredictResponse {
  results: Prediction[];
  agree: boolean | null;
  device: string;
}

export interface ModelStatus {
  model_key: ModelKey;
  name: string;
  status: "loaded" | "error";
  message: string | null;
  demo_mode: boolean;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  device: string;
  models: ModelStatus[];
  demo_random_weights: boolean;
}

export interface ModelInfo {
  model_key: ModelKey;
  name: string;
  short_name: string;
  description: string;
  parameters: number | null;
  biases_available: boolean;
  biases: number[] | null;
  tokenizer: string;
  truncation: string;
  loaded: boolean;
  demo_mode: boolean;
  error: string | null;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support?: number | null;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
}

export interface EvalMetrics {
  accuracy: number;
  precision_macro?: number | null;
  recall_macro?: number | null;
  f1_macro: number;
  f1_weighted?: number | null;
  per_class: Record<string, ClassMetrics>;
  confusion_matrix: ConfusionMatrixData;
}

export interface Bootstrap {
  macro_f1_diff_a_minus_b: number;
  ci95_low: number;
  ci95_high: number;
  p_a_better?: number | null;
}

export interface ComparisonView {
  size?: number | null;
  model_a_raw: EvalMetrics;
  model_a_logit_adjusted: EvalMetrics;
  model_b: EvalMetrics;
  bootstrap_a_adjusted_vs_b?: Bootstrap | null;
  agreement_rate_a_adjusted_vs_b?: number | null;
}

export interface LatencyStats {
  mean_ms?: number | null;
  median_ms?: number | null;
  p95_ms?: number | null;
  n?: number | null;
}

export interface ComparisonResults {
  description?: string | null;
  caveat?: string | null;
  eval_slice: Record<string, unknown>;
  models: Record<string, Record<string, unknown>>;
  views: Record<string, ComparisonView>;
  latency_same_machine?: {
    device?: string | null;
    gpu_name?: string | null;
    protocol?: string | null;
    model_a?: LatencyStats | null;
    model_b?: LatencyStats | null;
  } | null;
}

export interface TrainingSummary {
  parameters: number | null;
  training_time_sec: number | null;
  epochs: number | null;
  best_val_macro_f1: number | null;
  train_size: number | null;
  regime: string | null;
  gpu_name: string | null;
  gpu_source: string | null;
}

export interface SetupModel {
  splits: Record<string, number | null>;
  split_strategy: string;
  seed: number | null;
  tokenizer: string | null;
  pretrained_weights: boolean;
  max_length: number | null;
  head_tokens?: number | null;
  tail_tokens?: number | null;
  gpu: string | null;
  gpu_source?: string | null;
}

export interface SetupResponse {
  dataset: {
    name: string;
    slug: string | null;
    file: string | null;
    url: string | null;
    raw_reviews: number | null;
    duplicates_removed: number | null;
    deduped_reviews: number | null;
  };
  label_mapping: { id: number; label: Label; stars: string }[];
  model_a: SetupModel;
  model_b: SetupModel;
  comparison: { device: string | null; gpu: string | null; protocol: string | null };
}

export interface MetricsResponse {
  comparison: ComparisonResults;
  training: Record<string, TrainingSummary>;
}

export interface HistoryRow {
  step: number | null;
  epoch: number | null;
  train_loss: number | null;
  val_loss: number | null;
  val_macro_f1: number | null;
  val_accuracy: number | null;
  lr: number | null;
}

export interface ErrorRow {
  rank: number;
  text: string;
  true_label: string;
  pred_label: string;
  confidence: number;
  prob_negative: number | null;
  prob_neutral: number | null;
  prob_positive: number | null;
}
