import type {
  ErrorRow,
  HealthResponse,
  HistoryRow,
  MetricsResponse,
  ModelInfo,
  PredictRequest,
  PredictResponse,
} from "../types/api";

const BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number | null,
    public readonly kind: "network" | "http",
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export const isNetworkError = (e: unknown): boolean => e instanceof ApiError && e.kind === "network";

function detailMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : String(d)))
        .join("; ");
    }
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError("Cannot reach the backend API.", null, "network");
  }
  // The Vite dev proxy answers 502/504 when FastAPI is down; treat that as unreachable too.
  if (res.status === 502 || res.status === 504) {
    throw new ApiError("Cannot reach the backend API.", res.status, "network");
  }
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(detailMessage(body, `Request failed (${res.status})`), res.status, "http");
  return body as T;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  models: () => request<ModelInfo[]>("/api/models"),
  metrics: () => request<MetricsResponse>("/api/metrics"),
  history: () => request<HistoryRow[]>("/api/history/model_a"),
  errors: (limit = 20) => request<ErrorRow[]>(`/api/errors/model_a?limit=${limit}`),
  predict: (body: PredictRequest) =>
    request<PredictResponse>("/api/predict", { method: "POST", body: JSON.stringify(body) }),
};
