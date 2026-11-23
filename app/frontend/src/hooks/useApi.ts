import { useMutation, useQuery } from "@tanstack/react-query";
import { api, isNetworkError } from "../api/client";
import type { PredictRequest, PredictResponse } from "../types/api";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    retry: false,
    refetchInterval: (q) => (q.state.status === "error" ? 4000 : 15000),
    refetchOnWindowFocus: true,
  });
}

export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: api.models, retry: 1, staleTime: 60_000 });
}

const noRetryOn404 = (count: number, err: Error) =>
  !(err instanceof Error && "status" in err && (err as { status: number | null }).status === 404) &&
  !isNetworkError(err) &&
  count < 1;

export function useMetrics() {
  return useQuery({ queryKey: ["metrics"], queryFn: api.metrics, retry: noRetryOn404, staleTime: Infinity });
}

export function useHistory() {
  return useQuery({ queryKey: ["history"], queryFn: api.history, retry: noRetryOn404, staleTime: Infinity });
}

export function useErrors(limit = 20) {
  return useQuery({
    queryKey: ["errors", limit],
    queryFn: () => api.errors(limit),
    retry: noRetryOn404,
    staleTime: Infinity,
  });
}

export interface TimedPrediction extends PredictResponse {
  roundTripMs: number;
}

export function usePredict() {
  return useMutation({
    mutationFn: async (body: PredictRequest): Promise<TimedPrediction> => {
      const t0 = performance.now();
      const res = await api.predict(body);
      return { ...res, roundTripMs: performance.now() - t0 };
    },
  });
}
