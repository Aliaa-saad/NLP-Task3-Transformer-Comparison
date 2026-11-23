export const pct = (v: number | null | undefined, digits = 1): string =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const dec = (v: number | null | undefined, digits = 3): string =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const ms = (v: number | null | undefined): string =>
  v == null ? "—" : v >= 100 ? `${v.toFixed(0)} ms` : `${v.toFixed(1)} ms`;

export const millions = (v: number | null | undefined): string =>
  v == null ? "—" : `${(v / 1e6).toFixed(1)}M`;

export const int = (v: number | null | undefined): string => (v == null ? "—" : Math.round(v).toLocaleString("en-US"));

export function duration(sec: number | null | undefined): string {
  if (sec == null) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.round((sec % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m} min`;
}

export const signed = (v: number, digits = 3): string => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;

export const cap = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);

/** "NVIDIA A100-SXM4-80GB" -> "A100", "Tesla T4" -> "T4". */
export function shortGpu(name: string | null | undefined): string | null {
  if (!name) return null;
  const m = name.match(/\b(H100|A100|A10G?|L4|T4|V100|P100|K80|RTX\s?\w+)\b/i);
  return m?.[1] ?? name;
}
