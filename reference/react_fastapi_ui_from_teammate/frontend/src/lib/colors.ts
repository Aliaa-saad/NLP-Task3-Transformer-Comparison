import type { Label } from "../types/api";

const CLASS_VAR: Record<Label, string> = { negative: "var(--neg)", neutral: "var(--neu)", positive: "var(--pos)" };

export function classColorVar(label: string): string {
  return CLASS_VAR[label as Label] ?? "var(--muted)";
}
