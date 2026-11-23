import type { ReactNode } from "react";
import { ModelDot } from "./ui";

export interface MetricRow {
  label: string;
  group?: string;
  a: number | null | undefined;
  b: number | null | undefined;
  format: (v: number | null | undefined) => string;
  /** Which direction is better; "none" disables highlighting. */
  better?: "higher" | "lower" | "none";
  note?: ReactNode;
}

interface Props {
  rows: MetricRow[];
  nameA: string;
  nameB: string;
}

function winner(r: MetricRow): "a" | "b" | null {
  if (r.better === "none" || r.a == null || r.b == null || r.a === r.b) return null;
  const higher = (r.better ?? "higher") === "higher";
  return (r.a > r.b) === higher ? "a" : "b";
}

const bestStyle = {
  color: "var(--text)",
  background: "color-mix(in srgb, var(--accent) 18%, transparent)",
  boxShadow: "inset 0 0 0 1px color-mix(in srgb, var(--accent) 50%, transparent)",
};

export function MetricsTable({ rows, nameA, nameB }: Props) {
  let lastGroup: string | undefined;
  return (
    <div className="-mx-1 overflow-x-auto px-1">
      <table className="w-full border-separate border-spacing-y-1 text-[13px] sm:text-sm">
        <caption className="sr-only">Model A vs Model B metrics. The better value in each row is highlighted.</caption>
        <thead>
          <tr className="text-left">
            <th scope="col" className="py-2 pr-3 font-mono text-[11px] font-medium uppercase tracking-[0.15em] text-ink-faint">Metric</th>
            <th scope="col" className="w-[30%] px-1 py-2 text-right sm:w-auto sm:px-3">
              <span className="inline-flex items-center justify-end gap-1.5 font-semibold text-modelA"><ModelDot modelKey="model_a" />{nameA}</span>
            </th>
            <th scope="col" className="w-[30%] px-1 py-2 text-right sm:w-auto sm:px-3">
              <span className="inline-flex items-center justify-end gap-1.5 font-semibold text-modelB"><ModelDot modelKey="model_b" />{nameB}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const w = winner(r);
            const groupHeader = r.group && r.group !== lastGroup;
            lastGroup = r.group;
            return [
              groupHeader ? (
                <tr key={`g-${r.group}`}>
                  <th colSpan={3} scope="colgroup" className="pb-1 pt-4 text-left font-mono text-[10px] font-medium uppercase tracking-[0.2em] text-accent">
                    {r.group}
                  </th>
                </tr>
              ) : null,
              <tr key={r.label} className="group">
                <th scope="row" className="rounded-l-lg py-2 pr-2 text-left font-normal text-ink-muted group-hover:text-ink sm:pr-3">
                  {r.label}
                  {r.note && <span className="ml-1 hidden text-xs text-ink-faint sm:inline">{r.note}</span>}
                </th>
                {(["a", "b"] as const).map((k) => (
                  <td key={k} className="py-1 pl-1 text-right sm:px-1">
                    <span
                      className={`mono-num inline-block whitespace-nowrap rounded-md px-1.5 py-1 sm:px-2 ${w === k ? "font-semibold" : "text-ink-muted"}`}
                      style={w === k ? bestStyle : undefined}
                    >
                      {r.format(r[k])}
                      {w === k && <span className="sr-only"> (better)</span>}
                    </span>
                  </td>
                ))}
              </tr>,
            ];
          })}
        </tbody>
      </table>
    </div>
  );
}
