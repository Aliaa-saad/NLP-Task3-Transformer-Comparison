import type { ConfusionMatrixData, ModelKey } from "../types/api";
import { cap } from "../lib/format";
import { ModelDot } from "./ui";

/** Two-stop scale: black (#050607) -> deep green (#0E3B24) -> neon green (#3DFF8F). */
function heat(share: number): string {
  const t = Math.max(0, Math.min(1, share));
  if (t < 0.5) return `color-mix(in srgb, #0E3B24 ${Math.round((t / 0.5) * 100)}%, #050607)`;
  return `color-mix(in srgb, #3DFF8F ${Math.round(((t - 0.5) / 0.5) * 100)}%, #0E3B24)`;
}

interface Props {
  title: string;
  modelKey: ModelKey;
  data: ConfusionMatrixData;
}

/** Annotated heatmap (black → deep green → neon green), counts + row-normalised %, built with CSS grid. */
export function ConfusionMatrix({ title, modelKey, data }: Props) {
  const { labels, matrix } = data;
  const rowSums = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  const n = labels.length;

  return (
    <figure className="min-w-0">
      <figcaption className="mb-3 flex items-center gap-2 text-sm font-semibold text-ink">
        <ModelDot modelKey={modelKey} />
        {title}
      </figcaption>
      <div className="flex gap-2">
        <div className="flex items-center">
          <span className="eyebrow [writing-mode:vertical-rl] rotate-180 !text-[10px] !text-ink-faint">True label</span>
        </div>
        <div className="min-w-0 flex-1">
          <div className="grid gap-1.5" style={{ gridTemplateColumns: `minmax(56px,auto) repeat(${n}, minmax(0,1fr))` }} role="table" aria-label={`${title} confusion matrix`}>
            <span role="columnheader" />
            {labels.map((l) => (
              <span key={l} role="columnheader" className="pb-1 text-center font-mono text-[10px] uppercase tracking-wider text-ink-faint sm:text-[11px]">
                {cap(l).slice(0, 3)}
                <span className="hidden sm:inline">{l.slice(3)}</span>
              </span>
            ))}
            {matrix.map((row, i) => (
              <div key={labels[i]} role="row" className="contents">
                <span role="rowheader" className="flex items-center pr-1 text-xs font-medium text-ink-muted">
                  {cap(labels[i] ?? "")}
                </span>
                {row.map((count, j) => {
                  const share = rowSums[i] ? count / (rowSums[i] ?? 1) : 0;
                  const diag = i === j;
                  // black -> deep green -> neon green scale by row share
                  const bg = heat(share);
                  return (
                    <div
                      key={j}
                      role="cell"
                      className="flex aspect-[4/3] flex-col items-center justify-center rounded-lg text-center"
                      style={{
                        background: bg,
                        border: diag ? "1px solid rgba(61,255,143,0.6)" : "1px solid rgba(61,255,143,0.08)",
                        boxShadow: diag && share > 0.5 ? "0 0 18px -6px rgba(61,255,143,0.7)" : undefined,
                        color: share > 0.8 ? "#03140A" : "#E6FFF0",
                      }}
                      title={`True ${labels[i]}, predicted ${labels[j]}: ${count.toLocaleString()} (${(share * 100).toFixed(1)}% of row)`}
                    >
                      <span className="mono-num text-xs font-semibold sm:text-sm">{count.toLocaleString("en-US")}</span>
                      <span className="mono-num text-[10px] opacity-80 sm:text-[11px]">{(share * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
          <p className="mt-2 text-center font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faint">Predicted label</p>
        </div>
      </div>
    </figure>
  );
}
