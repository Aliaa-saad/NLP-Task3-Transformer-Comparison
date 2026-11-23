import { motion, useReducedMotion } from "framer-motion";
import type { ModelKey, Prediction } from "../types/api";
import { ProbabilityBars } from "./ProbabilityBars";
import { PredictionCardSkeleton } from "./Skeleton";
import { ClassPill, CountUp, ModelDot } from "./ui";

interface Props {
  modelKey: ModelKey;
  name: string;
  prediction?: Prediction | null;
  loading?: boolean;
  /** Load error reported by /api/health (shown even before a prediction is made). */
  healthError?: string | null;
}

const MODEL_VAR: Record<ModelKey, string> = { model_a: "var(--model-a)", model_b: "var(--model-b)" };

function Shell({ modelKey, name, children, tag }: { modelKey: ModelKey; name: string; children: React.ReactNode; tag?: React.ReactNode }) {
  const color = MODEL_VAR[modelKey];
  return (
    <article
      className="glass lift relative min-h-[300px] overflow-hidden p-5 sm:p-6"
      aria-label={`${name} result`}
    >
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${color}, transparent)` }} />
      <div
        className="pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full blur-3xl"
        style={{ background: `color-mix(in srgb, ${color} 22%, transparent)` }}
      />
      <header className="relative flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex min-w-0 items-center gap-2 text-sm font-semibold sm:text-base" style={{ color }}>
          <ModelDot modelKey={modelKey} />
          <span className="truncate">{name}</span>
        </h3>
        {tag}
      </header>
      <div className="relative">{children}</div>
    </article>
  );
}

export function PredictionCard({ modelKey, name, prediction, loading, healthError }: Props) {
  const reduce = useReducedMotion();
  if (loading) return <PredictionCardSkeleton />;

  const error = prediction?.error ?? (prediction ? null : healthError);
  if (error) {
    return (
      <Shell modelKey={modelKey} name={name} tag={<span className="rounded-full border border-[color:var(--neg)] px-2.5 py-0.5 text-xs font-semibold text-neg">Unavailable</span>}>
        <div className="mt-6" role="alert">
          <p className="font-semibold text-ink">This model failed to load.</p>
          <p className="mt-1 text-sm text-ink-muted">The other model and the rest of the app keep working.</p>
          <pre className="mt-4 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-line p-3 font-mono text-xs text-ink-muted">
            {error}
          </pre>
        </div>
      </Shell>
    );
  }

  if (!prediction || !prediction.probs || !prediction.label) {
    return (
      <Shell modelKey={modelKey} name={name}>
        <div className="flex min-h-[220px] flex-col items-center justify-center text-center">
          <div className="mb-3 grid grid-cols-3 gap-1.5 opacity-60" aria-hidden="true">
            {[0.25, 0.5, 0.9].map((o, i) => (
              <span key={i} className="h-3 w-3 rounded-[4px]" style={{ background: MODEL_VAR[modelKey], opacity: o }} />
            ))}
          </div>
          <p className="text-sm text-ink-muted">Ready. Run both models to see this card fill in.</p>
        </div>
      </Shell>
    );
  }

  const p = prediction;
  const conf = p.probs![p.label!];
  return (
    <Shell
      modelKey={modelKey}
      name={name}
      tag={
        <div className="flex items-center gap-2">
          {p.demo_mode && <span className="rounded-full bg-amber-400/15 px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-amber-400">Demo · random weights</span>}
          <ClassPill label={p.label!} />
        </div>
      }
    >
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.2, 0.8, 0.2, 1] }}
      >
        <div className="mt-5 flex items-baseline gap-2">
          <span className="text-5xl font-bold tracking-tightish text-ink [font-variant-numeric:tabular-nums]">
            <CountUp value={conf * 100} format={(v) => v.toFixed(1)} />
            <span className="text-2xl text-ink-muted">%</span>
          </span>
          <span className="text-sm text-ink-muted">confidence</span>
        </div>
        <div className="mt-5">
          <ProbabilityBars probs={p.probs!} predicted={p.label} />
        </div>
        <p className="mono-num mt-5 flex flex-wrap gap-x-2 gap-y-1 text-xs text-ink-muted">
          <span>
            latency <span className="text-ink">{p.latency_ms?.toFixed(1)} ms</span>
          </span>
          <span aria-hidden="true">·</span>
          <span>tokenize {p.tokenize_ms?.toFixed(1)} ms</span>
          <span aria-hidden="true">·</span>
          <span>{p.num_tokens} tokens</span>
          <span aria-hidden="true">·</span>
          <span className={p.truncated ? "text-amber-400" : ""}>
            {p.truncated ? `truncated (${p.original_tokens}→${p.num_tokens})` : "not truncated"}
          </span>
          {modelKey === "model_a" && (
            <>
              <span aria-hidden="true">·</span>
              <span>{p.biases_applied ? "biases on" : "biases off"}</span>
            </>
          )}
        </p>
        {p.warning && <p className="mt-2 text-xs text-amber-400">{p.warning}</p>}
      </motion.div>
    </Shell>
  );
}
