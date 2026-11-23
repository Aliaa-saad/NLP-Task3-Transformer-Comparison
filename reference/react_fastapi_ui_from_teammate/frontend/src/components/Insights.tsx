import type { ReactNode } from "react";
import { Card, ModelDot, SectionTitle } from "./ui";
import { dec, duration, int, millions, ms, shortGpu, signed } from "../lib/format";
import { LABELS, type ComparisonView, type Label, type MetricsResponse, type ModelKey } from "../types/api";

const MODEL_VAR: Record<ModelKey, string> = { model_a: "var(--model-a)", model_b: "var(--model-b)" };

function N({ children }: { children: ReactNode }) {
  return <strong className="mono-num font-semibold text-ink">{children}</strong>;
}

function ModelHead({ k, label }: { k: ModelKey; label: string }) {
  return (
    <span className="inline-flex items-center justify-end gap-1.5 font-semibold" style={{ color: MODEL_VAR[k] }}>
      <ModelDot modelKey={k} />
      {label}
    </span>
  );
}

// ------------------------------------------------------------------ Efficiency
export function EfficiencySection({ data }: { data: MetricsResponse }) {
  const t = data.training;
  const cmp = data.comparison;
  const lat = cmp.latency_same_machine;
  const params = (k: ModelKey) => t[k]?.parameters ?? (cmp.models[k]?.parameters as number | undefined) ?? null;
  const regime = (k: ModelKey) => {
    const r = t[k]?.regime ?? (k === "model_a" ? "scratch training" : "fine-tuning");
    const g = shortGpu(t[k]?.gpu_name);
    return g ? `${r} (${g})` : r;
  };
  const gpuA = t.model_a?.gpu_name;
  const gpuB = t.model_b?.gpu_name;

  const rows: { label: string; a: ReactNode; b: ReactNode; sub?: ReactNode }[] = [
    { label: "Parameters", a: millions(params("model_a")), b: millions(params("model_b")), sub: "counted from the loaded checkpoints" },
    {
      label: "Avg inference latency",
      a: ms(lat?.model_a?.mean_ms),
      b: ms(lat?.model_b?.mean_ms),
      sub: lat ? `same machine: ${lat.gpu_name ?? lat.device ?? "?"} · median ${ms(lat.model_a?.median_ms)} / ${ms(lat.model_b?.median_ms)} · p95 ${ms(lat.model_a?.p95_ms)} / ${ms(lat.model_b?.p95_ms)}` : "not available",
    },
    {
      label: "Training time",
      a: (
        <>
          {duration(t.model_a?.training_time_sec)}
          <span className="block text-[11px] font-normal text-ink-faint">{regime("model_a")}</span>
        </>
      ),
      b: (
        <>
          {duration(t.model_b?.training_time_sec)}
          <span className="block text-[11px] font-normal text-ink-faint">{regime("model_b")}</span>
        </>
      ),
      sub: "wall-clock, from each model's own training run",
    },
    {
      label: "Training reviews",
      a: int(t.model_a?.train_size),
      b: int(t.model_b?.train_size),
      sub: `${t.model_a?.epochs ?? "?"} vs ${t.model_b?.epochs ?? "?"} epochs`,
    },
  ];

  return (
    <Card>
      <SectionTitle eyebrow="Efficiency" title="Size, speed and training cost" />
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-y-1 text-[13px] sm:text-sm">
          <caption className="sr-only">Efficiency of Model A vs Model B</caption>
          <thead>
            <tr>
              <th scope="col" className="py-2 pr-2 text-left font-mono text-[11px] font-medium uppercase tracking-[0.15em] text-ink-faint">Metric</th>
              <th scope="col" className="w-[30%] py-2 pl-1 text-right"><ModelHead k="model_a" label="Model A" /></th>
              <th scope="col" className="w-[30%] py-2 pl-1 text-right"><ModelHead k="model_b" label="Model B" /></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label} className="align-top">
                <th scope="row" className="py-2 pr-2 text-left font-normal text-ink-muted">
                  {r.label}
                  {r.sub && <span className="block text-[11px] text-ink-faint">{r.sub}</span>}
                </th>
                <td className="mono-num py-2 pl-1 text-right font-semibold text-ink">{r.a}</td>
                <td className="mono-num py-2 pl-1 text-right font-semibold text-ink">{r.b}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 rounded-xl border border-line p-3 text-xs text-ink-muted">
        <span className="font-semibold text-warn">Hardware differs · </span>
        Model A was trained from scratch on {gpuA ?? "an unrecorded GPU"}; Model B was fine-tuned on {gpuB ?? "an unrecorded GPU"}. Training times are
        therefore not directly comparable. Inference latency was measured for both models on the same machine
        {lat?.protocol ? ` (${lat.protocol})` : ""}.
      </p>
    </Card>
  );
}

// ------------------------------------------------------------------ Why the difference?
interface Delta {
  diff: number; // B - A (A logit-adjusted)
  lo: number;
  hi: number;
  tied: boolean;
}

function deltaBA(v: ComparisonView): Delta {
  const bs = v.bootstrap_a_adjusted_vs_b;
  const diff = bs ? -bs.macro_f1_diff_a_minus_b : v.model_b.f1_macro - v.model_a_logit_adjusted.f1_macro;
  const lo = bs ? -bs.ci95_high : NaN;
  const hi = bs ? -bs.ci95_low : NaN;
  return { diff, lo, hi, tied: bs ? lo <= 0 && hi >= 0 : false };
}

function ciText(d: Delta) {
  return Number.isNaN(d.lo) ? "" : `, 95% CI [${signed(d.lo)}, ${signed(d.hi)}]`;
}

export function WhySection({ data }: { data: MetricsResponse }) {
  const views = data.comparison.views;
  const nat = views.natural;
  const bal = views.balanced;
  if (!nat || !bal) return null;
  const dN = deltaBA(nat);
  const dB = deltaBA(bal);
  const a = (v: ComparisonView) => v.model_a_logit_adjusted;
  const b = (v: ComparisonView) => v.model_b;

  // per-class F1 gaps on the balanced view
  const gaps = LABELS.map((l) => ({ l, gap: (b(bal).per_class[l]?.f1 ?? 0) - (a(bal).per_class[l]?.f1 ?? 0) }));
  const biggest = gaps.reduce((x, y) => (Math.abs(y.gap) > Math.abs(x.gap) ? y : x));
  const neuA = a(bal).per_class.neutral;
  const neuB = b(bal).per_class.neutral;

  const lowest = (m: ComparisonView["model_b"]): Label =>
    LABELS.reduce((x, y) => ((m.per_class[y]?.f1 ?? 1) < (m.per_class[x]?.f1 ?? 1) ? y : x));
  const neutralLowestEverywhere = [a(nat), b(nat), a(bal), b(bal)].every((m) => lowest(m) === "neutral");

  const trainA = data.training.model_a?.train_size ?? null;
  const trainB = data.training.model_b?.train_size ?? null;
  const ratio = trainA && trainB ? trainA / trainB : null;
  const verb = (d: Delta) => (d.tied ? "matches" : d.diff > 0 ? "beats" : "trails");
  const truncA = String(data.comparison.models.model_a?.truncation ?? "head+tail");
  const truncB = String(data.comparison.models.model_b?.truncation ?? "right truncation");

  const leader = (d: Delta) => (d.diff > 0 ? "Model B" : "Model A");

  return (
    <Card>
      <SectionTitle eyebrow="Analysis" title="Why the difference?" right={<span className="text-xs text-ink-faint">generated from comparison_results.json</span>} />
      <div className="space-y-4 text-sm leading-relaxed text-ink-muted">
        <p>
          <span className="eyebrow mr-2">Natural</span>
          On the real class distribution (<N>{int(nat.size)}</N> reviews), Macro-F1 is <N>{dec(a(nat).f1_macro)}</N> for Model A vs{" "}
          <N>{dec(b(nat).f1_macro)}</N> for Model B: B − A = <N>{signed(dN.diff)}</N>
          {ciText(dN)}.{" "}
          {dN.tied ? (
            <>The interval contains 0, so the two models are <strong className="text-ink">statistically tied</strong>.</>
          ) : (
            <>{leader(dN)} is ahead and the interval excludes 0.</>
          )}
        </p>
        <p>
          <span className="eyebrow mr-2">Balanced</span>
          With equal class counts (<N>{int(bal.size)}</N> reviews), {dB.tied ? "the models are statistically tied" : <>{leader(dB)} is ahead</>}:{" "}
          <N>{dec(b(bal).f1_macro)}</N> vs <N>{dec(a(bal).f1_macro)}</N> (B − A = <N>{signed(dB.diff)}</N>
          {ciText(dB)}). The largest per-class gap is on <strong className="text-ink">{biggest.l}</strong> (F1 {signed(biggest.gap)}).
          Neutral recall is <N>{dec(neuA?.recall)}</N> (A) vs <N>{dec(neuB?.recall)}</N> (B), and neutral F1 is <N>{dec(neuA?.f1)}</N> vs{" "}
          <N>{dec(neuB?.f1)}</N>.
        </p>
        <p>
          <span className="eyebrow mr-2">Pretraining</span>
          Model B was fine-tuned on <N>{int(trainB)}</N> reviews{ratio ? <>, about <N>{Math.round(ratio)}×</N> fewer</> : null} than Model A's{" "}
          <N>{int(trainA)}</N>, yet it {verb(dN)} Model A on the natural view and {verb(dB)} it on the balanced view. Language knowledge from
          DistilBERT's pretraining stands in for millions of labeled Yelp examples.
        </p>
        <div>
          <p className="eyebrow mb-2">Limitations</p>
          <ul className="list-disc space-y-1.5 pl-5 marker:text-accent">
            <li>
              Different training data and distributions: Model A saw <N>{int(trainA)}</N> reviews with the natural (mostly positive) class
              balance, Model B <N>{int(trainB)}</N> balanced reviews. This compares full pipelines, not only architectures, and B's balanced
              training likely helps on the balanced view.
            </li>
            <li>
              The Model B checkpoint is the first fine-tuning run (<code className="font-mono text-xs text-ink">distilbert_yelp_sentiment</code>).
            </li>
            <li>
              Neutral (3★) is inherently ambiguous: those reviews mix praise and complaints.
              {neutralLowestEverywhere ? " It is the lowest-F1 class for both models in both views." : ""}
            </li>
            <li>
              Truncation differs: Model A keeps {truncA}, Model B uses {truncB}, so long reviews are seen differently.
            </li>
          </ul>
        </div>
      </div>
    </Card>
  );
}
