import { useMemo, useState } from "react";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { ErrorBanner } from "../components/ErrorBanner";
import { EfficiencySection, WhySection } from "../components/Insights";
import { MetricsTable, type MetricRow } from "../components/MetricsTable";
import { GroupedBarChart, TrainingCurves } from "../components/Charts";
import { BlockSkeleton, Skeleton } from "../components/Skeleton";
import { Card, ClassPill, CountUp, FadeUp, Segmented, SectionTitle } from "../components/ui";
import { useErrors, useHistory, useMetrics } from "../hooks/useApi";
import { isNetworkError } from "../api/client";
import { cap, dec, duration, int, millions, ms, pct, signed } from "../lib/format";
import { LABELS, type ComparisonResults, type ComparisonView, type EvalMetrics, type Label, type MetricsResponse } from "../types/api";

type ViewKey = "balanced" | "natural";
type AMode = "adjusted" | "raw";

const VIEW_HELP: Record<ViewKey, string> = {
  balanced: "Equal number of reviews per class: Macro-F1 treats negative, neutral and positive equally.",
  natural: "Real Yelp class distribution (mostly positive): closer to what a deployed model would see.",
};

function Kpi({ label, value, sub, color }: { label: string; value: React.ReactNode; sub?: React.ReactNode; color?: string }) {
  return (
    <div className="glass lift p-5">
      <p className="eyebrow mb-2 !text-ink-faint" style={color ? { color } : undefined}>{label}</p>
      <p className="mono-num text-3xl font-semibold tracking-tight text-ink sm:text-4xl">{value}</p>
      {sub && <p className="mt-1.5 text-xs text-ink-muted">{sub}</p>}
    </div>
  );
}

function buildRows(a: EvalMetrics, b: EvalMetrics, data: MetricsResponse): MetricRow[] {
  const t = data.training;
  const lat = data.comparison.latency_same_machine;
  const rows: MetricRow[] = [
    { group: "Overall", label: "Accuracy", a: a.accuracy, b: b.accuracy, format: pct },
    { group: "Overall", label: "Macro-F1", a: a.f1_macro, b: b.f1_macro, format: (v) => dec(v) },
    { group: "Overall", label: "Weighted-F1", a: a.f1_weighted, b: b.f1_weighted, format: (v) => dec(v) },
    { group: "Overall", label: "Macro precision", a: a.precision_macro, b: b.precision_macro, format: (v) => dec(v) },
    { group: "Overall", label: "Macro recall", a: a.recall_macro, b: b.recall_macro, format: (v) => dec(v) },
  ];
  for (const l of LABELS) {
    const pa = a.per_class[l];
    const pb = b.per_class[l];
    rows.push(
      { group: "Per class", label: `${cap(l)} precision`, a: pa?.precision, b: pb?.precision, format: (v) => dec(v) },
      { group: "Per class", label: `${cap(l)} recall`, a: pa?.recall, b: pb?.recall, format: (v) => dec(v) },
      { group: "Per class", label: `${cap(l)} F1`, a: pa?.f1, b: pb?.f1, format: (v) => dec(v) },
    );
  }
  const params = (k: "model_a" | "model_b") =>
    t[k]?.parameters ?? (data.comparison.models[k]?.parameters as number | undefined) ?? null;
  rows.push(
    { group: "Cost", label: "Parameters", a: params("model_a"), b: params("model_b"), format: millions, better: "lower" },
    { group: "Cost", label: "Training time", a: t.model_a?.training_time_sec, b: t.model_b?.training_time_sec, format: duration, better: "lower", note: "(own GPU runs)" },
  );
  if (lat?.model_a || lat?.model_b) {
    rows.push(
      { group: "Cost", label: "Latency (mean, offline)", a: lat.model_a?.mean_ms, b: lat.model_b?.mean_ms, format: ms, better: "lower" },
      { group: "Cost", label: "Latency (p95, offline)", a: lat.model_a?.p95_ms, b: lat.model_b?.p95_ms, format: ms, better: "lower" },
    );
  }
  return rows;
}

function EvalSliceNote({ cmp, view }: { cmp: ComparisonResults; view: ComparisonView }) {
  const s = cmp.eval_slice;
  const num = (k: string) => (typeof s[k] === "number" ? (s[k] as number) : null);
  const counts = s.natural_class_counts as Record<string, number> | undefined;
  return (
    <Card>
      <SectionTitle eyebrow="Methodology" title="Evaluation slice" />
      {cmp.caveat && (
        <p className="rounded-xl border border-[color-mix(in_srgb,var(--warn)_35%,transparent)] bg-[color-mix(in_srgb,var(--warn)_6%,transparent)] p-3 text-sm text-ink-muted">
          <span className="font-semibold text-warn">Caveat · </span>
          {cmp.caveat}
        </p>
      )}
      {cmp.description && <p className="mt-3 text-sm text-ink-muted">{cmp.description}</p>}
      <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ["This view", int(view.size)],
          ["Natural size", int(num("natural_size"))],
          ["Balanced size", int(num("balanced_size"))],
          ["Removed (seen by B)", int(num("removed_seen_by_model_b"))],
        ].map(([k, v]) => (
          <div key={k} className="rounded-xl border border-line p-3">
            <dt className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-faint">{k}</dt>
            <dd className="mono-num mt-1 text-lg font-semibold text-ink">{v}</dd>
          </div>
        ))}
      </dl>
      {counts && (
        <p className="mt-3 flex flex-wrap items-center gap-2 text-xs text-ink-muted">
          Natural class counts:
          {LABELS.filter((l) => counts[l] != null).map((l) => (
            <span key={l} className="flex items-center gap-1">
              <ClassPill label={l} size="sm" /> <span className="mono-num">{int(counts[l])}</span>
            </span>
          ))}
        </p>
      )}
    </Card>
  );
}

function ErrorsTable() {
  const errors = useErrors(20);
  return (
    <Card>
      <SectionTitle eyebrow="Bonus" title="Model A: highest-confidence errors" right={<span className="text-xs text-ink-faint">own test set · logit-adjusted</span>} />
      {errors.isPending ? (
        <div className="space-y-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-10 w-full" />)}</div>
      ) : errors.isError ? (
        <p className="text-sm text-ink-muted">error_analysis_top50.csv not available ({errors.error.message}).</p>
      ) : (
        <ol className="space-y-2">
          {errors.data.map((r) => (
            <li key={r.rank} className="rounded-xl border border-line p-3">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="mono-num text-ink-faint">#{r.rank}</span>
                <span className="text-ink-muted">true</span>
                {LABELS.includes(r.true_label as Label) ? <ClassPill label={r.true_label as Label} size="sm" /> : r.true_label}
                <span className="text-ink-muted">→ predicted</span>
                {LABELS.includes(r.pred_label as Label) ? <ClassPill label={r.pred_label as Label} size="sm" /> : r.pred_label}
                <span className="mono-num ml-auto text-ink-muted">conf {pct(r.confidence)}</span>
              </div>
              <details className="group mt-2">
                <summary className="cursor-pointer list-none text-sm text-ink-muted [&::-webkit-details-marker]:hidden">
                  <span className="line-clamp-2 group-open:line-clamp-none whitespace-pre-line">{r.text}</span>
                  <span className="mt-1 inline-block text-xs text-accent group-open:hidden">Show full review</span>
                </summary>
              </details>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}

function HistoryCard() {
  const history = useHistory();
  return (
    <Card>
      <SectionTitle eyebrow="Model A" title="Training curves" right={<span className="text-xs text-ink-faint">training_history.csv</span>} />
      {history.isPending ? (
        <div className="skeleton h-[220px] w-full" />
      ) : history.isError ? (
        <p className="text-sm text-ink-muted">Training history not available ({history.error.message}).</p>
      ) : (
        <TrainingCurves rows={history.data} />
      )}
    </Card>
  );
}

export function ComparePage() {
  const metrics = useMetrics();
  const [view, setView] = useState<ViewKey>("balanced");
  const [aMode, setAMode] = useState<AMode>("adjusted");

  const derived = useMemo(() => {
    if (!metrics.data) return null;
    const v = metrics.data.comparison.views[view] ?? Object.values(metrics.data.comparison.views)[0];
    if (!v) return null;
    const a = aMode === "adjusted" ? v.model_a_logit_adjusted : v.model_a_raw;
    return { v, a, b: v.model_b };
  }, [metrics.data, view, aMode]);

  const controls = (
    <div className="flex flex-wrap items-center gap-3">
      <Segmented<ViewKey> label="Evaluation view" value={view} onChange={setView} options={[{ value: "balanced", label: "Balanced" }, { value: "natural", label: "Natural" }]} />
      <Segmented<AMode> label="Model A metrics" value={aMode} onChange={setAMode} options={[{ value: "adjusted", label: "A: logit-adjusted" }, { value: "raw", label: "A: raw" }]} />
    </div>
  );

  return (
    <div className="space-y-6">
      <FadeUp>
        <section className="pt-4">
          <p className="eyebrow mb-3">Comparison dashboard</p>
          <h1 className="text-4xl font-extrabold tracking-tightish text-ink sm:text-5xl">
            Head to head<span className="neon-text">.</span>
          </h1>
          <p className="mt-3 max-w-2xl text-ink-muted">
            Both models evaluated on the <strong className="text-ink">same held-out reviews</strong> that neither model saw during training.
          </p>
          <div className="mt-5">{controls}</div>
          <p className="mt-2 text-sm text-ink-faint" aria-live="polite">{VIEW_HELP[view]}</p>
        </section>
      </FadeUp>

      {metrics.isPending && (
        <div className="space-y-6" aria-busy="true">
          <div className="grid gap-4 sm:grid-cols-3">{[0, 1, 2].map((i) => <div key={i} className="glass p-5"><Skeleton className="h-3 w-24" /><Skeleton className="mt-3 h-9 w-28" /></div>)}</div>
          <BlockSkeleton />
          <BlockSkeleton height={340} />
        </div>
      )}

      {metrics.isError && (
        <ErrorBanner
          title={isNetworkError(metrics.error) ? "Backend unreachable" : "Comparison results not available"}
          command={isNetworkError(metrics.error) ? String.raw`cd app\backend; .\.venv\Scripts\python -m uvicorn app.main:app --port 8000` : undefined}
        >
          {isNetworkError(metrics.error)
            ? "Start the API to load the dashboard."
            : `${metrics.error.message}. Copy comparison_results.json (from the comparison notebook) into app/backend/artifacts/comparison/. Model A's training curves and error analysis below still work.`}
        </ErrorBanner>
      )}

      {metrics.data && derived && (
        <>
          <FadeUp delay={0.05} className="grid gap-4 sm:grid-cols-3">
            <Kpi
              label={`Macro-F1 · Model A${aMode === "raw" ? " (raw)" : ""}`}
              color="var(--model-a)"
              value={<CountUp value={derived.a.f1_macro} format={(x) => x.toFixed(3)} />}
              sub={`accuracy ${pct(derived.a.accuracy)}`}
            />
            <Kpi label="Macro-F1 · Model B" color="var(--model-b)" value={<CountUp value={derived.b.f1_macro} format={(x) => x.toFixed(3)} />} sub={`accuracy ${pct(derived.b.accuracy)}`} />
            {(() => {
              const bs = derived.v.bootstrap_a_adjusted_vs_b;
              const delta = derived.b.f1_macro - derived.a.f1_macro;
              return (
                <Kpi
                  label="Δ Macro-F1 (B − A)"
                  value={<CountUp value={delta} format={(x) => signed(x)} />}
                  sub={
                    bs ? (
                      <>
                        95% CI [{signed(-bs.ci95_high)}, {signed(-bs.ci95_low)}]
                        {aMode === "raw" && <span className="block text-ink-faint">CI computed for logit-adjusted A</span>}
                      </>
                    ) : (
                      "no bootstrap available"
                    )
                  }
                />
              );
            })()}
          </FadeUp>

          {derived.v.bootstrap_a_adjusted_vs_b && (
            <FadeUp delay={0.08}>
              <p className="glass px-5 py-4 text-sm text-ink-muted">
                <span className="eyebrow mr-2">Bootstrap</span>
                Macro-F1 difference (B − A, A logit-adjusted):{" "}
                <strong className="mono-num text-ink">{signed(-derived.v.bootstrap_a_adjusted_vs_b.macro_f1_diff_a_minus_b)}</strong>, 95% CI{" "}
                <span className="mono-num text-ink">
                  [{signed(-derived.v.bootstrap_a_adjusted_vs_b.ci95_high)}, {signed(-derived.v.bootstrap_a_adjusted_vs_b.ci95_low)}]
                </span>
                {derived.v.bootstrap_a_adjusted_vs_b.p_a_better != null && (
                  <> · P(A better) = <span className="mono-num text-ink">{dec(derived.v.bootstrap_a_adjusted_vs_b.p_a_better, 3)}</span></>
                )}
                {derived.v.agreement_rate_a_adjusted_vs_b != null && (
                  <> · models agree on <span className="mono-num text-ink">{pct(derived.v.agreement_rate_a_adjusted_vs_b)}</span> of reviews</>
                )}
              </p>
            </FadeUp>
          )}

          <FadeUp delay={0.1} className="grid gap-6 lg:grid-cols-2">
            <Card>
              <SectionTitle eyebrow="Overall" title="Accuracy · Macro-F1 · Weighted-F1" />
              <GroupedBarChart
                key={`${view}-${aMode}`}
                nameA="Model A"
                nameB="Model B"
                data={[
                  { name: "Accuracy", a: derived.a.accuracy, b: derived.b.accuracy },
                  { name: "Macro-F1", a: derived.a.f1_macro, b: derived.b.f1_macro },
                  { name: "Weighted-F1", a: derived.a.f1_weighted ?? 0, b: derived.b.f1_weighted ?? 0 },
                ]}
              />
            </Card>
            <Card>
              <SectionTitle eyebrow="Per class" title="F1 by sentiment class" />
              <GroupedBarChart
                key={`pc-${view}-${aMode}`}
                nameA="Model A"
                nameB="Model B"
                data={LABELS.map((l) => ({ name: cap(l), a: derived.a.per_class[l]?.f1 ?? 0, b: derived.b.per_class[l]?.f1 ?? 0 }))}
              />
            </Card>
          </FadeUp>

          <FadeUp delay={0.12}>
            <Card>
              <SectionTitle eyebrow="Metrics" title="Model A vs Model B" right={<span className="text-xs text-ink-faint">best value per row highlighted</span>} />
              <MetricsTable rows={buildRows(derived.a, derived.b, metrics.data)} nameA={aMode === "raw" ? "Model A (raw)" : "Model A"} nameB="Model B" />
            </Card>
          </FadeUp>

          <FadeUp delay={0.14}>
            <Card>
              <SectionTitle eyebrow="Errors" title="Confusion matrices" right={<span className="text-xs text-ink-faint">counts + row %</span>} />
              <div className="grid gap-8 md:grid-cols-2">
                <ConfusionMatrix title={aMode === "raw" ? "Model A (raw)" : "Model A (logit-adjusted)"} modelKey="model_a" data={derived.a.confusion_matrix} />
                <ConfusionMatrix title="Model B" modelKey="model_b" data={derived.b.confusion_matrix} />
              </div>
            </Card>
          </FadeUp>

          <FadeUp delay={0.16} className="grid gap-6 lg:grid-cols-2">
            <EfficiencySection data={metrics.data} />
            <EvalSliceNote cmp={metrics.data.comparison} view={derived.v} />
          </FadeUp>

          <FadeUp delay={0.17}>
            <WhySection data={metrics.data} />
          </FadeUp>
        </>
      )}

      {!metrics.isPending && (
        <>
          <FadeUp delay={0.18}>
            <HistoryCard />
          </FadeUp>
          <FadeUp delay={0.2}>
            <ErrorsTable />
          </FadeUp>
        </>
      )}
    </div>
  );
}
