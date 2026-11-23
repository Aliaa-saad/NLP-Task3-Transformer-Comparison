import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { LiveTicker, type LiveState } from "../components/LiveTicker";
import { ExampleChips } from "../components/ExampleChips";
import { PredictionCard } from "../components/PredictionCard";
import { ErrorBanner } from "../components/ErrorBanner";
import { ClassPill, FadeUp, Segmented, Spinner, Toggle } from "../components/ui";
import { useHealth, usePredict } from "../hooks/useApi";
import { isNetworkError } from "../api/client";
import { EXAMPLES, type Example } from "../lib/examples";
import { cap } from "../lib/format";
import type { Label, ModelKey, Prediction } from "../types/api";

const MAX_CHARS = 5000;
const RESULT_HOLD_MS = 9000;
const NAMES: Record<ModelKey, string> = {
  model_a: "Model A — Scratch Transformer (40.8M)",
  model_b: "Model B — DistilBERT fine-tuned (67.0M)",
};
type Expected = Label | "none";

function validate(text: string): string | null {
  if (!text.trim()) return "Please enter a review: the text is empty.";
  if (text.trim().length > MAX_CHARS) return `Review is too long: ${text.trim().length.toLocaleString()} characters (max ${MAX_CHARS.toLocaleString()}).`;
  return null;
}

function Hero({ live }: { live: LiveState | null }) {
  return (
    <section className="relative grid items-center gap-8 py-6 md:grid-cols-[1.15fr_1fr] md:py-10">
      <FadeUp>
        <p className="eyebrow mb-4">Sentiment · Transformers</p>
        <h1 className="text-[44px] font-extrabold leading-[1.02] tracking-tightish text-ink sm:text-[60px] lg:text-[72px]">
          Two Transformers.
          <br />
          One review<span className="neon-text">.</span>
        </h1>
        <p className="mt-5 max-w-xl text-base text-ink-muted sm:text-lg">
          Compare a Transformer built from scratch with fine-tuned DistilBERT on Yelp reviews. Paste a review and see both predictions side
          by side, with probabilities and measured latency.
        </p>
      </FadeUp>
      <FadeUp delay={0.1} className="flex justify-center md:justify-end">
        <div className="relative w-full max-w-[460px]">
          <div className="pointer-events-none absolute -inset-8 rounded-full blur-3xl" style={{ background: "radial-gradient(closest-side, rgba(61,255,143,0.14), transparent)" }} aria-hidden="true" />
          <LiveTicker live={live} />
        </div>
      </FadeUp>
    </section>
  );
}

function OverviewStrip() {
  const labels: { label: Label; stars: string }[] = [
    { label: "negative", stars: "1–2 ★" },
    { label: "neutral", stars: "3 ★" },
    { label: "positive", stars: "4–5 ★" },
  ];
  return (
    <FadeUp delay={0.15}>
      <section className="glass grid gap-5 p-5 sm:p-6 lg:grid-cols-[1.1fr_1fr_1fr]" aria-labelledby="overview-title">
        <div>
          <p className="eyebrow mb-1">What it does</p>
          <h2 id="overview-title" className="font-bold text-ink">3-class sentiment analysis of Yelp reviews</h2>
          <p className="mt-1 text-sm text-ink-muted">Each review is labelled by its star rating:</p>
          <ul className="mt-3 flex flex-wrap gap-2">
            {labels.map((l) => (
              <li key={l.label} className="flex items-center gap-1.5">
                <ClassPill label={l.label} size="sm" />
                <span className="font-mono text-xs text-ink-faint">{l.stars}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="border-line lg:border-l lg:pl-5">
          <p className="mb-1 flex items-center gap-2 text-sm font-semibold text-modelA">
            <span className="h-2.5 w-2.5 rounded-[3px] bg-modelA" aria-hidden="true" /> Model A
          </p>
          <p className="text-sm text-ink-muted">8-layer Pre-LN encoder Transformer (40.8M params) trained from scratch on ~6.9M reviews, with head+tail truncation and tuned logit biases.</p>
        </div>
        <div className="border-line lg:border-l lg:pl-5">
          <p className="mb-1 flex items-center gap-2 text-sm font-semibold text-modelB">
            <span className="h-2.5 w-2.5 rounded-[3px] bg-modelB" aria-hidden="true" /> Model B
          </p>
          <p className="text-sm text-ink-muted">Pretrained DistilBERT (67.0M params) fine-tuned on a balanced 120k-review sample, with standard right truncation.</p>
        </div>
      </section>
    </FadeUp>
  );
}

function verdictText(expected: Label, a: Prediction | null, b: Prediction | null): string {
  const okA = a?.label === expected;
  const okB = b?.label === expected;
  const lab = (p: Prediction | null) => p?.label ?? "no prediction";
  if (!a?.label && !b?.label) return "No prediction to check.";
  if (okA && okB) return `Both models got it right · ${expected}`;
  if (okB) return `Model B got it right · Model A predicted ${lab(a)}`;
  if (okA) return `Model A got it right · Model B predicted ${lab(b)}`;
  return `Both models missed it · A predicted ${lab(a)}, B predicted ${lab(b)} (expected ${expected})`;
}

export function AnalyzePage() {
  const [text, setText] = useState("");
  const [activeExample, setActiveExample] = useState<string | null>(null);
  const [applyBiases, setApplyBiases] = useState(true);
  const [expected, setExpected] = useState<Expected>("none");
  const [touchedError, setTouchedError] = useState<string | null>(null);
  const [showLiveResult, setShowLiveResult] = useState(false);
  const predict = usePredict();
  const health = useHealth();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const trimmedLen = text.trim().length;
  const over = trimmedLen > MAX_CHARS;
  const healthErr = (key: ModelKey) => {
    const m = health.data?.models.find((x) => x.model_key === key);
    return m && m.status === "error" ? m.message ?? "Model failed to load." : null;
  };

  // Hold the user's real result in the hero ticker for a few seconds, then resume the demo loop.
  useEffect(() => {
    if (!predict.data) return;
    setShowLiveResult(true);
    const t = window.setTimeout(() => setShowLiveResult(false), RESULT_HOLD_MS);
    return () => window.clearTimeout(t);
  }, [predict.data]);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const err = validate(text);
    setTouchedError(err);
    if (err) {
      textareaRef.current?.focus();
      return;
    }
    predict.mutate({ text: text.trim(), apply_biases: applyBiases });
  };

  const pick = (ex: Example) => {
    setText(ex.text);
    setActiveExample(ex.id);
    setTouchedError(null);
  };

  const results = predict.data?.results;
  const byKey = (k: ModelKey) => results?.find((r) => r.model_key === k) ?? null;
  const serverError = predict.error && !isNetworkError(predict.error) ? predict.error.message : null;
  const networkError = predict.error && isNetworkError(predict.error);
  const fieldError = touchedError ?? serverError;
  const ok = results?.filter((r) => !r.error && r.label) ?? [];
  const expectedLabel = expected === "none" ? null : expected;

  const live: LiveState | null = predict.isPending
    ? { status: "analyzing", text: predict.variables?.text ?? text, a: null, b: null }
    : showLiveResult && predict.data
      ? { status: "result", text: predict.variables?.text ?? "", a: byKey("model_a")?.label ?? null, b: byKey("model_b")?.label ?? null }
      : null;

  return (
    <div className="space-y-6">
      <Hero live={live} />
      <OverviewStrip />

      <FadeUp delay={0.2}>
        <form onSubmit={submit} className="glass p-5 sm:p-6" noValidate aria-labelledby="input-title">
          <div className="mb-3 flex items-end justify-between gap-3">
            <div>
              <p className="eyebrow mb-1">Input</p>
              <h2 id="input-title" className="text-lg font-bold text-ink sm:text-xl">
                <label htmlFor="review">Your review</label>
              </h2>
            </div>
            <span className={`mono-num text-xs ${over ? "font-semibold text-neg" : "text-ink-faint"}`} aria-live="polite" id="char-count">
              {trimmedLen.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            </span>
          </div>
          <textarea
            id="review"
            ref={textareaRef}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setActiveExample(null);
              if (touchedError) setTouchedError(validate(e.target.value));
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit();
            }}
            rows={6}
            placeholder="e.g. The tacos were incredible but the wait was way too long…"
            aria-invalid={!!fieldError || over}
            aria-describedby={`char-count${fieldError ? " review-error" : ""}`}
            className="block w-full resize-y rounded-xl border bg-transparent p-4 text-[15px] leading-relaxed text-ink placeholder:text-ink-faint focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
            style={{
              borderColor: fieldError || over ? "var(--neg)" : "var(--border)",
              background: "color-mix(in srgb, var(--bg-0) 55%, transparent)",
            }}
          />
          <div className="min-h-[24px] pt-1.5">
            {fieldError && (
              <p id="review-error" role="alert" className="text-sm text-neg">
                {fieldError}
              </p>
            )}
          </div>

          <div className="mt-2">
            <ExampleChips examples={EXAMPLES} activeId={activeExample} onPick={pick} disabled={predict.isPending} />
          </div>

          <div className="mt-5">
            <p className="eyebrow mb-2">Expected label (optional)</p>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <Segmented<Expected>
                label="Expected label (optional)"
                value={expected}
                onChange={setExpected}
                options={[
                  { value: "none", label: "None" },
                  { value: "negative", label: "Negative" },
                  { value: "neutral", label: "Neutral" },
                  { value: "positive", label: "Positive" },
                ]}
              />
              <span className="text-xs text-ink-faint">
                Checks this one review only. Overall accuracy is measured on thousands of held-out reviews:{" "}
                <Link to="/compare" className="text-accent underline-offset-2 hover:underline">see Compare</Link>.
              </span>
            </div>
          </div>

          <div className="mt-6 flex flex-col gap-4 border-t border-line pt-5 sm:flex-row sm:items-center sm:justify-between">
            <Toggle
              id="biases"
              checked={applyBiases}
              onChange={setApplyBiases}
              label="Apply tuned logit biases (Model A)"
              hint="argmax(logits + b), biases tuned on validation for Macro-F1"
            />
            <button type="submit" className="btn-primary w-full sm:w-auto" disabled={predict.isPending} data-loading={predict.isPending}>
              {predict.isPending ? <Spinner /> : (
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z" /></svg>
              )}
              {predict.isPending ? "Running…" : "Run both models"}
            </button>
          </div>
        </form>
      </FadeUp>

      {networkError && (
        <ErrorBanner title="Prediction failed: backend unreachable" command={String.raw`cd app\backend; .\.venv\Scripts\python -m uvicorn app.main:app --port 8000`}>
          Start the API, then try again.
        </ErrorBanner>
      )}

      <section aria-labelledby="results-title" aria-busy={predict.isPending}>
        <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
          <div>
            <p className="eyebrow mb-1">Results</p>
            <h2 id="results-title" className="text-lg font-bold text-ink sm:text-xl">Side-by-side inference</h2>
          </div>
          {predict.data && (
            <p className="mono-num text-xs text-ink-faint" title="Browser → API → browser, includes both models, tokenization and network">
              round-trip {predict.data.roundTripMs.toFixed(0)} ms · server device {predict.data.device}
              <span className="block text-[10px] sm:text-right">(model latency is measured on the server, forward pass only)</span>
            </p>
          )}
        </div>
        <div className="grid gap-5 md:grid-cols-2">
          {(["model_a", "model_b"] as const).map((k) => (
            <PredictionCard
              key={k}
              modelKey={k}
              name={NAMES[k]}
              prediction={byKey(k)}
              loading={predict.isPending}
              healthError={healthErr(k)}
              expected={expectedLabel}
            />
          ))}
        </div>

        <div className="mt-5 min-h-[56px] space-y-3">
          <AnimatePresence mode="wait">
            {!predict.isPending && results && (
              <motion.div
                key={String(predict.data?.agree) + ok.map((r) => r.label).join()}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.35, ease: [0.2, 0.8, 0.2, 1] }}
                className="glass flex flex-wrap items-center justify-center gap-3 px-5 py-4 text-center"
                role="status"
              >
                {predict.data?.agree === true && ok[0]?.label ? (
                  <>
                    <span className="text-lg text-accent" aria-hidden="true">✓</span>
                    <span className="font-semibold text-ink">Both models agree:</span>
                    <ClassPill label={ok[0].label} />
                  </>
                ) : predict.data?.agree === false ? (
                  <>
                    <span className="text-lg text-warn" aria-hidden="true">≠</span>
                    <span className="font-semibold text-ink">Models disagree</span>
                    {ok.map((r) => (
                      <span key={r.model_key} className="flex items-center gap-1.5 text-sm text-ink-muted">
                        {r.model_key === "model_a" ? "A" : "B"}: <ClassPill label={r.label!} size="sm" />
                      </span>
                    ))}
                  </>
                ) : (
                  <span className="text-sm text-ink-muted">Can't compare: one model did not return a prediction.</span>
                )}
              </motion.div>
            )}
          </AnimatePresence>
          {!predict.isPending && results && expectedLabel && (
            <p className="text-center text-sm text-ink-muted" role="status" data-testid="verdict">
              <span className="font-semibold text-ink">{verdictText(expectedLabel, byKey("model_a"), byKey("model_b"))}</span>
              <span className="mt-1 block text-xs text-ink-faint">
                Expected: {cap(expectedLabel)}, set by you. One sentence is an anecdote, not a benchmark:{" "}
                <Link to="/compare" className="text-accent underline-offset-2 hover:underline">overall performance is on the Compare page</Link>.
              </span>
            </p>
          )}
        </div>
      </section>
    </div>
  );
}
