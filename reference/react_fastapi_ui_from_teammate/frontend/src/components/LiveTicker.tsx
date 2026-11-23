import { useEffect, useState, type CSSProperties } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useTickerPredictions } from "../hooks/useApi";
import { TICKER_REVIEWS } from "../lib/examples";
import { classColorVar } from "../lib/colors";
import type { Label } from "../types/api";

export interface LiveState {
  status: "analyzing" | "result";
  text: string;
  a: Label | null;
  b: Label | null;
}

interface Props {
  /** A real prediction in flight / just finished; null = demo loop. */
  live: LiveState | null;
}

const CARD_H = 248;
const TYPE_MS = 34;
const HOLD_MS = 2400;
const FADE_MS = 450;

function excerpt(text: string, n = 150): string {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
}

function Chip({ model, label, delay }: { model: "A" | "B"; label: Label | null | "offline"; delay: number }) {
  const reduce = useReducedMotion();
  const modelColor = model === "A" ? "var(--model-a)" : "var(--model-b)";
  const text = label === "offline" ? "API offline" : label ?? "…";
  return (
    <motion.span
      initial={reduce ? false : { opacity: 0, scale: 0.85, y: 4 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.3, delay, ease: [0.2, 0.8, 0.2, 1] }}
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-mono text-xs"
      style={{
        color: modelColor,
        border: `1px solid color-mix(in srgb, ${modelColor} 45%, transparent)`,
        background: `color-mix(in srgb, ${modelColor} 8%, transparent)`,
      }}
    >
      {model} →
      {label && label !== "offline" && <span className="h-1.5 w-1.5 rounded-full" style={{ background: classColorVar(label) }} aria-hidden="true" />}
      <span className="text-ink">{text}</span>
    </motion.span>
  );
}

/** Terminal-style card that types out hand-written reviews and shows both models' real predictions. */
export function LiveTicker({ live }: Props) {
  const reduce = useReducedMotion();
  const preds = useTickerPredictions(TICKER_REVIEWS);
  const [idx, setIdx] = useState(0);
  const [chars, setChars] = useState(0);
  const [phase, setPhase] = useState<"typing" | "hold" | "fade">("typing");

  const review = TICKER_REVIEWS[idx % TICKER_REVIEWS.length] ?? "";

  // Demo loop state machine (paused while a real prediction is shown).
  useEffect(() => {
    if (live || reduce) return;
    let t: number;
    if (phase === "typing") {
      if (chars < review.length) t = window.setTimeout(() => setChars((c) => c + 1), TYPE_MS);
      else t = window.setTimeout(() => setPhase("hold"), 120);
    } else if (phase === "hold") {
      t = window.setTimeout(() => setPhase("fade"), HOLD_MS);
    } else {
      t = window.setTimeout(() => {
        setIdx((i) => (i + 1) % TICKER_REVIEWS.length);
        setChars(0);
        setPhase("typing");
      }, FADE_MS);
    }
    return () => window.clearTimeout(t);
  }, [live, reduce, phase, chars, review.length]);

  const demoLabels = (text: string) => {
    if (preds.isError) return { a: "offline" as const, b: "offline" as const };
    const p = preds.data?.[text];
    return { a: p?.a ?? null, b: p?.b ?? null };
  };

  const status = live ? (live.status === "analyzing" ? "analyzing…" : "your review") : "demo loop";

  let body: React.ReactNode;
  if (live?.status === "analyzing") {
    body = (
      <div key="analyzing">
        <p className="font-mono text-sm leading-relaxed text-ink-muted">{excerpt(live.text)}</p>
        <p className="mt-4 flex items-center gap-2 font-mono text-sm text-accent">
          <span className="pulse-dot inline-block h-2 w-2 rounded-full bg-accent shadow-neon-sm" aria-hidden="true" />
          analyzing…
        </p>
      </div>
    );
  } else if (live?.status === "result") {
    body = (
      <div key={`result-${live.text}`}>
        <p className="font-mono text-sm leading-relaxed text-ink">{excerpt(live.text)}</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Chip model="A" label={live.a} delay={0.05} />
          <Chip model="B" label={live.b} delay={0.18} />
        </div>
      </div>
    );
  } else {
    const shown = reduce ? review : review.slice(0, chars);
    const labels = demoLabels(review);
    body = (
      <motion.div
        key={`demo-${idx}`}
        animate={{ opacity: phase === "fade" ? 0 : 1 }}
        transition={{ duration: FADE_MS / 1000 }}
      >
        <p className="font-mono text-sm leading-relaxed text-ink">
          <span className="text-accent">&gt; </span>
          {shown}
          {!reduce && phase === "typing" && <span className="ticker-cursor" aria-hidden="true" />}
        </p>
        {(reduce || phase !== "typing") && (
          <div className="mt-4 flex flex-wrap gap-2">
            <Chip model="A" label={labels.a} delay={0.05} />
            <Chip model="B" label={labels.b} delay={0.2} />
          </div>
        )}
      </motion.div>
    );
  }

  return (
    <div
      className="glass relative w-full max-w-[460px] overflow-hidden"
      style={{ height: CARD_H, ["--scan-h" as string]: `${CARD_H}px` } as CSSProperties}
      role="region"
      aria-label="Live inference ticker: example reviews with both models' real predictions"
    >
      {!reduce && <div className="ticker-scan" aria-hidden="true" />}
      <div className="flex items-center gap-2 border-b border-line px-4 py-2.5">
        <span className="flex gap-1.5" aria-hidden="true">
          <span className="h-2.5 w-2.5 rounded-full bg-[#FF4D6D]/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-[#FFC857]/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-[#3DFF8F]/80" />
        </span>
        <span className="ml-1 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-muted">live inference</span>
        <span className="ml-auto font-mono text-[10px] text-ink-faint" aria-live="polite">
          {status}
        </span>
      </div>
      <div className="px-5 py-5">
        <AnimatePresence mode="wait">{body}</AnimatePresence>
      </div>
      <p className="absolute bottom-3 left-5 right-5 font-mono text-[10px] text-ink-faint">
        {live ? "real prediction for your input" : "hand-written demo reviews · labels are real model outputs"}
      </p>
    </div>
  );
}
