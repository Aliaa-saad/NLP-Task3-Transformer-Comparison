import { motion, useReducedMotion } from "framer-motion";
import { LABELS, type Label, type Probs } from "../types/api";
import { cap } from "../lib/format";
import { classColorVar } from "../lib/colors";

interface Props {
  probs: Probs;
  predicted: Label | null;
}

/** Horizontal bars for the three class probabilities; widths animate in from 0. */
export function ProbabilityBars({ probs, predicted }: Props) {
  const reduce = useReducedMotion();
  return (
    <ul className="space-y-3" aria-label="Class probabilities">
      {LABELS.map((label, i) => {
        const p = probs[label];
        const isTop = label === predicted;
        const color = classColorVar(label);
        return (
          <li key={label} className="grid grid-cols-[72px_1fr_56px] items-center gap-3">
            <span className={`text-sm ${isTop ? "font-semibold text-ink" : "text-ink-muted"}`}>{cap(label)}</span>
            <div
              className="h-2.5 overflow-hidden rounded-full"
              style={{ background: "color-mix(in srgb, var(--accent) 10%, transparent)" }}
              role="meter"
              aria-label={`${cap(label)} probability`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(p * 100)}
            >
              <motion.div
                className="h-full rounded-full"
                style={{ background: color, opacity: isTop ? 1 : 0.55 }}
                initial={reduce ? false : { width: 0 }}
                animate={{ width: `${Math.max(p * 100, 0.5)}%` }}
                transition={{ duration: 0.6, delay: 0.08 * i, ease: [0.2, 0.8, 0.2, 1] }}
              />
            </div>
            <span className={`mono-num text-right text-sm ${isTop ? "text-ink" : "text-ink-muted"}`}>{(p * 100).toFixed(1)}%</span>
          </li>
        );
      })}
    </ul>
  );
}
