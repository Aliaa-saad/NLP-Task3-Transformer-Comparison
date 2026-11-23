import { useMemo, type CSSProperties } from "react";
import type { Label } from "../types/api";
import { useTheme } from "../hooks/useTheme";

interface Props {
  state: "idle" | "running";
  flash: Label | null;
  size?: number;
}

const N = 7;

/** Decorative 7×7 "attention matrix" whose cells breathe in a diagonal wave. */
export function AttentionGrid({ state, flash }: Props) {
  const { palette } = useTheme();
  const cells = useMemo(() => {
    const out: { i: number; j: number; o: number; d: number }[] = [];
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const dist = Math.abs(i - j);
        // Deterministic pseudo-noise so off-diagonal cells vary without re-rendering differently.
        const noise = ((Math.sin(i * 12.9898 + j * 78.233) * 43758.5453) % 1 + 1) % 1;
        const o = Math.min(1, 0.1 + 0.85 * Math.exp(-dist * 0.85) + noise * 0.12);
        out.push({ i, j, o, d: (i + j) * 110 });
      }
    }
    return out;
  }, []);

  return (
    <div
      className="attn-grid grid aspect-square w-full max-w-[340px] grid-cols-7 gap-[7%]"
      style={{ gap: "clamp(6px, 1.6vw, 12px)", ["--flash" as string]: flash ? palette.cls[flash] : "transparent" } as CSSProperties}
      data-state={state}
      data-flash={flash ? "true" : "false"}
      aria-hidden="true"
    >
      {cells.map((c) => (
        <span
          key={`${c.i}-${c.j}`}
          className="attn-cell aspect-square"
          data-diag={c.i === c.j}
          style={{ ["--o" as string]: c.o.toFixed(3), ["--d" as string]: `${c.d}ms` } as CSSProperties}
        />
      ))}
    </div>
  );
}
