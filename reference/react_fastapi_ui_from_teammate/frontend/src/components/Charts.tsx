import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";
import { useReducedMotion } from "framer-motion";
import { useTheme } from "../hooks/useTheme";
import type { HistoryRow } from "../types/api";

interface GroupedDatum {
  name: string;
  a: number;
  b: number;
}

function ChartTooltip({ active, payload, label, digits = 3 }: TooltipProps<number, string> & { digits?: number }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass !rounded-xl px-3 py-2 text-xs" style={{ background: "var(--bg-1)" }}>
      <p className="mb-1 font-semibold text-ink">{label}</p>
      {payload.map((p) => (
        <p key={String(p.dataKey)} className="mono-num flex items-center gap-2 text-ink-muted">
          <span className="h-2 w-2 rounded-sm" style={{ background: p.color }} />
          {p.name}: <span className="text-ink">{typeof p.value === "number" ? p.value.toFixed(digits) : p.value}</span>
        </p>
      ))}
    </div>
  );
}

function useAxis() {
  const { palette } = useTheme();
  return {
    palette,
    tick: { fill: palette.muted, fontSize: 11 },
    axisLine: { stroke: palette.grid },
  };
}

export function GroupedBarChart({ data, nameA, nameB, height = 280 }: { data: GroupedDatum[]; nameA: string; nameB: string; height?: number }) {
  const { palette, tick, axisLine } = useAxis();
  const reduce = useReducedMotion();
  const values = data.flatMap((d) => [d.a, d.b]);
  const min = Math.max(0, Math.floor((Math.min(...values) - 0.05) * 10) / 10);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barGap={4} barCategoryGap="28%">
        <CartesianGrid vertical={false} stroke={palette.grid} />
        <XAxis dataKey="name" tick={tick} axisLine={axisLine} tickLine={false} />
        <YAxis domain={[min, 1]} tick={tick} axisLine={false} tickLine={false} tickFormatter={(v: number) => v.toFixed(2)} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(61,255,143,0.06)" }} />
        <Legend wrapperStyle={{ fontSize: 12, color: palette.muted }} iconType="square" />
        <Bar dataKey="a" name={nameA} fill={palette.model.model_a} radius={[6, 6, 0, 0]} isAnimationActive={!reduce} animationDuration={700} />
        <Bar dataKey="b" name={nameB} fill={palette.model.model_b} radius={[6, 6, 0, 0]} isAnimationActive={!reduce} animationDuration={700} animationBegin={120} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function TrainingCurves({ rows }: { rows: HistoryRow[] }) {
  const { palette, tick, axisLine } = useAxis();
  const reduce = useReducedMotion();
  const data = rows.map((r) => ({ ...r, x: r.epoch ?? r.step }));
  const common = {
    data,
    margin: { top: 8, right: 8, left: -12, bottom: 0 },
  };
  const xAxis = (
    <XAxis
      dataKey="x"
      type="number"
      domain={["dataMin", "dataMax"]}
      tick={tick}
      axisLine={axisLine}
      tickLine={false}
      tickFormatter={(v: number) => v.toFixed(1)}
      label={{ value: "epoch", position: "insideBottomRight", offset: -2, fill: palette.faint, fontSize: 10 }}
    />
  );
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div>
        <p className="mb-2 text-sm font-medium text-ink-muted">Loss</p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart {...common}>
            <CartesianGrid vertical={false} stroke={palette.grid} />
            {xAxis}
            <YAxis tick={tick} axisLine={false} tickLine={false} tickFormatter={(v: number) => v.toFixed(2)} domain={["auto", "auto"]} />
            <Tooltip content={<ChartTooltip />} labelFormatter={(v) => `epoch ${Number(v).toFixed(2)}`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="train_loss" name="train loss" stroke={palette.model.model_a} strokeWidth={2} dot={false} isAnimationActive={!reduce} connectNulls />
            <Line type="monotone" dataKey="val_loss" name="val loss" stroke={palette.model.model_a} strokeDasharray="5 4" strokeOpacity={0.75} strokeWidth={2} dot={false} isAnimationActive={!reduce} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div>
        <p className="mb-2 text-sm font-medium text-ink-muted">Validation metrics</p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart {...common}>
            <CartesianGrid vertical={false} stroke={palette.grid} />
            {xAxis}
            <YAxis tick={tick} axisLine={false} tickLine={false} tickFormatter={(v: number) => v.toFixed(2)} domain={["auto", "auto"]} />
            <Tooltip content={<ChartTooltip />} labelFormatter={(v) => `epoch ${Number(v).toFixed(2)}`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="val_macro_f1" name="val Macro-F1" stroke={palette.accent} strokeWidth={2} dot={false} isAnimationActive={!reduce} connectNulls />
            <Line type="monotone" dataKey="val_accuracy" name="val accuracy" stroke={palette.model.model_a} strokeDasharray="5 4" strokeOpacity={0.75} strokeWidth={2} dot={false} isAnimationActive={!reduce} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
