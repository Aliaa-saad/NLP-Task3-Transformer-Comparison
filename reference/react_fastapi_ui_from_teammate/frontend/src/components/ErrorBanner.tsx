import type { ReactNode } from "react";

interface Props {
  title: string;
  children?: ReactNode;
  command?: string;
  tone?: "error" | "warning";
}

export function ErrorBanner({ title, children, command, tone = "error" }: Props) {
  const color = tone === "error" ? "var(--neg)" : "var(--warn)";
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className="glass flex flex-col gap-2 px-4 py-3 sm:px-5"
      style={{ borderColor: `color-mix(in srgb, ${color} 45%, transparent)`, background: `color-mix(in srgb, ${color} 8%, var(--surface))` }}
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 font-mono text-sm font-bold" style={{ color }} aria-hidden="true">
          {tone === "error" ? "✕" : "!"}
        </span>
        <div className="min-w-0 flex-1">
          <p className="font-semibold text-ink">{title}</p>
          {children && <div className="mt-0.5 text-sm text-ink-muted">{children}</div>}
          {command && (
            <pre
              className="mt-2 overflow-x-auto rounded-lg border border-line px-3 py-2 font-mono text-xs text-ink"
              style={{ background: "color-mix(in srgb, var(--bg-0) 60%, transparent)" }}
            >
              <code>{command}</code>
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
