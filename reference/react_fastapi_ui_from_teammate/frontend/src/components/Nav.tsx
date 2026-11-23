import { NavLink } from "react-router-dom";
import { useHealth } from "../hooks/useApi";
import { useTheme } from "../hooks/useTheme";

const TABS = [
  { to: "/", label: "Analyze" },
  { to: "/compare", label: "Compare" },
  { to: "/about", label: "About" },
];

/** 2×2 logo mark (same geometry as public/favicon.svg) with a soft glow on the bright squares. */
function Logo() {
  return (
    <svg viewBox="0 0 32 32" width="22" height="22" fill="none" aria-hidden="true" className="shrink-0 overflow-visible">
      <defs>
        <filter id="logo-glow-green" x="-60%" y="-60%" width="220%" height="220%">
          <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="rgba(61,255,143,0.5)" />
        </filter>
        <filter id="logo-glow-cyan" x="-60%" y="-60%" width="220%" height="220%">
          <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="rgba(45,212,255,0.5)" />
        </filter>
      </defs>
      <rect x="3" y="3" width="11" height="11" rx="3" fill="#3DFF8F" filter="url(#logo-glow-green)" />
      <rect x="18" y="3" width="11" height="11" rx="3" fill="#1F5E3D" />
      <rect x="3" y="18" width="11" height="11" rx="3" fill="#1F5E3D" />
      <rect x="18" y="18" width="11" height="11" rx="3" fill="#2DD4FF" filter="url(#logo-glow-cyan)" />
    </svg>
  );
}

function StatusPill() {
  const { data, isError, isPending } = useHealth();
  const online = !!data && !isError;
  const degraded = online && data.status !== "ok";
  const color = isPending ? "var(--faint)" : !online ? "var(--neg)" : degraded ? "var(--warn)" : "var(--accent)";
  const text = isPending ? "Connecting…" : !online ? "API offline" : `API online · ${data.device}${degraded ? " · degraded" : ""}`;
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-2 py-2 font-mono text-[11px] text-ink-muted sm:px-3 sm:py-1"
      role="status"
      aria-live="polite"
      aria-label={text}
      title={degraded ? "One model failed to load: see its card" : undefined}
    >
      <span className="relative flex h-2 w-2">
        {online && <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-50 motion-reduce:hidden" style={{ background: color }} />}
        <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: color }} />
      </span>
      <span className="hidden sm:inline">{text}</span>
    </span>
  );
}

export function Nav() {
  const { theme, toggle } = useTheme();
  return (
    <header className="sticky top-0 z-30 border-b border-line backdrop-blur-xl" style={{ background: "color-mix(in srgb, var(--bg-0) 72%, transparent)" }}>
      <nav className="mx-auto flex max-w-page items-center gap-2 px-4 py-3 sm:gap-3 sm:px-6" aria-label="Main">
        <NavLink to="/" className="flex shrink-0 items-center gap-2.5 rounded-lg" aria-label="Home">
          <Logo />
          <span className="hidden text-sm font-bold tracking-tight text-ink md:inline">Yelp Sentiment · A vs B</span>
        </NavLink>
        <div className="mx-auto flex items-center gap-1 rounded-xl border border-line bg-surface p-1">
          {TABS.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              end
              className={({ isActive }) =>
                `rounded-lg px-2 py-1.5 text-[13px] font-medium transition-colors duration-200 sm:px-3.5 sm:text-sm ${
                  isActive ? "bg-neon-400 font-semibold text-onAccent shadow-neon-sm" : "text-ink-muted hover:text-ink"
                }`
              }
            >
              {t.label}
            </NavLink>
          ))}
        </div>
        <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
          <StatusPill />
          <button
            type="button"
            onClick={toggle}
            className="grid h-8 w-8 place-items-center rounded-lg border border-line bg-surface text-ink-muted transition-colors hover:text-ink"
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? (
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
              </svg>
            )}
          </button>
        </div>
      </nav>
    </header>
  );
}
