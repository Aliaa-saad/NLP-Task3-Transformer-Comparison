import type { Config } from "tailwindcss";

// Palette tokens resolve to CSS variables (see src/index.css) so dark/light themes swap cleanly.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: { 0: "var(--bg-0)", 1: "var(--bg-1)", 2: "var(--bg-2)" },
        // primary neon green (fixed hues; use `accent` for the theme-aware variant)
        neon: {
          300: "#8DFFC0",
          400: "#3DFF8F",
          500: "#3DFF8F",
          600: "#1FCB6B",
          800: "#0E3B24",
        },
        cyan: { 400: "#2DD4FF" },
        ink: { DEFAULT: "var(--text)", muted: "var(--muted)", faint: "var(--faint)" },
        accent: "var(--accent)",
        accent2: "var(--accent-2)",
        onAccent: "var(--on-accent)",
        modelA: "var(--model-a)",
        modelB: "var(--model-b)",
        neg: "var(--neg)",
        neu: "var(--neu)",
        pos: "var(--pos)",
        warn: "var(--warn)",
        line: "var(--border)",
        surface: "var(--surface)",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "IBM Plex Mono", "ui-monospace", "Consolas", "monospace"],
      },
      letterSpacing: { tightish: "-0.02em", eyebrow: "0.2em" },
      borderRadius: { card: "18px" },
      boxShadow: { neon: "0 0 24px rgba(61,255,143,0.45)", "neon-sm": "0 0 14px -2px rgba(61,255,143,0.55)" },
      transitionTimingFunction: { smooth: "cubic-bezier(.2,.8,.2,1)" },
      maxWidth: { page: "1200px" },
    },
  },
  plugins: [],
} satisfies Config;
