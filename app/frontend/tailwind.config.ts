import type { Config } from "tailwindcss";

// Palette tokens resolve to CSS variables (see src/index.css) so dark/light themes swap cleanly.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: { 0: "var(--bg-0)", 1: "var(--bg-1)", 2: "var(--bg-2)" },
        violet: {
          300: "#B9AEFF",
          400: "#9D8CFF",
          500: "#8B7CF6",
          600: "#6F5FD8",
          800: "#3B3478",
        },
        ink: { DEFAULT: "var(--text)", muted: "var(--muted)", faint: "var(--faint)" },
        accent: "var(--accent)",
        modelA: "var(--model-a)",
        modelB: "var(--model-b)",
        neg: "var(--neg)",
        neu: "var(--neu)",
        pos: "var(--pos)",
        line: "var(--border)",
        surface: "var(--surface)",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "IBM Plex Mono", "ui-monospace", "Consolas", "monospace"],
      },
      letterSpacing: { tightish: "-0.02em", eyebrow: "0.2em" },
      borderRadius: { card: "18px" },
      transitionTimingFunction: { smooth: "cubic-bezier(.2,.8,.2,1)" },
      maxWidth: { page: "1200px" },
    },
  },
  plugins: [],
} satisfies Config;
