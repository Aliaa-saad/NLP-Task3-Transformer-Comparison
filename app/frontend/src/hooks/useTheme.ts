import { createContext, useContext } from "react";
import type { Label, ModelKey } from "../types/api";

export type Theme = "dark" | "light";

export interface Palette {
  model: Record<ModelKey, string>;
  cls: Record<Label, string>;
  text: string;
  muted: string;
  faint: string;
  grid: string;
  tooltipBg: string;
  tooltipBorder: string;
  violet: string;
}

export const PALETTES: Record<Theme, Palette> = {
  dark: {
    model: { model_a: "#9D8CFF", model_b: "#5CC8FF" },
    cls: { negative: "#FB7185", neutral: "#C4B5FD", positive: "#5EEAD4" },
    text: "#F4F3FF",
    muted: "#A5A3C4",
    faint: "#6E6C8E",
    grid: "rgba(167,139,250,0.12)",
    tooltipBg: "rgba(13,16,36,0.96)",
    tooltipBorder: "rgba(167,139,250,0.3)",
    violet: "#9D8CFF",
  },
  light: {
    model: { model_a: "#6F5FD8", model_b: "#0284C7" },
    cls: { negative: "#E11D48", neutral: "#7C6FB0", positive: "#0F9488" },
    text: "#14163A",
    muted: "#4E4C70",
    faint: "#7A789A",
    grid: "rgba(59,52,120,0.12)",
    tooltipBg: "rgba(255,255,255,0.98)",
    tooltipBorder: "rgba(111,95,216,0.3)",
    violet: "#6F5FD8",
  },
};

export interface ThemeCtx {
  theme: Theme;
  toggle: () => void;
  palette: Palette;
}

export const ThemeContext = createContext<ThemeCtx | null>(null);

export function useTheme(): ThemeCtx {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside ThemeProvider");
  return ctx;
}
