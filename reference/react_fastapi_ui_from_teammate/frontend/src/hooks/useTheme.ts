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
  accent: string;
}

export const PALETTES: Record<Theme, Palette> = {
  dark: {
    model: { model_a: "#3DFF8F", model_b: "#2DD4FF" },
    cls: { negative: "#FF4D6D", neutral: "#FFC857", positive: "#3DFF8F" },
    text: "#EDFFF5",
    muted: "#9DB3A8",
    faint: "#5E6F67",
    grid: "rgba(61,255,143,0.10)",
    tooltipBg: "rgba(10,13,12,0.96)",
    tooltipBorder: "rgba(61,255,143,0.3)",
    accent: "#3DFF8F",
  },
  light: {
    model: { model_a: "#0E9F4F", model_b: "#0891B2" },
    cls: { negative: "#E11D48", neutral: "#B7791F", positive: "#0E9F4F" },
    text: "#07170E",
    muted: "#3F574A",
    faint: "#6F8378",
    grid: "rgba(14,122,62,0.12)",
    tooltipBg: "rgba(255,255,255,0.98)",
    tooltipBorder: "rgba(14,122,62,0.3)",
    accent: "#0E9F4F",
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
