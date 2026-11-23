import { Route, Routes } from "react-router-dom";
import { Nav } from "./components/Nav";
import { ErrorBanner } from "./components/ErrorBanner";
import { useHealth } from "./hooks/useApi";
import { isNetworkError } from "./api/client";
import { AnalyzePage } from "./pages/AnalyzePage";
import { ComparePage } from "./pages/ComparePage";
import { AboutPage } from "./pages/AboutPage";

const START_CMD = String.raw`cd app\backend; .\.venv\Scripts\python -m uvicorn app.main:app --port 8000   # or: .\run_dev.ps1`;

function GlobalBanners() {
  const health = useHealth();
  if (health.isError) {
    return (
      <ErrorBanner title="Backend unreachable" command={START_CMD}>
        {isNetworkError(health.error) ? "The FastAPI server is not responding. Start it from the repo root:" : String(health.error)}
      </ErrorBanner>
    );
  }
  const demo = health.data?.models.filter((m) => m.demo_mode) ?? [];
  if (demo.length) {
    return (
      <ErrorBanner tone="warning" title="Demo mode: random weights">
        {demo.map((m) => m.name).join(" and ")} {demo.length > 1 ? "are" : "is"} running with randomly initialised weights
        (DEMO_RANDOM_WEIGHTS=1). Predictions are meaningless. Place the real artifacts in app/backend/artifacts/.
      </ErrorBanner>
    );
  }
  return null;
}

export default function App() {
  return (
    <div className="min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-violet-600 focus:px-3 focus:py-2 focus:text-white">
        Skip to content
      </a>
      <Nav />
      <main id="main" className="mx-auto max-w-page px-4 pb-20 pt-6 sm:px-6">
        <div className="mb-4 empty:hidden">
          <GlobalBanners />
        </div>
        <Routes>
          <Route path="/" element={<AnalyzePage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<AnalyzePage />} />
        </Routes>
      </main>
      <footer className="mx-auto max-w-page px-4 pb-10 font-mono text-[11px] text-ink-faint sm:px-6">
        NLP · Task 03 · Yelp 3-class sentiment · Model A (scratch Transformer) vs Model B (DistilBERT)
      </footer>
    </div>
  );
}
