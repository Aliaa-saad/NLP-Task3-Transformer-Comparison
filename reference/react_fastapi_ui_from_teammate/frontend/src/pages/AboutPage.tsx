import { Card, ClassPill, FadeUp, ModelDot, SectionTitle } from "../components/ui";
import { useModels } from "../hooks/useApi";
import { DatasetSetupCard } from "../components/DatasetSetupCard";
import { millions } from "../lib/format";
import type { Label, ModelKey } from "../types/api";

interface Spec {
  key: ModelKey;
  title: string;
  rows: [string, string][];
}

const SPECS: Spec[] = [
  {
    key: "model_a",
    title: "Model A — Scratch Transformer",
    rows: [
      ["Architecture", "Pre-LN encoder-only Transformer: 8 layers, d_model 512, 8 heads, d_ff 2048, GELU, sinusoidal positions, masked mean pooling → Linear(512, 3)"],
      ["Parameters", "40.8M (15.6M embedding)"],
      ["Tokenizer", "bert-base-uncased WordPiece (30,522)"],
      ["Truncation", "Head+tail: first 192 + last 64 tokens (max 256)"],
      ["Training data", "Kaggle Yelp Open Dataset, deduplicated (~6.9M train / 20k val / 40k test, stratified)"],
      ["Training", "3 epochs, AdamW, label smoothing 0.1, bf16, from random init"],
      ["Post-processing", "Additive logit biases tuned on validation for Macro-F1"],
    ],
  },
  {
    key: "model_b",
    title: "Model B — DistilBERT fine-tuned",
    rows: [
      ["Architecture", "distilbert-base-uncased: 6 layers, d_model 768, 12 heads + classification head"],
      ["Parameters", "67.0M"],
      ["Tokenizer", "distilbert-base-uncased WordPiece (30,522)"],
      ["Truncation", "Standard right truncation (max 256)"],
      ["Training data", "Balanced 120k-review Yelp sample (15k val / 15k test)"],
      ["Training", "2 epochs, lr 2e-5, batch 16, AMP"],
      ["Post-processing", "None: plain softmax + argmax"],
    ],
  },
];

export function AboutPage() {
  const models = useModels();
  const info = (k: ModelKey) => models.data?.find((m) => m.model_key === k);
  const labels: { label: Label; stars: string; desc: string }[] = [
    { label: "negative", stars: "1–2 stars", desc: "Clear dissatisfaction" },
    { label: "neutral", stars: "3 stars", desc: "Mixed or lukewarm" },
    { label: "positive", stars: "4–5 stars", desc: "Clear satisfaction" },
  ];

  return (
    <div className="space-y-6">
      <FadeUp>
        <section className="pt-4">
          <p className="eyebrow mb-3">About</p>
          <h1 className="text-4xl font-extrabold tracking-tightish text-ink sm:text-5xl">
            Scratch vs pretrained<span className="neon-text">.</span>
          </h1>
          <p className="mt-3 max-w-3xl text-ink-muted">
            3-class sentiment analysis of Yelp reviews (negative, neutral, positive), comparing two Transformer classifiers: one
            built and trained entirely from scratch, and DistilBERT, pretrained on general text and then fine-tuned. Type any review to
            run both, and see how they compare on a shared held-out set.
          </p>
        </section>
      </FadeUp>

      <FadeUp delay={0.05}>
        <Card>
          <SectionTitle eyebrow="Labels" title="Star rating → sentiment" />
          <div className="grid gap-3 sm:grid-cols-3">
            {labels.map((l, i) => (
              <div key={l.label} className="rounded-xl border border-line p-4">
                <div className="flex items-center justify-between">
                  <ClassPill label={l.label} />
                  <span className="mono-num text-xs text-ink-faint">id {i}</span>
                </div>
                <p className="mt-3 font-semibold text-ink">{l.stars}</p>
                <p className="text-sm text-ink-muted">{l.desc}</p>
              </div>
            ))}
          </div>
        </Card>
      </FadeUp>

      <FadeUp delay={0.08}>
        <DatasetSetupCard />
      </FadeUp>

      <FadeUp delay={0.1} className="grid gap-6 lg:grid-cols-2">
        {SPECS.map((s) => {
          const m = info(s.key);
          return (
            <Card key={s.key} as="article" className="lift">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <h2 className="flex items-center gap-2 text-lg font-bold" style={{ color: s.key === "model_a" ? "var(--model-a)" : "var(--model-b)" }}>
                  <ModelDot modelKey={s.key} /> {s.title}
                </h2>
                {m && (
                  <span className={`rounded-full px-2.5 py-0.5 font-mono text-[11px] ${m.loaded ? "border border-line text-accent" : "border border-line text-neg"}`}>
                    {m.loaded ? `loaded · ${millions(m.parameters)}${m.demo_mode ? " · demo" : ""}` : "not loaded"}
                  </span>
                )}
              </div>
              <dl className="divide-y divide-[color:var(--border)]">
                {s.rows.map(([k, v]) => (
                  <div key={k} className="grid gap-1 py-2.5 sm:grid-cols-[130px_1fr]">
                    <dt className="font-mono text-[11px] uppercase tracking-[0.15em] text-ink-faint">{k}</dt>
                    <dd className="text-sm text-ink-muted">{v}</dd>
                  </div>
                ))}
                {s.key === "model_a" && m?.biases && (
                  <div className="grid gap-1 py-2.5 sm:grid-cols-[130px_1fr]">
                    <dt className="font-mono text-[11px] uppercase tracking-[0.15em] text-ink-faint">Biases</dt>
                    <dd className="mono-num text-sm text-ink-muted">[{m.biases.map((b) => b.toFixed(3)).join(", ")}]</dd>
                  </div>
                )}
              </dl>
            </Card>
          );
        })}
      </FadeUp>
    </div>
  );
}
