import type { ReactNode } from "react";
import { Card, ClassPill, ModelDot, SectionTitle } from "./ui";
import { Skeleton } from "./Skeleton";
import { useSetup } from "../hooks/useApi";
import { int } from "../lib/format";
import type { SetupModel } from "../types/api";

function Row({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="grid gap-1 py-2.5 sm:grid-cols-[150px_1fr]">
      <dt className="font-mono text-[11px] uppercase tracking-[0.15em] text-ink-faint">{k}</dt>
      <dd className="text-sm text-ink-muted">{children}</dd>
    </div>
  );
}

function splits(m: SetupModel): string {
  const s = m.splits;
  return `${int(s.train)} train / ${int(s.val)} val / ${int(s.test)} test`;
}

/** Dataset + experimental setup. Every value comes from /api/setup (the training artifacts). */
export function DatasetSetupCard() {
  const setup = useSetup();
  if (setup.isPending) {
    return (
      <Card>
        <Skeleton className="h-4 w-40" />
        <div className="mt-4 space-y-3">{[0, 1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-4 w-full" />)}</div>
      </Card>
    );
  }
  if (setup.isError) {
    return (
      <Card>
        <SectionTitle eyebrow="Data" title="Dataset & setup" />
        <p className="text-sm text-ink-muted">Setup details unavailable ({setup.error.message}).</p>
      </Card>
    );
  }
  const { dataset: d, label_mapping, model_a: a, model_b: b, comparison: c } = setup.data;
  return (
    <Card>
      <SectionTitle eyebrow="Data" title="Dataset & setup" right={<span className="text-xs text-ink-faint">read from the training artifacts</span>} />
      <dl className="divide-y divide-[color:var(--border)]">
        <Row k="Dataset">
          <span className="font-semibold text-ink">{d.name}</span>
          {d.file && <span className="ml-1 font-mono text-xs">({d.file})</span>}
          {d.url && (
            <a href={d.url} target="_blank" rel="noreferrer" className="mt-0.5 block break-all text-accent underline-offset-2 hover:underline">
              {d.url}
            </a>
          )}
        </Row>
        <Row k="Reviews">
          <span className="mono-num">{int(d.raw_reviews)}</span> raw, <span className="mono-num">{int(d.duplicates_removed)}</span> exact duplicates
          removed, leaving <span className="mono-num">{int(d.deduped_reviews)}</span>
        </Row>
        <Row k="Label mapping">
          <span className="flex flex-wrap items-center gap-2">
            {label_mapping.map((l) => (
              <span key={l.id} className="flex items-center gap-1">
                <ClassPill label={l.label} size="sm" />
                <span className="font-mono text-xs">{l.stars}★ → {l.id}</span>
              </span>
            ))}
          </span>
        </Row>
        <Row k="Splits">
          <span className="flex items-start gap-2"><ModelDot modelKey="model_a" /><span><span className="mono-num">{splits(a)}</span> ({a.split_strategy})</span></span>
          <span className="mt-1 flex items-start gap-2"><ModelDot modelKey="model_b" /><span><span className="mono-num">{splits(b)}</span> ({b.split_strategy})</span></span>
        </Row>
        <Row k="Random seed">
          <span className="mono-num">{a.seed ?? "—"}</span> (A) · <span className="mono-num">{b.seed ?? "—"}</span> (B)
        </Row>
        <Row k="Tokenizers">
          <span className="block">
            <span className="font-semibold text-modelA">A</span>: <code className="font-mono text-xs text-ink">{a.tokenizer}</code> vocabulary only.
            Model A uses the tokenizer, not pretrained weights; it was trained from random init.
          </span>
          <span className="mt-1 block">
            <span className="font-semibold text-modelB">B</span>: <code className="font-mono text-xs text-ink">{b.tokenizer}</code> tokenizer and
            pretrained weights, fine-tuned.
          </span>
        </Row>
        <Row k="Max length">
          <span className="mono-num">{a.max_length}</span> tokens for both. A keeps head+tail ({a.head_tokens} + {a.tail_tokens}); B truncates from the right.
        </Row>
        <Row k="Hardware">
          <span className="block">Model A training: {a.gpu ?? "not recorded"}</span>
          <span className="block">Model B fine-tuning: {b.gpu ?? "not recorded"}</span>
          <span className="block">Same-machine latency benchmark: {c.gpu ?? c.device ?? "not recorded"}</span>
        </Row>
      </dl>
    </Card>
  );
}
