import type { Example } from "../lib/examples";

interface Props {
  examples: Example[];
  activeId: string | null;
  onPick: (ex: Example) => void;
  disabled?: boolean;
}

export function ExampleChips({ examples, activeId, onPick, disabled }: Props) {
  return (
    <div>
      <p className="eyebrow mb-2" id="examples-label">
        Try an example
      </p>
      <div className="flex flex-wrap gap-2" role="group" aria-labelledby="examples-label">
        {examples.map((ex) => (
          <button
            key={ex.id}
            type="button"
            className="chip"
            aria-pressed={activeId === ex.id}
            disabled={disabled}
            onClick={() => onPick(ex)}
          >
            {ex.label}
          </button>
        ))}
      </div>
    </div>
  );
}
