export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}

/** Placeholder that matches PredictionCard's footprint to avoid layout shift. */
export function PredictionCardSkeleton() {
  return (
    <div className="glass min-h-[300px] p-5 sm:p-6" role="status" aria-label="Running model">
      <div className="flex items-center justify-between">
        <Skeleton className="h-4 w-48" />
        <Skeleton className="h-7 w-24 rounded-full" />
      </div>
      <Skeleton className="mt-6 h-12 w-32" />
      <div className="mt-6 space-y-4">
        {[0, 1, 2].map((i) => (
          <div key={i} className="flex items-center gap-3">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 flex-1" />
            <Skeleton className="h-3 w-12" />
          </div>
        ))}
      </div>
      <Skeleton className="mt-6 h-3 w-64" />
    </div>
  );
}

export function BlockSkeleton({ height = 260 }: { height?: number }) {
  return (
    <div className="glass p-6" role="status" aria-label="Loading">
      <Skeleton className="h-4 w-40" />
      <Skeleton className="mt-5 w-full" />
      <div className="skeleton mt-5 w-full" style={{ height }} />
    </div>
  );
}
