/**
 * LoadingState — Skeleton/pulse placeholder while data loads.
 */

interface LoadingStateProps {
  /** Number of skeleton rows to render (default 3) */
  rows?: number;
  /** Additional Tailwind classes */
  className?: string;
}

export function LoadingState({ rows = 3, className = '' }: LoadingStateProps) {
  return (
    <div className={`space-y-4 ${className}`} aria-busy="true">
      {Array.from({ length: rows }, (_, i) => (
        <div key={`skeleton-${String(i)}`} className="animate-pulse space-y-3">
          <div className="h-4 w-1/3 rounded-lg bg-stroke/30" />
          <div className="h-8 w-2/3 rounded-lg bg-stroke/20" />
        </div>
      ))}
      <span className="sr-only">Loading…</span>
    </div>
  );
}
