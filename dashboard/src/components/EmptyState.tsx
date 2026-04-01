/**
 * EmptyState — "No data available" display with an icon.
 */

interface EmptyStateProps {
  /** Custom message (default: "No data available") */
  message?: string;
  /** Additional Tailwind classes */
  className?: string;
}

export function EmptyState({
  message = 'No data available',
  className = '',
}: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center gap-4 rounded-2xl border border-stroke/20 bg-well/40 px-8 py-16 text-center ${className}`}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-stroke/10">
        <svg
          className="h-6 w-6 text-ink-dim/60"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          role="img"
          aria-label="Empty"
        >
          <title>No data</title>
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M20.25 7.5l-.625 10.632a2.25 2.25 0 0 1-2.247 2.118H6.622a2.25 2.25 0 0 1-2.247-2.118L3.75 7.5m6 4.125 2.25 2.25m0 0 2.25 2.25M12 13.875l2.25-2.25M12 13.875l-2.25 2.25M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125Z"
          />
        </svg>
      </div>
      <p className="text-sm font-medium text-ink-dim">{message}</p>
    </div>
  );
}
