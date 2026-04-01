/**
 * ErrorState — Error display with optional retry button.
 */

interface ErrorStateProps {
  /** Error message to display */
  message?: string;
  /** Callback for the retry button; button hidden when omitted */
  onRetry?: () => void;
}

export function ErrorState({
  message = 'Something went wrong.',
  onRetry,
}: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 rounded-2xl border border-negative/20 bg-negative/5 px-8 py-12 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-negative/10">
        <svg
          className="h-6 w-6 text-negative"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
          role="img"
          aria-label="Error"
        >
          <title>Error</title>
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z"
          />
        </svg>
      </div>
      <p className="text-sm font-medium text-ink-dim">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="rounded-lg border border-stroke/30 bg-panel/40 px-4 py-2 text-xs font-bold uppercase tracking-widest text-ink transition-colors hover:border-accent/40 hover:text-accent"
        >
          Retry
        </button>
      )}
    </div>
  );
}
