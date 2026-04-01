/**
 * GlassCard — Reusable glass-panel container with optional title, optional icon, children slot.
 * Uses the `.glass-panel` CSS class for backdrop-filter blur effect.
 */
import type { ReactNode } from 'react';

interface GlassCardProps {
  /** Optional heading displayed at the top */
  title?: string;
  /** Optional icon rendered in a rounded accent box */
  icon?: ReactNode;
  /** Card content */
  children: ReactNode;
  /** Additional Tailwind classes */
  className?: string;
}

export function GlassCard({
  title,
  icon,
  children,
  className = '',
}: GlassCardProps) {
  return (
    <div
      className={`glass-panel rounded-2xl border border-stroke/30 p-6 ${className}`}
    >
      {(title || icon) && (
        <div className="mb-5 flex items-center gap-3">
          {icon && (
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent/10">
              <span className="text-accent">{icon}</span>
            </div>
          )}
          {title && (
            <h3 className="font-heading text-lg font-extrabold tracking-tight text-ink">
              {title}
            </h3>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
