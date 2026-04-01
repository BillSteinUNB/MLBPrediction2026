/**
 * SortableTable — Generic table with clickable column headers for client-side sorting.
 * Supports ascending/descending toggle, null-safe comparisons, and optional row click.
 */
import { useState, useMemo, useCallback, type ReactNode } from 'react';

type SortDirection = 'asc' | 'desc';

export interface Column<T> {
  /** Unique key for this column */
  key: string;
  /** Column header text */
  header: string;
  /** Render function for cell content */
  render: (row: T) => ReactNode;
  /** Extract a sortable value; column is not sortable when omitted */
  sortValue?: (row: T) => number | string | null;
  /** Text alignment (default left) */
  align?: 'left' | 'center' | 'right';
}

interface SortableTableProps<T> {
  /** Column definitions */
  columns: Column<T>[];
  /** Row data */
  data: T[];
  /** Extract a unique key for each row */
  rowKey: (row: T) => string;
  /** Callback when a row is clicked */
  onRowClick?: (row: T) => void;
  /** Additional Tailwind classes on the wrapper */
  className?: string;
}

const ALIGN_CLASS = {
  left: 'text-left',
  center: 'text-center',
  right: 'text-right',
} as const;

function SortIcon({ active, direction }: { active: boolean; direction: SortDirection }) {
  return (
    <svg
      className={`ml-1 inline-block h-3 w-3 transition-colors ${active ? 'text-accent' : 'text-stroke'}`}
      viewBox="0 0 12 12"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      role="img"
      aria-label={`Sort ${direction}`}
    >
      <title>{direction === 'asc' ? 'Sort ascending' : 'Sort descending'}</title>
      {direction === 'asc' ? (
        <path d="M2 8l4-4 4 4" />
      ) : (
        <path d="M2 4l4 4 4-4" />
      )}
    </svg>
  );
}

export function SortableTable<T>({
  columns,
  data,
  rowKey,
  onRowClick,
  className = '',
}: SortableTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDirection>('desc');

  const handleSort = useCallback(
    (key: string) => {
      if (sortKey === key) {
        setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
      } else {
        setSortKey(key);
        setSortDir('desc');
      }
    },
    [sortKey],
  );

  const sorted = useMemo(() => {
    if (!sortKey) return data;
    const col = columns.find((c) => c.key === sortKey);
    if (!col?.sortValue) return data;
    const accessor = col.sortValue;
    return [...data].sort((a, b) => {
      const va = accessor(a);
      const vb = accessor(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      const cmp =
        typeof va === 'number' && typeof vb === 'number'
          ? va - vb
          : String(va).localeCompare(String(vb));
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [data, columns, sortKey, sortDir]);

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-stroke/30">
            {columns.map((col) => {
              const sortable = col.sortValue != null;
              const active = sortKey === col.key;
              const align = ALIGN_CLASS[col.align ?? 'left'];
              return (
                <th
                  key={col.key}
                  className={`px-4 py-3 text-[11px] font-bold uppercase tracking-widest text-ink-dim ${align} ${sortable ? 'cursor-pointer select-none hover:text-accent' : ''}`}
                  onClick={sortable ? () => handleSort(col.key) : undefined}
                >
                  {col.header}
                  {sortable && <SortIcon active={active} direction={active ? sortDir : 'desc'} />}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr
              key={rowKey(row)}
              className={`border-b border-stroke/10 transition-colors hover:bg-well/60 ${onRowClick ? 'cursor-pointer' : ''}`}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={`px-4 py-3 text-ink ${ALIGN_CLASS[col.align ?? 'left']}`}
                >
                  {col.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
