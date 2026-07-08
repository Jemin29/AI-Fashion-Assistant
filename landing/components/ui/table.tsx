import * as React from "react";

export interface ColumnDef<T> {
  header: string;
  accessorKey: keyof T | string;
  cell?: (row: T) => React.ReactNode;
}

export interface TableProps<T> {
  columns: ColumnDef<T>[];
  data: T[];
  emptyText?: string;
  onRowClick?: (row: T) => void;
  className?: string;
}

export function Table<T extends Record<string, any>>({
  columns,
  data,
  emptyText = "No records found.",
  onRowClick,
  className = "",
}: TableProps<T>) {
  return (
    <div className={`w-full overflow-x-auto rounded-2xl border border-white/5 bg-surface-card ${className}`}>
      <table className="w-full border-collapse text-left text-sm text-slate-300">
        <thead>
          <tr className="border-b border-white/5 bg-black/20 text-slate-400 text-xs font-bold uppercase tracking-wider select-none">
            {columns.map((col, idx) => (
              <th key={idx} className="px-6 py-4">
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-white/5">
          {data.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-6 py-12 text-center text-slate-500 font-light">
                {emptyText}
              </td>
            </tr>
          ) : (
            data.map((row, rowIdx) => (
              <tr
                key={rowIdx}
                onClick={() => onRowClick && onRowClick(row)}
                className={`transition-colors duration-150 ${
                  onRowClick ? "hover:bg-white/5 cursor-pointer" : "hover:bg-white/2"
                }`}
              >
                {columns.map((col, colIdx) => {
                  const content = col.cell
                    ? col.cell(row)
                    : row[col.accessorKey as keyof T] !== undefined
                    ? String(row[col.accessorKey as keyof T])
                    : null;
                  return (
                    <td key={colIdx} className="px-6 py-4 font-medium">
                      {content}
                    </td>
                  );
                })}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
