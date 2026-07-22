import * as React from "react";
import { cn } from "@/lib/utils";

/* ─── Table ─────────────────────────────────────────────────────────────────── */
const Table = React.forwardRef<
  HTMLTableElement,
  React.HTMLAttributes<HTMLTableElement>
>(({ className, ...props }, ref) => (
  <div className="relative w-full overflow-x-auto rounded-xl border border-border">
    <table
      ref={ref}
      className={cn("w-full caption-bottom text-sm border-collapse", className)}
      {...props}
    />
  </div>
));
Table.displayName = "Table";

/* ─── Table Header ──────────────────────────────────────────────────────────── */
const TableHeader = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <thead
    ref={ref}
    className={cn(
      "bg-surface-1 border-b border-border",
      "[&_tr]:border-b-0",
      className
    )}
    {...props}
  />
));
TableHeader.displayName = "TableHeader";

/* ─── Table Body ────────────────────────────────────────────────────────────── */
const TableBody = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tbody
    ref={ref}
    className={cn(
      "bg-surface-2 divide-y divide-border",
      "[&_tr:last-child]:border-0",
      className
    )}
    {...props}
  />
));
TableBody.displayName = "TableBody";

/* ─── Table Footer ──────────────────────────────────────────────────────────── */
const TableFooter = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tfoot
    ref={ref}
    className={cn(
      "bg-surface-1 border-t border-border font-medium",
      "[&>tr]:last:border-b-0",
      className
    )}
    {...props}
  />
));
TableFooter.displayName = "TableFooter";

/* ─── Table Row ─────────────────────────────────────────────────────────────── */
const TableRow = React.forwardRef<
  HTMLTableRowElement,
  React.HTMLAttributes<HTMLTableRowElement>
>(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn(
      "border-b border-border transition-colors duration-100",
      "hover:bg-surface-3/50",
      "data-[state=selected]:bg-primary/5",
      className
    )}
    {...props}
  />
));
TableRow.displayName = "TableRow";

/* ─── Table Head (Column header) ────────────────────────────────────────────── */
const TableHead = React.forwardRef<
  HTMLTableCellElement,
  React.ThHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <th
    ref={ref}
    className={cn(
      "h-10 px-4 text-left align-middle",
      "text-overline text-foreground-muted",
      "whitespace-nowrap",
      "[&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]",
      className
    )}
    {...props}
  />
));
TableHead.displayName = "TableHead";

/* ─── Table Cell ────────────────────────────────────────────────────────────── */
const TableCell = React.forwardRef<
  HTMLTableCellElement,
  React.TdHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <td
    ref={ref}
    className={cn(
      "px-4 py-3 align-middle text-sm text-foreground",
      "[&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]",
      className
    )}
    {...props}
  />
));
TableCell.displayName = "TableCell";

/* ─── Table Caption ─────────────────────────────────────────────────────────── */
const TableCaption = React.forwardRef<
  HTMLTableCaptionElement,
  React.HTMLAttributes<HTMLTableCaptionElement>
>(({ className, ...props }, ref) => (
  <caption
    ref={ref}
    className={cn("mt-4 text-sm text-foreground-muted", className)}
    {...props}
  />
));
TableCaption.displayName = "TableCaption";

export {
  Table,
  TableHeader,
  TableBody,
  TableFooter,
  TableHead,
  TableRow,
  TableCell,
  TableCaption,
};
