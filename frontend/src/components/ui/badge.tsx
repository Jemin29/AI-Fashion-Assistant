import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  [
    "inline-flex items-center gap-1 rounded-full font-medium",
    "transition-colors duration-150",
    "border",
  ].join(" "),
  {
    variants: {
      variant: {
        /* Solid filled */
        default:     "bg-surface-3 text-foreground border-border",
        primary:     "bg-primary/12 text-primary border-primary/25 hover:bg-primary/20",
        success:     "bg-success/12 text-success border-success/25",
        warning:     "bg-warning/12 text-warning border-warning/25",
        error:       "bg-destructive/12 text-destructive border-destructive/25",
        info:        "bg-info/12 text-info border-info/25",

        /* Solid opaque */
        "solid-primary":  "bg-primary text-primary-foreground border-transparent shadow-sm shadow-primary/25",
        "solid-success":  "bg-success/80 text-success-foreground border-transparent",
        "solid-warning":  "bg-warning/80 text-warning-foreground border-transparent",
        "solid-error":    "bg-destructive/80 text-destructive-foreground border-transparent",

        /* Gradient */
        gradient: [
          "border-transparent text-white",
          "bg-gradient-to-r from-violet-600 to-fuchsia-600",
          "shadow-sm shadow-violet-600/25",
        ].join(" "),

        /* Outline only */
        outline:   "bg-transparent text-foreground border-border",
        "outline-primary": "bg-transparent text-primary border-primary/40",

        /* Ghost */
        ghost: "bg-transparent text-foreground-muted border-transparent hover:bg-surface-2",
      },
      size: {
        xs:  "h-4 px-1.5 text-[10px] gap-0.5",
        sm:  "h-5 px-2   text-[11px] gap-0.5",
        md:  "h-6 px-2.5 text-xs     gap-1",
        lg:  "h-7 px-3   text-sm     gap-1",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {
  dot?: boolean;
}

function Badge({ className, variant, size, dot, children, ...props }: BadgeProps) {
  return (
    <span
      data-slot="badge"
      className={cn(badgeVariants({ variant, size, className }))}
      {...props}
    >
      {dot && (
        <span
          className={cn(
            "h-1.5 w-1.5 shrink-0 rounded-full",
            variant === "success" || variant === "solid-success"
              ? "bg-success"
              : variant === "warning" || variant === "solid-warning"
              ? "bg-warning"
              : variant === "error" || variant === "solid-error"
              ? "bg-destructive"
              : variant === "info"
              ? "bg-info"
              : "bg-primary"
          )}
        />
      )}
      {children}
    </span>
  );
}

/* ─── Status Badge (with animated dot) ─────────────────────────────────────── */
type Status = "online" | "offline" | "busy" | "away" | "pending";

interface StatusBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  status: Status;
  label?: string;
  animated?: boolean;
}

const STATUS_MAP: Record<
  Status,
  { color: string; bg: string; border: string; label: string }
> = {
  online:  { color: "bg-success",     bg: "bg-success/10",     border: "border-success/25",     label: "Online" },
  offline: { color: "bg-foreground-subtle", bg: "bg-surface-3", border: "border-border",        label: "Offline" },
  busy:    { color: "bg-destructive",  bg: "bg-destructive/10", border: "border-destructive/25", label: "Busy" },
  away:    { color: "bg-warning",      bg: "bg-warning/10",     border: "border-warning/25",     label: "Away" },
  pending: { color: "bg-info",         bg: "bg-info/10",        border: "border-info/25",        label: "Pending" },
};

function StatusBadge({
  status,
  label,
  animated = false,
  className,
  ...props
}: StatusBadgeProps) {
  const s = STATUS_MAP[status];
  return (
    <span
      data-slot="status-badge"
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
        s.bg,
        s.border,
        className
      )}
      {...props}
    >
      <span className="relative flex h-1.5 w-1.5">
        {animated && status === "online" && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full animate-ping rounded-full opacity-75",
              s.color
            )}
          />
        )}
        <span className={cn("relative inline-flex h-1.5 w-1.5 rounded-full", s.color)} />
      </span>
      {label ?? s.label}
    </span>
  );
}

export { Badge, StatusBadge, badgeVariants };
