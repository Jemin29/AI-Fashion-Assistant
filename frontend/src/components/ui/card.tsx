import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

/* ─── Card Root ─────────────────────────────────────────────────────────────── */
const cardVariants = cva(
  "relative flex flex-col rounded-2xl overflow-hidden transition-all duration-200",
  {
    variants: {
      variant: {
        /* Default elevated card */
        default: [
          "bg-surface-2 border border-border",
          "shadow-ds-sm",
          "hover:border-border-strong hover:shadow-ds-md",
        ].join(" "),

        /* Glassmorphism card */
        glass: [
          "glass",
          "hover:bg-[var(--glass-bg-hover)] hover:border-[var(--glass-border-hover)]",
          "shadow-ds-sm",
        ].join(" "),

        /* Gradient border card */
        gradient: [
          "bg-surface-2 shadow-ds-sm",
          "border border-transparent",
          "before:absolute before:inset-0 before:rounded-2xl before:p-px",
          "before:bg-gradient-to-br before:from-violet-500/40 before:via-transparent before:to-fuchsia-500/30",
          "before:-z-10",
        ].join(" "),

        /* Flat minimal card */
        flat: "bg-surface-1 border border-border",

        /* Interactive card (clickable) */
        interactive: [
          "bg-surface-2 border border-border cursor-pointer",
          "shadow-ds-sm hover:shadow-ds-lg",
          "hover:-translate-y-0.5 hover:border-border-strong",
          "active:translate-y-0 active:shadow-ds-sm",
        ].join(" "),

        /* Glow card */
        glow: [
          "bg-surface-2 border border-violet-500/20",
          "shadow-[0_0_0_1px_oklch(0.62_0.22_275/0.15),0_8px_24px_oklch(0_0_0/0.4)]",
          "hover:border-violet-500/35 hover:shadow-[0_0_0_1px_oklch(0.62_0.22_275/0.25),0_12px_32px_oklch(0_0_0/0.5),0_0_40px_oklch(0.62_0.22_275/0.12)]",
        ].join(" "),
      },
      padding: {
        none: "p-0",
        sm:   "p-4",
        md:   "p-5",
        lg:   "p-6",
        xl:   "p-8",
      },
    },
    defaultVariants: {
      variant: "default",
      padding: "lg",
    },
  }
);

export interface CardProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof cardVariants> {}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant, padding, ...props }, ref) => (
    <div
      ref={ref}
      data-slot="card"
      className={cn(cardVariants({ variant, padding, className }))}
      {...props}
    />
  )
);
Card.displayName = "Card";

/* ─── Card Header ───────────────────────────────────────────────────────────── */
const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    data-slot="card-header"
    className={cn("flex flex-col gap-1.5 pb-4", className)}
    {...props}
  />
));
CardHeader.displayName = "CardHeader";

/* ─── Card Title ────────────────────────────────────────────────────────────── */
const CardTitle = React.forwardRef<
  HTMLHeadingElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    data-slot="card-title"
    className={cn("text-heading-md text-foreground", className)}
    {...props}
  />
));
CardTitle.displayName = "CardTitle";

/* ─── Card Description ──────────────────────────────────────────────────────── */
const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    data-slot="card-description"
    className={cn("text-body-sm text-foreground-muted leading-relaxed", className)}
    {...props}
  />
));
CardDescription.displayName = "CardDescription";

/* ─── Card Content ──────────────────────────────────────────────────────────── */
const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    data-slot="card-content"
    className={cn("flex-1", className)}
    {...props}
  />
));
CardContent.displayName = "CardContent";

/* ─── Card Footer ───────────────────────────────────────────────────────────── */
const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    data-slot="card-footer"
    className={cn("flex items-center justify-between pt-4 border-t border-border mt-4", className)}
    {...props}
  />
));
CardFooter.displayName = "CardFooter";

/* ─── Stats Card ────────────────────────────────────────────────────────────── */
interface StatsCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string;
  value: string | number;
  change?: string;
  trend?: "up" | "down" | "flat";
  icon?: React.ReactNode;
  iconColor?: string;
}

function StatsCard({
  title,
  value,
  change,
  trend = "flat",
  icon,
  iconColor = "bg-gradient-to-br from-violet-500 to-fuchsia-500",
  className,
  ...props
}: StatsCardProps) {
  const trendColor =
    trend === "up"
      ? "text-success"
      : trend === "down"
      ? "text-destructive"
      : "text-foreground-muted";

  return (
    <div
      data-slot="stats-card"
      className={cn(
        "relative flex flex-col gap-3 rounded-2xl p-5",
        "bg-surface-2 border border-border shadow-ds-sm",
        "hover:shadow-ds-md hover:border-border-strong transition-all duration-200",
        className
      )}
      {...props}
    >
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="text-label-md text-foreground-muted">{title}</span>
        {icon && (
          <div
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-xl shadow-lg",
              iconColor
            )}
          >
            {icon}
          </div>
        )}
      </div>

      {/* Value */}
      <div className="flex items-end gap-2">
        <span className="text-display-sm text-foreground tabular-nums">{value}</span>
        {change && (
          <span className={cn("mb-1 text-label-sm", trendColor)}>
            {trend === "up" ? "↑" : trend === "down" ? "↓" : "→"} {change}
          </span>
        )}
      </div>

      {/* Subtle gradient overlay */}
      <div
        className="pointer-events-none absolute inset-0 rounded-2xl opacity-40"
        style={{
          background:
            "radial-gradient(ellipse at 80% 0%, oklch(0.62 0.22 275 / 0.06), transparent 60%)",
        }}
      />
    </div>
  );
}

export {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  StatsCard,
  cardVariants,
};
