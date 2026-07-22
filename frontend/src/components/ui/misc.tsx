import * as React from "react";
import { cn } from "@/lib/utils";

/* ─── Separator ─────────────────────────────────────────────────────────────── */
interface SeparatorProps extends React.HTMLAttributes<HTMLDivElement> {
  orientation?: "horizontal" | "vertical";
  gradient?: boolean;
}

function Separator({
  className,
  orientation = "horizontal",
  gradient = false,
  ...props
}: SeparatorProps) {
  return (
    <div
      role="separator"
      data-slot="separator"
      className={cn(
        "shrink-0",
        orientation === "horizontal"
          ? "h-px w-full"
          : "h-full w-px self-stretch",
        gradient
          ? orientation === "horizontal"
            ? "bg-gradient-to-r from-transparent via-border-strong to-transparent"
            : "bg-gradient-to-b from-transparent via-border-strong to-transparent"
          : "bg-border",
        className
      )}
      {...props}
    />
  );
}

/* ─── Skeleton ──────────────────────────────────────────────────────────────── */
interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  shimmer?: boolean;
}

function Skeleton({ className, shimmer = true, ...props }: SkeletonProps) {
  return (
    <div
      data-slot="skeleton"
      className={cn(
        "rounded-xl bg-surface-2 border border-border/50",
        shimmer && [
          "relative overflow-hidden",
          "after:absolute after:inset-0",
          "after:bg-gradient-to-r after:from-transparent after:via-white/[0.04] after:to-transparent",
          "after:animate-shimmer after:bg-[length:200%_100%]",
        ],
        className
      )}
      {...props}
    />
  );
}

/* ─── Avatar ────────────────────────────────────────────────────────────────── */
import * as AvatarPrimitive from "@radix-ui/react-avatar";

const Avatar = React.forwardRef<
  React.ElementRef<typeof AvatarPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Root> & {
    size?: "xs" | "sm" | "md" | "lg" | "xl";
  }
>(({ className, size = "md", ...props }, ref) => {
  const sizes = {
    xs: "h-6 w-6 text-xs",
    sm: "h-8 w-8 text-xs",
    md: "h-9 w-9 text-sm",
    lg: "h-11 w-11 text-base",
    xl: "h-14 w-14 text-lg",
  };
  return (
    <AvatarPrimitive.Root
      ref={ref}
      className={cn(
        "relative flex shrink-0 overflow-hidden rounded-full",
        "ring-2 ring-border",
        sizes[size],
        className
      )}
      {...props}
    />
  );
});
Avatar.displayName = AvatarPrimitive.Root.displayName;

const AvatarImage = React.forwardRef<
  React.ElementRef<typeof AvatarPrimitive.Image>,
  React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Image>
>(({ className, ...props }, ref) => (
  <AvatarPrimitive.Image
    ref={ref}
    className={cn("aspect-square h-full w-full object-cover", className)}
    {...props}
  />
));
AvatarImage.displayName = AvatarPrimitive.Image.displayName;

const AvatarFallback = React.forwardRef<
  React.ElementRef<typeof AvatarPrimitive.Fallback>,
  React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Fallback>
>(({ className, ...props }, ref) => (
  <AvatarPrimitive.Fallback
    ref={ref}
    className={cn(
      "flex h-full w-full items-center justify-center rounded-full",
      "bg-gradient-to-br from-violet-500 to-fuchsia-500",
      "text-white font-semibold",
      className
    )}
    {...props}
  />
));
AvatarFallback.displayName = AvatarPrimitive.Fallback.displayName;

/* ─── Progress ──────────────────────────────────────────────────────────────── */
import * as ProgressPrimitive from "@radix-ui/react-progress";

interface ProgressProps
  extends React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> {
  color?: "primary" | "success" | "warning" | "error" | "info";
  showValue?: boolean;
  label?: string;
}

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  ProgressProps
>(
  (
    { className, value, color = "primary", showValue, label, ...props },
    ref
  ) => {
    const colorMap = {
      primary: "from-violet-600 to-fuchsia-600",
      success: "from-emerald-500 to-emerald-400",
      warning: "from-amber-500 to-amber-400",
      error:   "from-red-600   to-red-500",
      info:    "from-blue-600  to-blue-400",
    };

    return (
      <div className="flex flex-col gap-1.5 w-full">
        {(label || showValue) && (
          <div className="flex items-center justify-between">
            {label && <span className="text-label-sm text-foreground-muted">{label}</span>}
            {showValue && (
              <span className="text-label-sm text-foreground-muted tabular-nums">
                {value ?? 0}%
              </span>
            )}
          </div>
        )}
        <ProgressPrimitive.Root
          ref={ref}
          className={cn(
            "relative h-1.5 w-full overflow-hidden rounded-full",
            "bg-surface-3 border border-border/50",
            className
          )}
          value={value}
          {...props}
        >
          <ProgressPrimitive.Indicator
            className={cn(
              "h-full rounded-full transition-all duration-500 ease-out-expo",
              "bg-gradient-to-r",
              colorMap[color],
              "shadow-sm"
            )}
            style={{ width: `${value ?? 0}%` }}
          />
        </ProgressPrimitive.Root>
      </div>
    );
  }
);
Progress.displayName = ProgressPrimitive.Root.displayName;

/* ─── Kbd ───────────────────────────────────────────────────────────────────── */
function Kbd({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  return (
    <kbd
      className={cn(
        "inline-flex items-center justify-center",
        "rounded-md border border-border bg-surface-2",
        "px-1.5 py-0.5 font-mono text-[11px] text-foreground-muted",
        "shadow-[0_1px_0_1px_var(--border)]",
        className
      )}
      {...props}
    >
      {children}
    </kbd>
  );
}

export {
  Separator,
  Skeleton,
  Avatar,
  AvatarImage,
  AvatarFallback,
  Progress,
  Kbd,
};
