import { Button as ButtonPrimitive } from "@base-ui/react/button";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  [
    "group/button relative inline-flex shrink-0 items-center justify-center gap-2",
    "rounded-xl border border-transparent font-medium text-sm whitespace-nowrap",
    "select-none outline-none transition-all",
    "focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:ring-offset-1 focus-visible:ring-offset-background",
    "disabled:pointer-events-none disabled:opacity-40",
    "active:scale-[0.97]",
    "[&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  ].join(" "),
  {
    variants: {
      variant: {
        /* ── Primary gradient — violet to fuchsia ── */
        default: [
          "bg-gradient-to-r from-violet-600 to-fuchsia-600",
          "text-white shadow-lg shadow-violet-600/25",
          "hover:shadow-violet-500/40 hover:-translate-y-px hover:from-violet-500 hover:to-fuchsia-500",
          "active:translate-y-0 active:shadow-md",
        ].join(" "),

        /* ── Glow — primary with ambient glow ── */
        glow: [
          "bg-primary text-primary-foreground",
          "shadow-[0_0_20px_var(--color-primary,oklch(0.62_0.22_275))/40,0_4px_12px_oklch(0_0_0/0.3)]",
          "hover:shadow-[0_0_32px_var(--color-primary,oklch(0.62_0.22_275))/55,0_6px_16px_oklch(0_0_0/0.35)]",
          "hover:-translate-y-px",
        ].join(" "),

        /* ── Secondary — subtle surface ── */
        secondary: [
          "bg-surface-2 text-foreground border-border",
          "hover:bg-surface-3 hover:border-border-strong",
        ].join(" "),

        /* ── Outline — border only ── */
        outline: [
          "border-border bg-transparent text-foreground",
          "hover:bg-surface-2 hover:border-border-strong",
        ].join(" "),

        /* ── Ghost — transparent ── */
        ghost: [
          "text-foreground-muted",
          "hover:bg-surface-2 hover:text-foreground",
        ].join(" "),

        /* ── Glass — frosted ── */
        glass: [
          "glass text-foreground",
          "hover:bg-[var(--glass-bg-hover)] hover:border-[var(--glass-border-hover)]",
        ].join(" "),

        /* ── Destructive ── */
        destructive: [
          "bg-destructive/10 text-destructive border-destructive/20",
          "hover:bg-destructive/20 hover:border-destructive/40",
          "focus-visible:ring-destructive/50",
        ].join(" "),

        /* ── Link ── */
        link: "text-primary underline-offset-4 hover:underline h-auto px-0 py-0",

        /* ── Success ── */
        success: [
          "bg-success/10 text-success border-success/20",
          "hover:bg-success/20 hover:border-success/40",
        ].join(" "),
      },
      size: {
        xs:      "h-6  px-2.5 text-xs  rounded-lg  gap-1",
        sm:      "h-8  px-3   text-sm  rounded-xl  gap-1.5",
        default: "h-9  px-4   text-sm  rounded-xl  gap-2",
        lg:      "h-10 px-5   text-sm  rounded-xl  gap-2",
        xl:      "h-12 px-6   text-base rounded-2xl gap-2.5",
        icon:    "h-9  w-9   rounded-xl p-0",
        "icon-sm": "h-8 w-8 rounded-xl p-0",
        "icon-xs": "h-6 w-6 rounded-lg p-0",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends ButtonPrimitive.Props,
    VariantProps<typeof buttonVariants> {}

function Button({
  className,
  variant = "default",
  size = "default",
  ...props
}: ButtonProps) {
  return (
    <ButtonPrimitive
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Button, buttonVariants };
