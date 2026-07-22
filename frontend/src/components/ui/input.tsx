import * as React from "react";
import { cn } from "@/lib/utils";

/* ─── Input ─────────────────────────────────────────────────────────────────── */
export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  startIcon?: React.ReactNode;
  endIcon?: React.ReactNode;
  error?: boolean;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, startIcon, endIcon, error, ...props }, ref) => {
    if (startIcon || endIcon) {
      return (
        <div className="relative flex items-center">
          {startIcon && (
            <span className="pointer-events-none absolute left-3 flex items-center text-foreground-subtle">
              {startIcon}
            </span>
          )}
          <input
            ref={ref}
            type={type}
            data-slot="input"
            className={cn(
              inputBaseClasses,
              startIcon && "pl-9",
              endIcon && "pr-9",
              error && inputErrorClasses,
              className
            )}
            {...props}
          />
          {endIcon && (
            <span className="pointer-events-none absolute right-3 flex items-center text-foreground-subtle">
              {endIcon}
            </span>
          )}
        </div>
      );
    }

    return (
      <input
        ref={ref}
        type={type}
        data-slot="input"
        className={cn(inputBaseClasses, error && inputErrorClasses, className)}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

const inputBaseClasses = [
  "flex h-9 w-full rounded-xl px-3 py-2 text-sm",
  "bg-surface-2 border border-border",
  "text-foreground placeholder:text-foreground-subtle",
  "transition-all duration-150",
  "focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/15 focus:bg-surface-3",
  "hover:border-border-strong",
  "disabled:cursor-not-allowed disabled:opacity-40",
  "file:border-0 file:bg-transparent file:text-sm file:font-medium",
].join(" ");

const inputErrorClasses =
  "border-destructive/50 focus:border-destructive/70 focus:ring-destructive/15";

/* ─── Textarea ──────────────────────────────────────────────────────────────── */
export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  error?: boolean;
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, error, ...props }, ref) => (
    <textarea
      ref={ref}
      data-slot="textarea"
      className={cn(
        "flex min-h-[90px] w-full resize-none rounded-xl px-3 py-2.5 text-sm",
        "bg-surface-2 border border-border",
        "text-foreground placeholder:text-foreground-subtle",
        "transition-all duration-150",
        "focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/15 focus:bg-surface-3",
        "hover:border-border-strong",
        "disabled:cursor-not-allowed disabled:opacity-40",
        "scrollbar-thin",
        error && inputErrorClasses,
        className
      )}
      {...props}
    />
  )
);
Textarea.displayName = "Textarea";

/* ─── Label ─────────────────────────────────────────────────────────────────── */
export interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement> {
  required?: boolean;
  hint?: string;
}

const Label = React.forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, required, hint, children, ...props }, ref) => (
    <label
      ref={ref}
      data-slot="label"
      className={cn(
        "text-label-md text-foreground leading-none cursor-default",
        className
      )}
      {...props}
    >
      {children}
      {required && (
        <span className="ml-1 text-destructive" aria-hidden>
          *
        </span>
      )}
      {hint && (
        <span className="ml-2 text-foreground-subtle font-normal">{hint}</span>
      )}
    </label>
  )
);
Label.displayName = "Label";

/* ─── Form Field (Label + Input + Helper) ───────────────────────────────────── */
interface FieldProps extends React.HTMLAttributes<HTMLDivElement> {
  label?: string;
  required?: boolean;
  hint?: string;
  error?: string;
  helper?: string;
}

function Field({
  label,
  required,
  hint,
  error,
  helper,
  children,
  className,
  ...props
}: FieldProps) {
  return (
    <div data-slot="field" className={cn("flex flex-col gap-1.5", className)} {...props}>
      {label && (
        <Label required={required} hint={hint}>
          {label}
        </Label>
      )}
      {children}
      {error ? (
        <p className="text-xs text-destructive flex items-center gap-1">
          <span aria-hidden>⚠</span> {error}
        </p>
      ) : helper ? (
        <p className="text-xs text-foreground-subtle">{helper}</p>
      ) : null}
    </div>
  );
}

export { Input, Textarea, Label, Field };
