"use client";

import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

const Dialog = DialogPrimitive.Root;
const DialogTrigger = DialogPrimitive.Trigger;
const DialogPortal = DialogPrimitive.Portal;
const DialogClose = DialogPrimitive.Close;

/* ─── Overlay ───────────────────────────────────────────────────────────────── */
const DialogOverlay = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn(
      "fixed inset-0 z-50",
      "bg-black/60 backdrop-blur-sm",
      "data-[state=open]:animate-in data-[state=closed]:animate-out",
      "data-[state=open]:fade-in-0 data-[state=closed]:fade-out-0",
      className
    )}
    {...props}
  />
));
DialogOverlay.displayName = DialogPrimitive.Overlay.displayName;

/* ─── Content ───────────────────────────────────────────────────────────────── */
interface DialogContentProps
  extends React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> {
  showClose?: boolean;
}

const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  DialogContentProps
>(({ className, children, showClose = true, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        "fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2",
        "w-full max-w-lg",
        /* Glass surface */
        "glass-heavy rounded-2xl",
        "shadow-2xl shadow-black/60",
        "border border-border",
        /* Ring glow */
        "ring-1 ring-white/5",
        /* Animations */
        "data-[state=open]:animate-in data-[state=closed]:animate-out",
        "data-[state=open]:fade-in-0 data-[state=closed]:fade-out-0",
        "data-[state=open]:zoom-in-95 data-[state=closed]:zoom-out-95",
        "data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%]",
        "data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%]",
        "duration-200",
        className
      )}
      {...props}
    >
      {/* Gradient header line */}
      <div className="absolute top-0 left-6 right-6 h-px bg-gradient-to-r from-transparent via-violet-500/40 to-transparent rounded-full" />

      {children}

      {showClose && (
        <DialogPrimitive.Close
          className={cn(
            "absolute right-4 top-4 rounded-lg p-1",
            "text-foreground-subtle hover:text-foreground",
            "hover:bg-surface-3 transition-all duration-150",
            "focus:outline-none focus:ring-2 focus:ring-ring/50",
            "disabled:pointer-events-none"
          )}
        >
          <X className="h-4 w-4" />
          <span className="sr-only">Close</span>
        </DialogPrimitive.Close>
      )}
    </DialogPrimitive.Content>
  </DialogPortal>
));
DialogContent.displayName = DialogPrimitive.Content.displayName;

/* ─── Header ────────────────────────────────────────────────────────────────── */
const DialogHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    data-slot="dialog-header"
    className={cn("px-6 pt-6 pb-4 flex flex-col gap-1.5", className)}
    {...props}
  />
);

/* ─── Footer ────────────────────────────────────────────────────────────────── */
const DialogFooter = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    data-slot="dialog-footer"
    className={cn(
      "flex flex-col-reverse sm:flex-row sm:justify-end gap-2",
      "px-6 pb-6 pt-4 border-t border-border",
      className
    )}
    {...props}
  />
);

/* ─── Title ─────────────────────────────────────────────────────────────────── */
const DialogTitle = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    ref={ref}
    className={cn("text-heading-lg text-foreground", className)}
    {...props}
  />
));
DialogTitle.displayName = DialogPrimitive.Title.displayName;

/* ─── Description ───────────────────────────────────────────────────────────── */
const DialogDescription = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description
    ref={ref}
    className={cn("text-body-sm text-foreground-muted leading-relaxed", className)}
    {...props}
  />
));
DialogDescription.displayName = DialogPrimitive.Description.displayName;

/* ─── Body ──────────────────────────────────────────────────────────────────── */
const DialogBody = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    data-slot="dialog-body"
    className={cn("px-6 py-2", className)}
    {...props}
  />
);

export {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogBody,
  DialogOverlay,
  DialogPortal,
  DialogTitle,
  DialogTrigger,
};
