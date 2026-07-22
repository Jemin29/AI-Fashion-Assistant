"use client";

import { useState } from "react";
import { Sparkles, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogBody,
  DialogFooter,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Input, Field } from "@/components/ui/input";
import { Progress } from "@/components/ui/misc";

export function DialogShowcase() {
  const [open, setOpen] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  return (
    <div className="flex flex-wrap gap-3">
      {/* Default Dialog */}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button>
            <Sparkles className="h-4 w-4" />
            Open Premium Dialog
          </Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <div className="flex items-center gap-2 mb-0.5">
              <Badge variant="gradient" size="xs">New</Badge>
              <Badge variant="primary" dot size="xs">AI Powered</Badge>
            </div>
            <DialogTitle>Generate Style Profile</DialogTitle>
            <DialogDescription>
              Our AI will analyze your preferences and create a personalized style
              profile. This usually takes 2–5 seconds.
            </DialogDescription>
          </DialogHeader>

          <DialogBody className="space-y-4">
            <Field label="Style Aesthetic" helper="What resonates with your brand?">
              <Input placeholder="e.g. minimalist techwear, luxury streetwear…" />
            </Field>
            <Field label="Target Season">
              <select className="flex h-9 w-full rounded-xl px-3 text-sm bg-surface-2 border border-border text-foreground focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/15 transition-all">
                <option>Spring / Summer 2026</option>
                <option>Autumn / Winter 2026</option>
              </select>
            </Field>
            <Progress value={0} color="primary" label="Analyzing…" />
          </DialogBody>

          <DialogFooter>
            <Button variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button onClick={() => setOpen(false)}>
              <Sparkles className="h-4 w-4" />
              Generate Profile
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Confirmation Dialog */}
      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogTrigger asChild>
          <Button variant="destructive">Delete Confirm</Button>
        </DialogTrigger>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-destructive/10 border border-destructive/20 mb-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
            </div>
            <DialogTitle>Delete Style Profile?</DialogTitle>
            <DialogDescription>
              This action cannot be undone. The style profile and all associated
              recommendations will be permanently removed.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirmOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={() => setConfirmOpen(false)}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
