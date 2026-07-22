"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Lock, ArrowLeft, RefreshCw, CheckCircle } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Input, Field } from "@/components/ui/input";
import { toast } from "sonner";

export default function ResetPasswordPage() {
  const { resetPassword, isLoading } = useAuth();
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!password || !confirmPassword) return;

    if (password !== confirmPassword) {
      toast.error("Validation error: Password confirmation does not match.");
      return;
    }

    await resetPassword(password);
  };

  return (
    <Card variant="glow" className="w-full max-w-md border border-violet-500/20 shadow-ds-2xl">
      <CardHeader className="space-y-2 text-center">
        <CardTitle className="text-heading-xl text-foreground">Configure New Password</CardTitle>
        <CardDescription>
          Establish fresh access credentials for your design profile.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Field label="New Password" helper="Must be at least 6 characters">
            <Input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              startIcon={<Lock className="h-4 w-4" />}
              required
            />
          </Field>

          <Field label="Confirm New Password">
            <Input
              type="password"
              placeholder="••••••••"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              startIcon={<Lock className="h-4 w-4" />}
              required
            />
          </Field>

          <Button type="submit" variant="default" className="w-full" disabled={isLoading}>
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                Configuring Password...
              </>
            ) : (
              "Reset Password"
            )}
          </Button>
        </form>
      </CardContent>
      <CardFooter className="border-t border-border justify-center">
        <Link
          href="/login"
          className="text-xs text-foreground-subtle hover:text-foreground flex items-center gap-1.5 transition-colors"
        >
          <ArrowLeft className="h-3 w-3" />
          Back to login
        </Link>
      </CardFooter>
    </Card>
  );
}
