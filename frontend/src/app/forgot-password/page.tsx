"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Mail, ArrowLeft, RefreshCw } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Input, Field } from "@/components/ui/input";

export default function ForgotPasswordPage() {
  const { forgotPassword, isLoading } = useAuth();
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;
    const success = await forgotPassword(email);
    if (success) {
      setSent(true);
    }
  };

  return (
    <Card variant="glow" className="w-full max-w-md border border-violet-500/20 shadow-ds-2xl">
      <CardHeader className="space-y-2 text-center">
        <CardTitle className="text-heading-xl text-foreground">Recover Credentials</CardTitle>
        <CardDescription>
          {sent
            ? "We've sent recovery details to your email."
            : "Enter your work email and we'll dispatch a password reset link."}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {sent ? (
          <div className="space-y-4 text-center py-4">
            <p className="text-xs text-foreground-muted leading-relaxed">
              Check your inbox at <span className="text-foreground font-semibold">{email}</span>. Click the link in the email to configure new password credentials.
            </p>
            <Button
              variant="outline"
              className="w-full"
              onClick={() => setSent(false)}
            >
              Resend Link
            </Button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <Field label="Work Email Address">
              <Input
                type="email"
                placeholder="name@fashionstudio.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                startIcon={<Mail className="h-4 w-4" />}
                required
              />
            </Field>

            <Button type="submit" variant="default" className="w-full" disabled={isLoading}>
              {isLoading ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  Dispatching Link...
                </>
              ) : (
                "Send Reset Link"
              )}
            </Button>
          </form>
        )}
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
