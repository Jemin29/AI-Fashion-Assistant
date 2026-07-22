"use client";

import React, { useState } from "react";
import Link from "next/link";
import { ShieldCheck, ArrowLeft, RefreshCw } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Input, Field } from "@/components/ui/input";

export default function VerifyEmailPage() {
  const { verifyEmail, isLoading } = useAuth();
  const [code, setCode] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!code || code.length !== 6) return;
    await verifyEmail(code);
  };

  return (
    <Card variant="glow" className="w-full max-w-md border border-violet-500/20 shadow-ds-2xl">
      <CardHeader className="space-y-2 text-center">
        <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg shadow-violet-500/25">
          <ShieldCheck className="h-5 w-5 text-white" />
        </div>
        <CardTitle className="text-heading-xl text-foreground">Secure Verification</CardTitle>
        <CardDescription>
          Enter the 6-digit confirmation code dispatched to your email.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Field label="Verification Code" helper="Enter 6 digit numerical code">
            <Input
              placeholder="e.g. 123456"
              maxLength={6}
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
              className="text-center font-mono tracking-widest text-lg font-bold"
              required
            />
          </Field>

          <Button type="submit" variant="default" className="w-full" disabled={isLoading || code.length !== 6}>
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                Unlocking profile...
              </>
            ) : (
              "Confirm Code"
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
