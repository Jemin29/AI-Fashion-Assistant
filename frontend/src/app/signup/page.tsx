"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Sparkles, Mail, Lock, User, ArrowRight, RefreshCw } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Input, Field } from "@/components/ui/input";

export default function SignupPage() {
  const { signup, isLoading } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !email || !password) return;
    await signup(name, email, password);
  };

  return (
    <Card variant="glow" className="w-full max-w-md border border-violet-500/20 shadow-ds-2xl">
      <CardHeader className="space-y-2 text-center">
        <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg shadow-violet-500/25">
          <Sparkles className="h-5 w-5 text-white" />
        </div>
        <CardTitle className="text-heading-xl text-foreground">Request Studio Access</CardTitle>
        <CardDescription>
          Create an enterprise account to access our fashion pipelines.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Field label="Full Name">
            <Input
              placeholder="e.g. Jane Designer"
              value={name}
              onChange={(e) => setName(e.target.value)}
              startIcon={<User className="h-4 w-4" />}
              required
            />
          </Field>

          <Field label="Work Email">
            <Input
              type="email"
              placeholder="name@fashionstudio.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              startIcon={<Mail className="h-4 w-4" />}
              required
            />
          </Field>

          <Field label="Password" helper="Must be at least 6 characters">
            <Input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              startIcon={<Lock className="h-4 w-4" />}
              required
            />
          </Field>

          <Button type="submit" variant="default" className="w-full" disabled={isLoading}>
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                Registering Account...
              </>
            ) : (
              <>
                Create Account
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </Button>
        </form>
      </CardContent>
      <CardFooter className="border-t border-border justify-center">
        <p className="text-xs text-foreground-subtle">
          Already have an account?{" "}
          <Link href="/login" className="text-primary hover:underline font-semibold">
            Login
          </Link>
        </p>
      </CardFooter>
    </Card>
  );
}
