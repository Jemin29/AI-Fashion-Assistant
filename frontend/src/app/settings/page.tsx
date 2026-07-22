"use client";

import React, { useState } from "react";
import { Sliders, Shield, Bell, Moon, Sun, Lock, Save, RefreshCw } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Input, Field } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/misc";
import { Header } from "@/components/layout/header";
import { toast } from "sonner";

export default function SettingsPage() {
  const { user, updateProfile } = useAuth();
  const { theme, setTheme } = useTheme();

  // Settings states
  const [twoFactor, setTwoFactor] = useState(user?.twoFactorEnabled || false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [isSavingPass, setIsSavingPass] = useState(false);

  const handleUpdateSecurity = async (e: React.FormEvent) => {
    e.preventDefault();
    await updateProfile({ twoFactorEnabled: twoFactor });
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentPassword || !newPassword) return;

    if (newPassword.length < 6) {
      toast.error("Security error: New password must be at least 6 characters.");
      return;
    }

    setIsSavingPass(true);
    // Simulate FastAPI update password call latency
    await new Promise((resolve) => setTimeout(resolve, 800));

    toast.success("Security credentials updated successfully.");
    setCurrentPassword("");
    setNewPassword("");
    setIsSavingPass(false);
  };

  return (
    <>
      <Header title="Workspace Settings" description="Configure secure credentials, notification channels, and themes" />

      <div className="px-6 py-8 max-w-3xl space-y-6">
        {/* Workspace Theme & Preferences */}
        <Card variant="glass">
          <CardHeader>
            <CardTitle className="text-sm font-bold flex items-center gap-2">
              <Sliders className="h-4 w-4 text-primary" />
              Workspace Preferences
            </CardTitle>
            <CardDescription>Configure your default theme aesthetics.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-4">
            <div className="flex items-center justify-between text-xs p-2.5 rounded-lg bg-surface-1 border border-border">
              <div>
                <h4 className="font-semibold text-foreground">Workspace Theme</h4>
                <p className="text-[10px] text-foreground-subtle mt-0.5">Toggle light or dark layout defaults.</p>
              </div>
              <div className="flex gap-1.5">
                <Button
                  size="sm"
                  variant={theme === "dark" ? "default" : "outline"}
                  onClick={() => setTheme("dark")}
                  className="gap-1"
                >
                  <Moon className="h-3.5 w-3.5" />
                  Dark
                </Button>
                <Button
                  size="sm"
                  variant={theme === "light" ? "default" : "outline"}
                  onClick={() => setTheme("light")}
                  className="gap-1"
                >
                  <Sun className="h-3.5 w-3.5" />
                  Light
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Security Controls */}
        <Card variant="default">
          <form onSubmit={handleUpdateSecurity}>
            <CardHeader>
              <CardTitle className="text-sm font-bold flex items-center gap-2">
                <Shield className="h-4 w-4 text-violet-400" />
                Security & Two-Factor Authentication
              </CardTitle>
              <CardDescription>Enable multi-factor protection credentials.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-4">
              <div className="flex items-center justify-between text-xs p-2.5 rounded-lg bg-surface-1 border border-border">
                <div>
                  <h4 className="font-semibold text-foreground">Two-Factor Authentication (2FA)</h4>
                  <p className="text-[10px] text-foreground-subtle mt-0.5">Dispatches verification tokens to your email during login.</p>
                </div>
                <button
                  type="button"
                  onClick={() => setTwoFactor(!twoFactor)}
                  className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background ${
                    twoFactor ? "bg-primary" : "bg-surface-3"
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out ${
                      twoFactor ? "translate-x-4" : "translate-x-0"
                    }`}
                  />
                </button>
              </div>
            </CardContent>
            <CardFooter className="border-t border-border pt-4 justify-end">
              <Button type="submit">
                <Save className="h-4 w-4" />
                Save Settings
              </Button>
            </CardFooter>
          </form>
        </Card>

        {/* Password Credentials Change */}
        <Card variant="default">
          <form onSubmit={handleChangePassword}>
            <CardHeader>
              <CardTitle className="text-sm font-bold flex items-center gap-2">
                <Lock className="h-4 w-4 text-fuchsia-400" />
                Configure Password Credentials
              </CardTitle>
              <CardDescription>Update active login password credentials.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Field label="Current Password">
                  <Input
                    type="password"
                    placeholder="••••••••"
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    required
                  />
                </Field>

                <Field label="New Password">
                  <Input
                    type="password"
                    placeholder="••••••••"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    required
                  />
                </Field>
              </div>
            </CardContent>
            <CardFooter className="border-t border-border pt-4 justify-end">
              <Button type="submit" disabled={isSavingPass}>
                {isSavingPass ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Updating Credentials...
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-4 w-4" />
                    Update Password
                  </>
                )}
              </Button>
            </CardFooter>
          </form>
        </Card>
      </div>
    </>
  );
}
