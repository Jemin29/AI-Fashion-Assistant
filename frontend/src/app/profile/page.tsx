"use client";

import React, { useState, useEffect } from "react";
import { User, Mail, Globe, Briefcase, ShieldAlert, CheckCircle2, Lock } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Input, Field } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/misc";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/layout/header";

export default function ProfilePage() {
  const { user, updateProfile } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("");
  const [region, setRegion] = useState("");

  // Sync state with authenticated user
  useEffect(() => {
    if (user) {
      setName(user.name);
      setEmail(user.email);
      setRole(user.role);
      setRegion(user.region);
    }
  }, [user]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    await updateProfile({
      name,
      email,
      role,
      region,
      avatar: name.split(" ").map(n => n[0]).join("").toUpperCase().slice(0, 2),
    });
  };

  return (
    <>
      <Header title="User Profile" description="Update your design studio profile settings" />

      <div className="px-6 py-8 max-w-3xl space-y-6">
        <form onSubmit={handleSave} className="space-y-6">
          {/* Profile Overview Card */}
          <Card variant="glass" className="overflow-hidden">
            <CardHeader className="pb-4">
              <div className="flex flex-col sm:flex-row items-center gap-4 text-center sm:text-left">
                <Avatar size="xl">
                  <AvatarFallback>{user?.avatar || "JD"}</AvatarFallback>
                </Avatar>
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2 justify-center sm:justify-start">
                    <CardTitle>{user?.name}</CardTitle>
                    {user?.verified ? (
                      <Badge variant="success" dot size="xs">Verified Profile</Badge>
                    ) : (
                      <Badge variant="warning" dot size="xs">Verification Pending</Badge>
                    )}
                  </div>
                  <CardDescription>{user?.role}</CardDescription>
                  <p className="text-[10px] text-foreground-subtle">{user?.email}</p>
                </div>
              </div>
            </CardHeader>
          </Card>

          {/* Profile Details Card */}
          <Card variant="default">
            <CardHeader>
              <CardTitle>Profile Specifications</CardTitle>
              <CardDescription>Personal details matching workspace logs.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Field label="Full Name">
                  <Input
                    placeholder="Jane Designer"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    startIcon={<User className="h-4 w-4" />}
                    required
                  />
                </Field>

                <Field label="Work Email Address">
                  <Input
                    type="email"
                    placeholder="jane@fashionstudio.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    startIcon={<Mail className="h-4 w-4" />}
                    required
                  />
                </Field>

                <Field label="Studio Role/Title">
                  <Input
                    placeholder="Lead Fashion Architect"
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    startIcon={<Briefcase className="h-4 w-4" />}
                    required
                  />
                </Field>

                <Field label="Target Region/Workspace">
                  <Input
                    placeholder="AP-Northeast (Tokyo)"
                    value={region}
                    onChange={(e) => setRegion(e.target.value)}
                    startIcon={<Globe className="h-4 w-4" />}
                    required
                  />
                </Field>
              </div>
            </CardContent>
            <CardFooter className="border-t border-border pt-4 justify-end gap-3">
              <Button type="submit">
                <CheckCircle2 className="h-4 w-4" />
                Save Changes
              </Button>
            </CardFooter>
          </Card>
        </form>
      </div>
    </>
  );
}
