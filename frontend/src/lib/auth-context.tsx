"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

export interface UserProfile {
  name: string;
  email: string;
  avatar: string;
  role: string;
  region: string;
  verified: boolean;
  twoFactorEnabled: boolean;
}

interface AuthContextType {
  user: UserProfile | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  signup: (name: string, email: string, password: string) => Promise<boolean>;
  logout: () => void;
  forgotPassword: (email: string) => Promise<boolean>;
  resetPassword: (password: string) => Promise<boolean>;
  verifyEmail: (code: string) => Promise<boolean>;
  updateProfile: (updates: Partial<UserProfile>) => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const DEFAULT_USER: UserProfile = {
  name: "Jane Designer",
  email: "jane@fashionstudio.com",
  avatar: "JD",
  role: "Lead Fashion Architect",
  region: "AP-Northeast (Tokyo)",
  verified: true,
  twoFactorEnabled: false,
};

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Synchronize with localStorage
  useEffect(() => {
    const savedUser = localStorage.getItem("fashion_auth_user");
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    } else {
      // Mock starting session with Jane
      setUser(DEFAULT_USER);
      localStorage.setItem("fashion_auth_user", JSON.stringify(DEFAULT_USER));
    }
    setIsLoading(false);
  }, []);

  const login = async (email: string, password: string): Promise<boolean> => {
    setIsLoading(true);
    // Simulate FastAPI auth endpoint call latency
    await new Promise((resolve) => setTimeout(resolve, 800));

    if (password.length < 6) {
      toast.error("Authentication failed: Password must be at least 6 characters.");
      setIsLoading(false);
      return false;
    }

    const mockProfile: UserProfile = {
      ...DEFAULT_USER,
      email,
      name: email.split("@")[0].split(".").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" "),
      avatar: email.charAt(0).toUpperCase() + (email.split("@")[0].split(".")[1]?.charAt(0) || "").toUpperCase(),
    };

    setUser(mockProfile);
    localStorage.setItem("fashion_auth_user", JSON.stringify(mockProfile));
    toast.success("Successfully authenticated. Welcome back!");
    router.push("/dashboard");
    setIsLoading(false);
    return true;
  };

  const signup = async (name: string, email: string, password: string): Promise<boolean> => {
    setIsLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 900));

    if (password.length < 6) {
      toast.error("Registration failed: Password must be at least 6 characters.");
      setIsLoading(false);
      return false;
    }

    const mockProfile: UserProfile = {
      name,
      email,
      avatar: name.split(" ").map(n => n[0]).join("").toUpperCase().slice(0, 2),
      role: "Junior Designer",
      region: "AP-Northeast (Tokyo)",
      verified: false,
      twoFactorEnabled: false,
    };

    setUser(mockProfile);
    localStorage.setItem("fashion_auth_user", JSON.stringify(mockProfile));
    toast.success("Account registered. Verification code dispatched!");
    router.push("/verify-email");
    setIsLoading(false);
    return true;
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem("fashion_auth_user");
    toast.success("Session ended successfully.");
    router.push("/login");
  };

  const forgotPassword = async (email: string): Promise<boolean> => {
    setIsLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 600));
    toast.success(`Reset link dispatched to ${email}`);
    setIsLoading(false);
    return true;
  };

  const resetPassword = async (password: string): Promise<boolean> => {
    setIsLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 800));
    if (password.length < 6) {
      toast.error("Password must be at least 6 characters.");
      setIsLoading(false);
      return false;
    }
    toast.success("Credentials updated successfully. Please login.");
    router.push("/login");
    setIsLoading(false);
    return true;
  };

  const verifyEmail = async (code: string): Promise<boolean> => {
    setIsLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 700));
    if (code.length !== 6) {
      toast.error("Invalid verification code format.");
      setIsLoading(false);
      return false;
    }

    if (user) {
      const updated = { ...user, verified: true };
      setUser(updated);
      localStorage.setItem("fashion_auth_user", JSON.stringify(updated));
    }
    toast.success("Email verified. Access unlocked!");
    router.push("/dashboard");
    setIsLoading(false);
    return true;
  };

  const updateProfile = async (updates: Partial<UserProfile>): Promise<boolean> => {
    if (user) {
      const updated = { ...user, ...updates };
      setUser(updated);
      localStorage.setItem("fashion_auth_user", JSON.stringify(updated));
      toast.success("Profile records updated successfully.");
      return true;
    }
    return false;
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        signup,
        logout,
        forgotPassword,
        resetPassword,
        verifyEmail,
        updateProfile,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used inside an AuthProvider");
  }
  return context;
}
