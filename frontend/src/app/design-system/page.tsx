"use client";

import { Header } from "@/components/layout/header";

import {
  Palette, Type, Square, Layers, Sparkles, LayoutGrid,
  MessageSquare, ArrowUp, ArrowDown, Minus, Star,
  ChevronDown, Settings, LogOut, User, Bell, Zap, Code,
  BarChart3, TrendingUp, ShoppingBag, Package, CheckCircle2,
  XCircle, Clock, AlertCircle, Search, Moon, Sun, Copy,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter, StatsCard,
} from "@/components/ui/card";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Input, Textarea, Label, Field } from "@/components/ui/input";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Tabs, TabsList, TabsTrigger, TabsContent, TabsTriggerUnderline,
} from "@/components/ui/tabs";
import {
  Skeleton, Separator, Avatar, AvatarFallback, AvatarImage, Progress, Kbd,
} from "@/components/ui/misc";
import {
  PremiumAreaChart, PremiumBarChart, PremiumDonutChart, CHART_COLORS,
} from "@/components/ui/chart";
import { DialogShowcase } from "./dialog-showcase";

/* ─── Section Wrapper ───────────────────────────────────────────────────────── */
function Section({
  id,
  icon: Icon,
  title,
  description,
  children,
}: {
  id: string;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-20">
      <div className="mb-6">
        <div className="flex items-center gap-2.5 mb-1">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10 border border-primary/20">
            <Icon className="h-3.5 w-3.5 text-primary" />
          </div>
          <h2 className="text-heading-xl text-foreground">{title}</h2>
        </div>
        <p className="text-body-sm text-foreground-muted pl-9">{description}</p>
      </div>
      {children}
      <Separator className="mt-14" gradient />
    </section>
  );
}

/* ─── Sample Data ───────────────────────────────────────────────────────────── */
const trendData = [
  { month: "Jan", streetwear: 4200, luxury: 2800, minimalist: 3100 },
  { month: "Feb", streetwear: 3800, luxury: 3200, minimalist: 2900 },
  { month: "Mar", streetwear: 5100, luxury: 2900, minimalist: 3400 },
  { month: "Apr", streetwear: 4800, luxury: 3600, minimalist: 4200 },
  { month: "May", streetwear: 6200, luxury: 4100, minimalist: 5800 },
  { month: "Jun", streetwear: 5900, luxury: 4400, minimalist: 6100 },
  { month: "Jul", streetwear: 7300, luxury: 5000, minimalist: 7400 },
];

const categoryData = [
  { category: "Streetwear", products: 482, revenue: 128000 },
  { category: "Luxury",     products: 213, revenue: 340000 },
  { category: "Minimalist", products: 367, revenue: 98000 },
  { category: "Techwear",   products: 198, revenue: 76000 },
  { category: "Athleisure", products: 290, revenue: 112000 },
];

const pieData = [
  { name: "Streetwear",  value: 35, color: CHART_COLORS.violet },
  { name: "Luxury",      value: 22, color: CHART_COLORS.fuchsia },
  { name: "Minimalist",  value: 18, color: CHART_COLORS.emerald },
  { name: "Techwear",    value: 13, color: CHART_COLORS.blue },
  { name: "Athleisure",  value: 12, color: CHART_COLORS.amber },
];

const tableData = [
  { brand: "Arc'teryx",   category: "Techwear",   price: "$580",  status: "online",  rating: 4.9 },
  { brand: "Off-White",   category: "Streetwear", price: "$1,200",status: "busy",    rating: 4.7 },
  { brand: "Maison Margiela", category: "Luxury", price: "$2,400",status: "online",  rating: 4.8 },
  { brand: "Stone Island",category: "Streetwear", price: "$890",  status: "away",    rating: 4.6 },
  { brand: "Acne Studios",category: "Minimalist", price: "$650",  status: "online",  rating: 4.7 },
  { brand: "Fear of God", category: "Luxury",     price: "$1,850",status: "offline", rating: 4.5 },
];

/* ─── Color Tokens ──────────────────────────────────────────────────────────── */
const colorTokens = [
  { name: "Primary",      value: "oklch(0.62 0.22 275)",  class: "bg-primary",     textClass: "text-primary-foreground" },
  { name: "Background",   value: "oklch(0.065 0.012 258)",class: "bg-background",  textClass: "text-foreground" },
  { name: "Surface 2",    value: "oklch(0.12 0.013 258)", class: "bg-surface-2",   textClass: "text-foreground" },
  { name: "Success",      value: "oklch(0.72 0.17 148)",  class: "bg-success",     textClass: "text-success-foreground" },
  { name: "Warning",      value: "oklch(0.80 0.16 72)",   class: "bg-warning",     textClass: "text-warning-foreground" },
  { name: "Error",        value: "oklch(0.62 0.22 22)",   class: "bg-destructive", textClass: "text-destructive-foreground" },
  { name: "Info",         value: "oklch(0.68 0.18 234)",  class: "bg-info",        textClass: "text-info-foreground" },
  { name: "Fuchsia Accent",value:"oklch(0.72 0.19 315)",  class: "bg-gradient-to-r from-violet-600 to-fuchsia-600", textClass: "text-white" },
];

/* ─── Page ──────────────────────────────────────────────────────────────────── */
export default function DesignSystemPage() {
  return (
    <>
      <Header title="Design System" description="Enterprise component library v1.0" />

      <div className="px-6 py-10 max-w-5xl space-y-20">

        {/* ── HERO ── */}
        <div className="relative rounded-3xl overflow-hidden border border-border p-10 bg-surface-1 bg-mesh-gradient">
          {/* Ambient glow */}
          <div className="pointer-events-none absolute -top-24 left-1/2 -translate-x-1/2 h-48 w-96 rounded-full bg-violet-600/12 blur-3xl" />
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />

          <div className="relative">
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/25 bg-primary/8 px-3 py-1 text-xs font-medium text-primary mb-5">
              <Sparkles className="h-3 w-3" />
              Premium Enterprise Design System
            </div>
            <h1 className="text-display-lg text-gradient-primary mb-3 max-w-xl">
              Built for the next generation of AI interfaces
            </h1>
            <p className="text-body-lg text-foreground-muted max-w-2xl leading-relaxed">
              A meticulously crafted component library inspired by Apple, Vercel, Linear,
              Midjourney, Figma, and Adobe Firefly. Every token, every component, every
              interaction designed with intention.
            </p>

            <div className="flex items-center gap-3 mt-8 flex-wrap">
              <div className="flex items-center gap-2 text-sm text-foreground-muted">
                <span className="h-1.5 w-1.5 rounded-full bg-success animate-ping" />
                OKLCH Color System
              </div>
              <Separator orientation="vertical" className="h-4" />
              <div className="flex items-center gap-2 text-sm text-foreground-muted">
                <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                Tailwind CSS v4
              </div>
              <Separator orientation="vertical" className="h-4" />
              <div className="flex items-center gap-2 text-sm text-foreground-muted">
                <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                shadcn/ui + Radix UI
              </div>
              <Separator orientation="vertical" className="h-4" />
              <div className="flex items-center gap-2 text-sm text-foreground-muted">
                <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                Framer Motion
              </div>
            </div>
          </div>
        </div>

        {/* ════ 1. COLORS ════ */}
        <Section id="colors" icon={Palette} title="Color System" description="Perceptually uniform OKLCH color palette with semantic naming and dark/light theme support.">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {colorTokens.map((t) => (
              <div key={t.name} className="rounded-xl overflow-hidden border border-border">
                <div className={`h-16 w-full ${t.class}`} />
                <div className="px-3 py-2 bg-surface-2">
                  <p className="text-label-md text-foreground">{t.name}</p>
                  <p className="text-[10px] text-foreground-subtle font-mono mt-0.5 truncate">{t.value}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Gradient showcase */}
          <div className="mt-4 rounded-xl overflow-hidden border border-border h-14 bg-gradient-aurora" />
          <p className="text-xs text-foreground-subtle mt-2 ml-1">Aurora gradient — animated, multi-stop violet→fuchsia</p>
        </Section>

        {/* ════ 2. TYPOGRAPHY ════ */}
        <Section id="typography" icon={Type} title="Typography Scale" description="Inter typeface with a 12-level modular scale from display to overline.">
          <div className="space-y-6 rounded-2xl border border-border bg-surface-2 p-6">
            <div className="space-y-1">
              <span className="text-overline text-foreground-subtle">Display 2XL · 4.5rem · -0.04em</span>
              <p className="text-display-2xl text-foreground leading-none">Fashion AI</p>
            </div>
            <Separator gradient />
            <div className="space-y-1">
              <span className="text-overline text-foreground-subtle">Display LG · 3rem · -0.025em</span>
              <p className="text-display-lg text-foreground">AI-Powered Design Intelligence</p>
            </div>
            <Separator gradient />
            <div className="space-y-1">
              <span className="text-overline text-foreground-subtle">Heading XL · 1.5rem · -0.01em</span>
              <p className="text-heading-xl text-foreground">Style Recommendations</p>
            </div>
            <Separator gradient />
            <div className="space-y-1">
              <span className="text-overline text-foreground-subtle">Body LG · 1rem · relaxed</span>
              <p className="text-body-lg text-foreground-muted max-w-2xl">
                Context-aware fashion recommendations powered by RAG pipelines, ChromaDB
                vector search, and curated fashion knowledge bases.
              </p>
            </div>
            <Separator gradient />
            <div className="flex flex-wrap gap-6">
              <div>
                <span className="text-overline text-foreground-subtle block mb-1.5">Label MD</span>
                <p className="text-label-md text-foreground">Form label text</p>
              </div>
              <div>
                <span className="text-overline text-foreground-subtle block mb-1.5">Caption</span>
                <p className="text-caption">Supplementary detail text</p>
              </div>
              <div>
                <span className="text-overline text-foreground-subtle block mb-1.5">Overline</span>
                <p className="text-overline text-foreground-muted">Section label</p>
              </div>
              <div>
                <span className="text-overline text-foreground-subtle block mb-1.5">Code</span>
                <code className="text-code text-primary px-1.5 py-0.5 bg-primary/8 rounded">
                  const ai = new FashionAssistant()
                </code>
              </div>
            </div>
          </div>
        </Section>

        {/* ════ 3. BUTTONS ════ */}
        <Section id="buttons" icon={Square} title="Buttons" description="7 variants, 7 sizes. Gradient, glow, glass, and semantic states.">
          <div className="space-y-8 rounded-2xl border border-border bg-surface-2 p-6">
            {/* Variants */}
            <div>
              <p className="text-overline text-foreground-subtle mb-4">Variants</p>
              <div className="flex flex-wrap gap-3">
                <Button variant="default">
                  <Sparkles className="h-4 w-4" />
                  Primary Gradient
                </Button>
                <Button variant="glow">
                  <Zap className="h-4 w-4" />
                  Glow Effect
                </Button>
                <Button variant="glass">
                  <Moon className="h-4 w-4" />
                  Glass
                </Button>
                <Button variant="secondary">Secondary</Button>
                <Button variant="outline">Outline</Button>
                <Button variant="ghost">Ghost</Button>
                <Button variant="destructive">Destructive</Button>
                <Button variant="success">Success</Button>
                <Button variant="link">Link →</Button>
              </div>
            </div>

            <Separator gradient />

            {/* Sizes */}
            <div>
              <p className="text-overline text-foreground-subtle mb-4">Sizes</p>
              <div className="flex flex-wrap items-center gap-3">
                <Button size="xs">XSmall</Button>
                <Button size="sm">Small</Button>
                <Button size="default">Default</Button>
                <Button size="lg">Large</Button>
                <Button size="xl">XLarge</Button>
                <Button size="icon" variant="outline">
                  <Settings className="h-4 w-4" />
                </Button>
                <Button size="icon-sm" variant="ghost">
                  <Bell className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <Separator gradient />

            {/* States */}
            <div>
              <p className="text-overline text-foreground-subtle mb-4">States</p>
              <div className="flex flex-wrap gap-3">
                <Button disabled>Disabled</Button>
                <Button variant="outline" disabled>Disabled Outline</Button>
                <Button variant="ghost" disabled>Disabled Ghost</Button>
              </div>
            </div>
          </div>
        </Section>

        {/* ════ 4. CARDS ════ */}
        <Section id="cards" icon={Layers} title="Cards" description="6 card variants for every context — from flat surfaces to glowing focal elements.">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            <StatsCard
              title="Total Revenue"
              value="$1.24M"
              change="12.4%"
              trend="up"
              icon={<BarChart3 className="h-4 w-4 text-white" />}
              iconColor="bg-gradient-to-br from-violet-500 to-fuchsia-500"
            />
            <StatsCard
              title="Active Brands"
              value="4,832"
              change="3.1%"
              trend="down"
              icon={<ShoppingBag className="h-4 w-4 text-white" />}
              iconColor="bg-gradient-to-br from-blue-500 to-violet-500"
            />
            <StatsCard
              title="Trend Index"
              value="94.7"
              change="stable"
              trend="flat"
              icon={<TrendingUp className="h-4 w-4 text-white" />}
              iconColor="bg-gradient-to-br from-emerald-500 to-cyan-500"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Card variant="glass">
              <CardHeader>
                <CardTitle>Glass Card</CardTitle>
                <CardDescription>Frosted glassmorphism surface with backdrop blur and subtle transparency.</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-body-sm text-foreground-muted">Perfect for overlaid content, modals, and sidebar panels.</p>
              </CardContent>
              <CardFooter>
                <Badge variant="primary" dot>Active</Badge>
                <Button variant="ghost" size="sm">View →</Button>
              </CardFooter>
            </Card>

            <Card variant="glow">
              <CardHeader>
                <CardTitle>Glow Card</CardTitle>
                <CardDescription>Violet ambient glow that intensifies on hover for focal elements.</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-body-sm text-foreground-muted">Ideal for featured content, AI outputs, and primary CTAs.</p>
              </CardContent>
              <CardFooter>
                <Badge variant="gradient">Featured</Badge>
                <Button size="sm">Explore</Button>
              </CardFooter>
            </Card>

            <Card variant="interactive" className="col-span-full sm:col-span-1">
              <CardHeader>
                <CardTitle>Interactive Card</CardTitle>
                <CardDescription>Lifts on hover with enhanced shadow. Great for navigable items.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex gap-2 flex-wrap mt-2">
                  <Badge variant="info">RAG Pipeline</Badge>
                  <Badge variant="success">ChromaDB</Badge>
                  <Badge>FastAPI</Badge>
                </div>
              </CardContent>
            </Card>

            <Card variant="flat" className="col-span-full sm:col-span-1">
              <CardHeader>
                <CardTitle>Flat Card</CardTitle>
                <CardDescription>Minimal flat surface for dense information layouts and tables.</CardDescription>
              </CardHeader>
              <CardContent>
                <Progress value={72} color="primary" label="Model Accuracy" showValue />
              </CardContent>
            </Card>
          </div>
        </Section>

        {/* ════ 5. FORM CONTROLS ════ */}
        <Section id="forms" icon={Code} title="Form Controls" description="Input, Textarea, and Label with error, focus, and helper states.">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 rounded-2xl border border-border bg-surface-2 p-6">
            <Field label="Search Styles" helper="Type a style keyword or aesthetic">
              <Input
                placeholder="streetwear, techwear, minimalist…"
                startIcon={<Search className="h-4 w-4" />}
              />
            </Field>

            <Field label="Brand Name" required hint="(public)" helper="Your brand's display name">
              <Input placeholder="e.g. Arc'teryx" />
            </Field>

            <Field label="Email Address" error="Invalid email format">
              <Input
                type="email"
                placeholder="designer@studio.com"
                error
              />
            </Field>

            <Field label="Season" helper="Select a forecast window">
              <select className="flex h-9 w-full rounded-xl px-3 py-2 text-sm bg-surface-2 border border-border text-foreground focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/15 transition-all appearance-none">
                <option value="spring_summer">Spring / Summer</option>
                <option value="autumn_winter">Autumn / Winter</option>
              </select>
            </Field>

            <Field
              label="Brand Description"
              required
              helper="Describe your aesthetic in 2–3 sentences"
              className="sm:col-span-2"
            >
              <Textarea
                rows={3}
                placeholder="Techwear-forward brand blending urban functionality with high-performance materials…"
              />
            </Field>

            <div className="sm:col-span-2 flex gap-3">
              <Button type="submit">
                <Sparkles className="h-4 w-4" />
                Generate Recommendations
              </Button>
              <Button variant="ghost">Clear</Button>
            </div>
          </div>
        </Section>

        {/* ════ 6. BADGES ════ */}
        <Section id="badges" icon={Star} title="Badges & Status" description="13 badge variants for labeling, status, and classification.">
          <div className="space-y-6 rounded-2xl border border-border bg-surface-2 p-6">
            <div>
              <p className="text-overline text-foreground-subtle mb-3">Semantic</p>
              <div className="flex flex-wrap gap-2">
                <Badge variant="primary" dot>Primary</Badge>
                <Badge variant="success" dot>Success</Badge>
                <Badge variant="warning" dot>Warning</Badge>
                <Badge variant="error" dot>Error</Badge>
                <Badge variant="info" dot>Info</Badge>
                <Badge>Default</Badge>
              </div>
            </div>
            <Separator gradient />
            <div>
              <p className="text-overline text-foreground-subtle mb-3">Solid & Gradient</p>
              <div className="flex flex-wrap gap-2">
                <Badge variant="solid-primary">Solid Primary</Badge>
                <Badge variant="solid-success">Live</Badge>
                <Badge variant="solid-warning">In Review</Badge>
                <Badge variant="solid-error">Deprecated</Badge>
                <Badge variant="gradient">AI Generated</Badge>
              </div>
            </div>
            <Separator gradient />
            <div>
              <p className="text-overline text-foreground-subtle mb-3">Status Indicators</p>
              <div className="flex flex-wrap gap-3">
                <StatusBadge status="online"  animated />
                <StatusBadge status="offline" />
                <StatusBadge status="busy"    label="In Review" />
                <StatusBadge status="away"    label="Paused" />
                <StatusBadge status="pending" label="Processing" />
              </div>
            </div>
            <Separator gradient />
            <div>
              <p className="text-overline text-foreground-subtle mb-3">Sizes</p>
              <div className="flex flex-wrap items-center gap-2">
                <Badge size="xs" variant="primary">XS</Badge>
                <Badge size="sm" variant="primary">SM</Badge>
                <Badge size="md" variant="primary">MD</Badge>
                <Badge size="lg" variant="primary">LG</Badge>
              </div>
            </div>
          </div>
        </Section>

        {/* ════ 7. TABLE ════ */}
        <Section id="tables" icon={LayoutGrid} title="Data Table" description="Premium table with striped rows, status badges, and sortable header styling.">
          <Tabs defaultValue="all">
            <TabsList className="mb-4">
              <TabsTrigger value="all">All Brands</TabsTrigger>
              <TabsTrigger value="active">Active</TabsTrigger>
              <TabsTrigger value="luxury">Luxury</TabsTrigger>
            </TabsList>
            <TabsContent value="all">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Brand</TableHead>
                    <TableHead>Category</TableHead>
                    <TableHead>Price Range</TableHead>
                    <TableHead>Rating</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {tableData.map((row) => (
                    <TableRow key={row.brand}>
                      <TableCell className="font-semibold text-foreground">{row.brand}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{row.category}</Badge>
                      </TableCell>
                      <TableCell className="font-mono text-foreground-muted">{row.price}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1.5">
                          <Star className="h-3.5 w-3.5 fill-amber-400 text-amber-400" />
                          <span className="text-sm tabular-nums">{row.rating}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={row.status as "online" | "offline" | "busy" | "away" | "pending"} animated={row.status === "online"} />
                      </TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="sm">View</Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TabsContent>
            <TabsContent value="active">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Brand</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {tableData.filter(r => r.status === "online").map(row => (
                    <TableRow key={row.brand}>
                      <TableCell className="font-semibold">{row.brand}</TableCell>
                      <TableCell><StatusBadge status="online" animated /></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TabsContent>
            <TabsContent value="luxury">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Brand</TableHead>
                    <TableHead>Price</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {tableData.filter(r => r.category === "Luxury").map(row => (
                    <TableRow key={row.brand}>
                      <TableCell className="font-semibold">{row.brand}</TableCell>
                      <TableCell className="font-mono">{row.price}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TabsContent>
          </Tabs>
        </Section>

        {/* ════ 8. CHARTS ════ */}
        <Section id="charts" icon={BarChart3} title="Charts" description="recharts-based chart library with custom dark theme, glassmorphism tooltips, and gradient fills.">
          <div className="space-y-4">
            <PremiumAreaChart
              title="Style Trend Performance"
              description="Monthly search volume by style category (Jan–Jul)"
              data={trendData}
              areas={[
                { key: "streetwear", color: CHART_COLORS.violet, name: "Streetwear" },
                { key: "luxury",     color: CHART_COLORS.fuchsia, name: "Luxury" },
                { key: "minimalist", color: CHART_COLORS.emerald, name: "Minimalist" },
              ]}
              xKey="month"
              height={280}
            />

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <PremiumBarChart
                title="Category Revenue"
                description="Total revenue by style segment"
                data={categoryData}
                bars={[{ key: "revenue", color: CHART_COLORS.violet, name: "Revenue" }]}
                xKey="category"
                height={240}
                valueFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
              />
              <PremiumDonutChart
                title="Market Share"
                description="Style segment distribution"
                data={pieData}
                height={240}
                innerRadius={60}
                centerLabel="100%"
              />
            </div>
          </div>
        </Section>

        {/* ════ 9. ELEVATION / SHADOWS ════ */}
        <Section id="elevation" icon={Layers} title="Elevation & Shadows" description="6-level shadow system simulating realistic material depth.">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            {[
              { name: "xs", shadow: "shadow-ds-xs" },
              { name: "sm", shadow: "shadow-ds-sm" },
              { name: "md", shadow: "shadow-ds-md" },
              { name: "lg", shadow: "shadow-ds-lg" },
              { name: "xl", shadow: "shadow-ds-xl" },
              { name: "2xl",shadow: "shadow-ds-2xl" },
            ].map((s) => (
              <div
                key={s.name}
                className={`rounded-xl bg-surface-2 border border-border p-5 flex items-center justify-between ${s.shadow}`}
              >
                <span className="text-label-md text-foreground">Elevation</span>
                <Badge variant="outline" size="xs">{s.name}</Badge>
              </div>
            ))}
          </div>

          <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { name: "Glow XS", cls: "glow-xs" },
              { name: "Glow SM", cls: "glow-sm" },
              { name: "Glow MD", cls: "glow-md" },
              { name: "Glow LG", cls: "glow-lg" },
            ].map((g) => (
              <div
                key={g.name}
                className={`rounded-xl bg-surface-2 border border-violet-500/20 p-5 flex items-center justify-center ${g.cls}`}
              >
                <span className="text-label-md text-foreground">{g.name}</span>
              </div>
            ))}
          </div>
        </Section>

        {/* ════ 10. GLASSMORPHISM ════ */}
        <Section id="glass" icon={Sparkles} title="Glassmorphism" description="Frosted glass surfaces with backdrop blur for layered, depth-aware UIs.">
          <div
            className="relative rounded-2xl p-8 overflow-hidden border border-border"
            style={{
              background: "linear-gradient(135deg, oklch(0.55 0.24 275 / 0.3), oklch(0.72 0.19 315 / 0.2))",
            }}
          >
            {/* Background orbs */}
            <div className="absolute top-4 left-8  h-32 w-32 rounded-full bg-violet-500/25 blur-2xl" />
            <div className="absolute bottom-4 right-8 h-24 w-24 rounded-full bg-fuchsia-500/20 blur-xl" />

            <div className="relative grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="glass rounded-xl p-4">
                <p className="text-label-md text-foreground mb-1">Glass Default</p>
                <p className="text-body-xs text-foreground-muted">4% white, blur(16px)</p>
                <Badge variant="primary" className="mt-3">Active</Badge>
              </div>
              <div className="glass-hover rounded-xl p-4">
                <p className="text-label-md text-foreground mb-1">Glass Hover</p>
                <p className="text-body-xs text-foreground-muted">Transitions on hover</p>
                <Badge variant="success" className="mt-3">Hover me</Badge>
              </div>
              <div className="glass-heavy rounded-xl p-4">
                <p className="text-label-md text-foreground mb-1">Glass Heavy</p>
                <p className="text-body-xs text-foreground-muted">85% opaque, blur(24px)</p>
                <Badge variant="gradient" className="mt-3">Modal-grade</Badge>
              </div>
            </div>
          </div>
        </Section>

        {/* ════ 11. MISCELLANEOUS ════ */}
        <Section id="misc" icon={Package} title="Misc Components" description="Avatar, Progress, Skeleton, Separator, Kbd — the supporting cast.">
          <div className="space-y-6 rounded-2xl border border-border bg-surface-2 p-6">
            {/* Avatars */}
            <div>
              <p className="text-overline text-foreground-subtle mb-3">Avatars</p>
              <div className="flex items-center gap-3">
                {(["xs", "sm", "md", "lg", "xl"] as const).map((size) => (
                  <Avatar key={size} size={size}>
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
                ))}
              </div>
            </div>

            <Separator gradient />

            {/* Progress */}
            <div>
              <p className="text-overline text-foreground-subtle mb-4">Progress</p>
              <div className="space-y-3 max-w-md">
                <Progress value={92} color="primary" label="Model Confidence" showValue />
                <Progress value={74} color="success" label="Style Match" showValue />
                <Progress value={38} color="warning" label="Data Coverage" showValue />
                <Progress value={15} color="error"   label="Error Rate" showValue />
                <Progress value={61} color="info"    label="Trend Strength" showValue />
              </div>
            </div>

            <Separator gradient />

            {/* Skeleton */}
            <div>
              <p className="text-overline text-foreground-subtle mb-3">Skeletons (shimmer)</p>
              <div className="space-y-2 max-w-sm">
                <Skeleton className="h-5 w-3/4" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-5/6" />
                <Skeleton className="h-4 w-2/3" />
                <div className="flex gap-2 mt-3">
                  <Skeleton className="h-8 w-24 rounded-xl" />
                  <Skeleton className="h-8 w-16 rounded-xl" />
                </div>
              </div>
            </div>

            <Separator gradient />

            {/* Kbd */}
            <div>
              <p className="text-overline text-foreground-subtle mb-3">Keyboard Shortcuts</p>
              <div className="flex flex-wrap gap-4 text-sm text-foreground-muted">
                <span className="flex items-center gap-1.5">
                  <Kbd>⌘</Kbd><Kbd>K</Kbd>
                  <span className="ml-1">Command palette</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <Kbd>⌘</Kbd><Kbd>Enter</Kbd>
                  <span className="ml-1">Submit</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <Kbd>Esc</Kbd>
                  <span className="ml-1">Close</span>
                </span>
              </div>
            </div>
          </div>
        </Section>

        {/* ════ 12. DIALOG ════ */}
        <Section id="dialog" icon={MessageSquare} title="Dialogs" description="Glassmorphism modal with gradient header line, animated open/close, and accessible focus management.">
          <div className="rounded-2xl border border-border bg-surface-2 p-6">
            <p className="text-body-sm text-foreground-muted mb-4">
              Click the button to open a dialog with the full glassmorphism treatment.
            </p>
            <DialogShowcase />
          </div>
        </Section>

        {/* ── Bottom padding ── */}
        <div className="h-8" />
      </div>
    </>
  );
}
