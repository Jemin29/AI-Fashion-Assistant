"use client";
import * as React from "react";
import Link from "next/link";
import {
  Button,
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  Input,
  Textarea,
  Select,
  Dialog,
  Badge,
  Table,
  BarChart,
  LineChart,
} from "@/components/ui";
import { ArrowLeft, Sparkles, Send, HelpCircle, Mail, Key } from "lucide-react";

const DEMO_TABLE_DATA = [
  { id: "T2F-9821", style: "Streetwear", status: "active", resolution: "1024x1024", score: 0.94 },
  { id: "S2D-1294", style: "Couture", status: "new", resolution: "512x512", score: 0.88 },
  { id: "BRD-0043", style: "Minimalist", status: "mock", resolution: "1024x1024", score: 0.92 },
  { id: "T2F-8812", style: "Techwear", status: "active", resolution: "512x512", score: 0.95 },
];

const TABLE_COLUMNS = [
  { header: "Design ID", accessorKey: "id" },
  { header: "Style Preset", accessorKey: "style" },
  {
    header: "Pipeline Status",
    accessorKey: "status",
    cell: (row: any) => (
      <Badge variant={row.status as any}>{row.status}</Badge>
    ),
  },
  { header: "Dimension", accessorKey: "resolution" },
  {
    header: "CLIP Score",
    accessorKey: "score",
    cell: (row: any) => (
      <span className="font-mono text-emerald-400 font-bold">{(row.score * 100).toFixed(0)}%</span>
    ),
  },
];

const BAR_CHART_DATA = [
  { label: "Nike LoRA", value: 1420 },
  { label: "Gucci", value: 980 },
  { label: "Zara", value: 2150 },
  { label: "H&M", value: 1840 },
  { label: "Couture", value: 1100 },
];

const LINE_CHART_DATA = [
  { label: "Mon", value: 450 },
  { label: "Tue", value: 590 },
  { label: "Wed", value: 800 },
  { label: "Thu", value: 810 },
  { label: "Fri", value: 960 },
  { label: "Sat", value: 1400 },
  { label: "Sun", value: 1250 },
];

export default function DesignSystemPreviewPage() {
  const [selectVal, setSelectVal] = React.useState("studio");
  const [dialogOpen, setDialogOpen] = React.useState(false);
  const [inputValue, setInputValue] = React.useState("");

  return (
    <div className="min-h-screen bg-[hsl(225,25%,6%)] text-slate-100 py-16 px-6 relative overflow-hidden">
      {/* Background radial blobs */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-10 left-10 w-96 h-96 rounded-full bg-indigo-600/10 blur-3xl" />
        <div className="absolute bottom-10 right-10 w-96 h-96 rounded-full bg-brand-coral/10 blur-3xl" />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto flex flex-col gap-12">
        {/* Header */}
        <div className="flex flex-col gap-4 border-b border-white/5 pb-8">
          <Link
            href="/"
            className="flex items-center gap-2 text-sm text-indigo-400 hover:text-indigo-300 transition-colors w-fit group"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
            Back to landing page
          </Link>
          <div className="flex items-center gap-3">
            <h1 className="text-4xl md:text-5xl font-black text-white tracking-tight">
              Enterprise <span className="gradient-text">Design System</span>
            </h1>
            <Badge variant="new">v1.0.0</Badge>
          </div>
          <p className="text-lg text-slate-400 font-light">
            A production-ready set of custom UI components styled with Tailwind CSS and Framer Motion.
          </p>
        </div>

        {/* Section: Palette */}
        <div className="flex flex-col gap-6">
          <h2 className="text-2xl font-bold text-white tracking-tight">Colors & Palettes</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { name: "Primary Brand", color: "bg-indigo-600", desc: "hsl(245, 70%, 62%)" },
              { name: "Secondary Coral", color: "bg-brand-coral", desc: "hsl(15, 88%, 65%)" },
              { name: "Teal Highlight", color: "bg-brand-teal", desc: "hsl(175, 65%, 50%)" },
              { name: "Deep Surface", color: "bg-surface-deep", desc: "hsl(225, 25%, 6%)" },
              { name: "Success Green", color: "bg-emerald-600", desc: "Success indicator" },
              { name: "Warning Orange", color: "bg-amber-600", desc: "Warning states" },
              { name: "Error Red", color: "bg-red-600", desc: "Destructive actions" },
              { name: "Surface Card", color: "bg-surface-card", desc: "hsl(225, 22%, 9%)" },
            ].map((c) => (
              <Card key={c.name} className="overflow-hidden border-white/5">
                <div className={`h-24 ${c.color}`} />
                <CardContent className="p-4 flex flex-col gap-1">
                  <div className="text-sm font-bold text-white">{c.name}</div>
                  <div className="text-xs text-slate-500 font-mono">{c.desc}</div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Section: Buttons */}
        <div className="flex flex-col gap-6">
          <h2 className="text-2xl font-bold text-white tracking-tight">Buttons</h2>
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary">Primary Action</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="outline">Outline View</Button>
            <Button variant="glass">Glass Button</Button>
            <Button variant="ghost">Ghost Option</Button>
            <Button variant="success">Success Run</Button>
            <Button variant="destructive">Destructive</Button>
            <Button variant="link">Underlined Link</Button>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" size="xs">Extra Small</Button>
            <Button variant="primary" size="sm">Small</Button>
            <Button variant="primary" size="md">Medium</Button>
            <Button variant="primary" size="lg">Large Scale</Button>
            <Button variant="primary" size="xl" rightIcon={<Sparkles className="w-4 h-4" />}>
              Extra Large
            </Button>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" isLoading>Processing</Button>
            <Button variant="secondary" leftIcon={<Send className="w-4 h-4" />}>Send Message</Button>
            <Button variant="outline" disabled>Disabled State</Button>
          </div>
        </div>

        {/* Section: Badges */}
        <div className="flex flex-col gap-6">
          <h2 className="text-2xl font-bold text-white tracking-tight">Badges</h2>
          <div className="flex flex-wrap gap-2.5">
            <Badge variant="primary">Primary</Badge>
            <Badge variant="secondary">Secondary</Badge>
            <Badge variant="success">Success</Badge>
            <Badge variant="warning">Warning</Badge>
            <Badge variant="error">Error</Badge>
            <Badge variant="info">Info</Badge>
            <Badge variant="outline">Outline</Badge>
            <Badge variant="active">Active System</Badge>
            <Badge variant="mock">Mock Inference</Badge>
            <Badge variant="new">New Release</Badge>
          </div>
        </div>

        {/* Section: Cards & Glassmorphism */}
        <div className="flex flex-col gap-6">
          <h2 className="text-2xl font-bold text-white tracking-tight">Cards & Glassmorphism</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card variant="glass">
              <CardHeader>
                <CardTitle>Glassmorphism Card</CardTitle>
                <CardDescription>A blur-backdrop luxury layout</CardDescription>
              </CardHeader>
              <CardContent>
                Perfect for cards overlaid on dynamic animated gradients or blobs.
              </CardContent>
              <CardFooter className="text-xs text-slate-500">
                Border: 1px border-white/5
              </CardFooter>
            </Card>

            <Card variant="glass-strong">
              <CardHeader>
                <CardTitle>Strong Glass</CardTitle>
                <CardDescription>Increased opacity layout</CardDescription>
              </CardHeader>
              <CardContent>
                Recommended for modals or widgets requiring higher visual separation.
              </CardContent>
              <CardFooter className="text-xs text-slate-500">
                Border: 1px border-white/10
              </CardFooter>
            </Card>

            <Card variant="interactive">
              <CardHeader>
                <CardTitle>Interactive Card</CardTitle>
                <CardDescription>Hover lift and outer glow</CardDescription>
              </CardHeader>
              <CardContent>
                Hover to check transition lifting and border indigo illumination.
              </CardContent>
              <CardFooter className="text-xs text-slate-500">
                Scale transitions activated
              </CardFooter>
            </Card>
          </div>

          <h3 className="text-lg font-bold text-white tracking-tight mt-4">Card State Modifiers</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card variant="glass" isLoading={true}>
              <CardHeader>
                <CardTitle>Loading state</CardTitle>
              </CardHeader>
              <CardContent>This text is blurred behind the loading backdrop overlay.</CardContent>
            </Card>

            <Card variant="glass" isEmpty={true} emptyTitle="Custom Empty Inbox" emptyDescription="No items have been registered in this section." />

            <Card variant="glass" isSuccess={true}>
              <CardHeader>
                <CardTitle>Success state</CardTitle>
                <CardDescription>Operation complete</CardDescription>
              </CardHeader>
              <CardContent>The card border glows emerald and renders a status pill badge.</CardContent>
            </Card>

            <Card variant="glass" isError={true}>
              <CardHeader>
                <CardTitle>Error state</CardTitle>
                <CardDescription>Pipeline warning</CardDescription>
              </CardHeader>
              <CardContent>The card border glows crimson and renders a caution badge.</CardContent>
            </Card>
          </div>
        </div>

        {/* Section: Forms & Inputs */}
        <div className="flex flex-col gap-6">
          <h2 className="text-2xl font-bold text-white tracking-tight">Form Inputs & Controls</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Input
              label="Standard Text"
              placeholder="Type something here..."
              helperText="This is standard input text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
            />
            <Input
              label="Input with Left Icon"
              placeholder="Enter email address"
              leftIcon={<Mail className="w-4 h-4" />}
            />
            <Input
              label="Error State"
              placeholder="Choose a password"
              error="Password must contain at least 8 characters."
              leftIcon={<Key className="w-4 h-4" />}
            />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Select
              label="Select Dropdown"
              options={[
                { value: "studio", label: "Creative Studio Layout" },
                { value: "canvas", label: "Design Canvas Mode" },
                { value: "history", label: "Gallery History View" },
              ]}
              value={selectVal}
              onChange={setSelectVal}
            />
            <Textarea
              label="Multi-line Textarea"
              placeholder="Describe your design vision in full details..."
              rows={3}
            />
          </div>
        </div>

        {/* Section: Dialogs */}
        <div className="flex flex-col gap-6">
          <h2 className="text-2xl font-bold text-white tracking-tight">Dialogs (Modals)</h2>
          <Card className="p-8 flex items-center justify-center border-white/5">
            <div className="text-center flex flex-col items-center gap-4">
              <HelpCircle className="w-12 h-12 text-indigo-400" />
              <div>
                <h3 className="text-lg font-bold text-white">Trigger Dialog Box</h3>
                <p className="text-sm text-slate-500 max-w-sm mt-1">
                  Launch the animated modal popup built with Framer Motion transitions.
                </p>
              </div>
              <Button variant="primary" onClick={() => setDialogOpen(true)}>
                Open Modal
              </Button>
            </div>
          </Card>

          <Dialog
            isOpen={dialogOpen}
            onClose={() => setDialogOpen(false)}
            title="Design Metadata"
            description="Detailed model settings for generation job #T2F-9821."
            footer={
              <>
                <Button variant="outline" onClick={() => setDialogOpen(false)}>
                  Close
                </Button>
                <Button variant="primary" onClick={() => setDialogOpen(false)}>
                  Confirm Settings
                </Button>
              </>
            }
          >
            <div className="space-y-4">
              <p>
                The generation utilizes a combination of **Stable Diffusion XL** base models with fine-tuned **LoRA adapters** for brand styling.
              </p>
              <div className="grid grid-cols-2 gap-4 bg-white/5 rounded-2xl p-4 text-xs font-mono">
                <div>Model: SDXL Base v1.0</div>
                <div>Seed: 88129482</div>
                <div>CFG Scale: 7.5</div>
                <div>Steps: 30</div>
              </div>
            </div>
          </Dialog>
        </div>

        {/* Section: Tables */}
        <div className="flex flex-col gap-6">
          <h2 className="text-2xl font-bold text-white tracking-tight">Tables</h2>
          <Table
            columns={TABLE_COLUMNS}
            data={DEMO_TABLE_DATA}
            onRowClick={(row) => alert(`Clicked row: ${row.id}`)}
          />
        </div>

        {/* Section: Charts */}
        <div className="flex flex-col gap-6">
          <h2 className="text-2xl font-bold text-white tracking-tight">SVG Data Visualizations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Card className="p-6 border-white/5">
              <CardHeader className="px-0 pt-0 pb-4 mb-4">
                <CardTitle>Usage Velocity</CardTitle>
                <CardDescription>Weekly design generations by LoRA model</CardDescription>
              </CardHeader>
              <BarChart data={BAR_CHART_DATA} color="indigo" />
            </Card>

            <Card className="p-6 border-white/5">
              <CardHeader className="px-0 pt-0 pb-4 mb-4">
                <CardTitle>Inference Latency</CardTitle>
                <CardDescription>Daily generation execution speed (ms)</CardDescription>
              </CardHeader>
              <LineChart data={LINE_CHART_DATA} />
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
