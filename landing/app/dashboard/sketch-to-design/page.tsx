"use client";
import * as React from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  Button,
  Badge,
  Input,
  Select,
  Slider,
  DashboardLayout,
  PageHeader,
  Section,
  ResponsiveGrid,
  CardGrid,
} from "@/components/ui";
import {
  Pencil,
  Upload,
  Undo2,
  Redo2,
  Trash2,
  Maximize2,
  ZoomIn,
  ZoomOut,
  Sliders,
  History,
  Check,
  ChevronRight,
  Sparkles,
  Info
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface SketchRun {
  id: string;
  type: string;
  strength: number;
  bgGradient: string;
  emoji: string;
  date: string;
}

const INITIAL_RUNS: SketchRun[] = [
  { id: "sk-01", type: "Canny Edge", strength: 1.0, bgGradient: "from-purple-950 via-indigo-950 to-black", emoji: "👗", date: "12 mins ago" },
  { id: "sk-02", type: "Human Pose", strength: 0.8, bgGradient: "from-orange-950 via-red-950 to-black", emoji: "🥷", date: "1 hour ago" },
];

export default function SketchToDesignPage() {
  const [controlType, setControlType] = React.useState("canny");
  const [strength, setStrength] = React.useState(1.0);
  const [isGenerating, setIsGenerating] = React.useState(false);
  const [history, setHistory] = React.useState<SketchRun[]>(INITIAL_RUNS);
  const [activeOutput, setActiveOutput] = React.useState<SketchRun | null>(INITIAL_RUNS[0]);
  const [zoom, setZoom] = React.useState(1);
  const [sliderPosition, setSliderPosition] = React.useState(50); // percentage for before/after split
  const [uploadPreview, setUploadPreview] = React.useState<string | null>(null);
  const [isDragging, setIsDragging] = React.useState(false);

  // Drawing Canvas Ref & Undo/Redo Stacks
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = React.useState(false);
  const [brushSize, setBrushSize] = React.useState(4);
  const [undoStack, setUndoStack] = React.useState<string[]>([]);
  const [redoStack, setRedoStack] = React.useState<string[]>([]);

  // Initialize Canvas background
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#111118";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
    }
  }, []);

  // Save canvas state to undo stack
  const saveCanvasState = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const state = canvas.toDataURL();
      setUndoStack((prev) => [...prev, state]);
      setRedoStack([]); // Clear redo stack on new action
    }
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    saveCanvasState();
    setIsDrawing(true);
    draw(e);
  };

  const endDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.beginPath(); // clear active path
    }
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.lineWidth = brushSize;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#ffffff"; // Draw in white sketch edge lines

    const rect = canvas.getBoundingClientRect();
    
    // Get mouse or touch coordinates
    let clientX, clientY;
    if ("touches" in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    const x = clientX - rect.left;
    const y = clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const handleUndo = () => {
    if (undoStack.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Save current state to redo stack
    const currentState = canvas.toDataURL();
    setRedoStack((prev) => [...prev, currentState]);

    // Pop state from undo stack
    const previousState = undoStack[undoStack.length - 1];
    setUndoStack((prev) => prev.slice(0, -1));

    const img = new Image();
    img.src = previousState;
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
    };
  };

  const handleRedo = () => {
    if (redoStack.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Save current state to undo stack
    const currentState = canvas.toDataURL();
    setUndoStack((prev) => [...prev, currentState]);

    // Pop state from redo stack
    const nextState = redoStack[redoStack.length - 1];
    setRedoStack((prev) => prev.slice(0, -1));

    const img = new Image();
    img.src = nextState;
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
    };
  };

  const handleClear = () => {
    saveCanvasState();
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#111118";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
    }
    setUploadPreview(null);
  };

  // Drag and Drop Upload Handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setUploadPreview(event.target.result as string);
          // Draw upload on canvas
          const img = new Image();
          img.src = event.target.result as string;
          img.onload = () => {
            const canvas = canvasRef.current;
            if (canvas) {
              const ctx = canvas.getContext("2d");
              if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
              }
            }
          };
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    await new Promise((r) => setTimeout(r, 2000));

    const gradients = [
      "from-purple-950 via-indigo-950 to-black",
      "from-orange-950 via-red-950 to-black",
      "from-slate-800 via-gray-900 to-black",
      "from-teal-900 via-cyan-900 to-black",
    ];
    const emojis = ["👗", "🏀", "🤍", "🥷"];

    const newRun: SketchRun = {
      id: `sk-${Date.now().toString().slice(-4)}`,
      type: controlType === "canny" ? "Canny Edge" : controlType === "pose" ? "Human Pose" : "Depth Map",
      strength,
      bgGradient: gradients[Math.floor(Math.random() * gradients.length)],
      emoji: emojis[Math.floor(Math.random() * emojis.length)],
      date: "Just now",
    };

    setHistory((prev) => [newRun, ...prev]);
    setActiveOutput(newRun);
    setIsGenerating(false);
  };

  return (
    <DashboardLayout>
      <PageHeader
        title="✏️ Sketch2Design Preprocessing Studio"
        badge="ControlNet Enabled"
        description="Condition generation layouts on drawing edge parameters or pose maps."
      />
      <Section>

        {/* Studio Canvas side-by-side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          
          {/* Left Column: Sketch drawing canvas / Upload */}
          <div className="flex flex-col gap-6">
            
            {/* Sketch Area Card */}
            <Card className="border-white/5 overflow-hidden">
              <CardHeader className="p-4 border-b border-white/5 flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-sm font-bold">Sketch Drawing Canvas</CardTitle>
                  <CardDescription className="text-[10px]">Draw edge contours or drop reference photo</CardDescription>
                </div>
                <div className="flex gap-1">
                  <button
                    onClick={handleUndo}
                    disabled={undoStack.length === 0}
                    className="p-1.5 rounded-lg glass border border-white/5 text-slate-400 hover:text-white disabled:opacity-30 transition-all"
                    title="Undo Action"
                  >
                    <Undo2 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={handleRedo}
                    disabled={redoStack.length === 0}
                    className="p-1.5 rounded-lg glass border border-white/5 text-slate-400 hover:text-white disabled:opacity-30 transition-all"
                    title="Redo Action"
                  >
                    <Redo2 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={handleClear}
                    className="p-1.5 rounded-lg glass border border-white/5 text-red-400 hover:bg-red-500/10 transition-all"
                    title="Clear Canvas"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </CardHeader>
              <CardContent className="p-0 relative bg-black flex items-center justify-center">
                {/* HTML5 Canvas */}
                <canvas
                  ref={canvasRef}
                  width={500}
                  height={500}
                  onMouseDown={startDrawing}
                  onMouseUp={endDrawing}
                  onMouseLeave={endDrawing}
                  onMouseMove={draw}
                  onTouchStart={startDrawing}
                  onTouchEnd={endDrawing}
                  onTouchMove={draw}
                  className="w-full aspect-square max-w-[500px] cursor-crosshair relative z-10"
                />

                {/* Drag and Drop Overlay */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`absolute inset-0 z-20 flex flex-col items-center justify-center p-6 text-center transition-all ${
                    isDragging
                      ? "bg-indigo-600/35 backdrop-blur-sm border-2 border-dashed border-indigo-400 opacity-100"
                      : "bg-transparent opacity-0 hover:opacity-10 pointer-events-none"
                  }`}
                >
                  <Upload className="w-12 h-12 text-white mb-2" />
                  <h3 className="text-white font-bold text-sm">Drop reference template here</h3>
                </div>
              </CardContent>
              <CardFooter className="p-4 bg-black/10 flex flex-col gap-3">
                <Slider
                  label="Brush Size"
                  min={2}
                  max={20}
                  value={brushSize}
                  unit="px"
                  onChange={(e: any) => setBrushSize(parseInt(e.target.value))}
                />
              </CardFooter>
            </Card>

            {/* Conditioning settings */}
            <Card className="border-white/5">
              <CardHeader className="p-4 border-b border-white/5">
                <CardTitle className="text-sm font-bold">ControlNet Configuration</CardTitle>
                <CardDescription className="text-[10px]">Select adapter conditioning properties</CardDescription>
              </CardHeader>
              <CardContent className="p-4 flex flex-col gap-4">
                <Select
                  label="Conditioning Adapter"
                  options={[
                    { value: "canny", label: "Canny Edge Preprocessor" },
                    { value: "pose", label: "Human OpenPose Adapter" },
                    { value: "depth", label: "Depth-Map Estimator" },
                  ]}
                  value={controlType}
                  onChange={setControlType}
                />

                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">
                    Conditioning Strength: {strength.toFixed(1)}
                  </span>
                  <input
                    type="range"
                    min={0.1}
                    max={2.0}
                    step={0.1}
                    value={strength}
                    onChange={(e) => setStrength(parseFloat(e.target.value))}
                    className="accent-indigo-600 cursor-pointer"
                  />
                </div>
              </CardContent>
            </Card>

            <Button variant="primary" size="lg" onClick={handleGenerate} disabled={isGenerating}>
              {isGenerating ? "Processing Preprocessors..." : "Render Sketch to Design"}
            </Button>
          </div>

          {/* Right Column: Preview comparison / before-after split slider */}
          <div className="flex flex-col gap-6">
            
            {/* Before After Split Screen Slider Container */}
            <Card className="border-white/5 relative aspect-square w-full rounded-2xl overflow-hidden bg-black/10 select-none">
              <AnimatePresence mode="wait">
                {isGenerating ? (
                  <motion.div
                    key="generating"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="w-full h-full flex flex-col items-center justify-center p-8 text-center bg-black/30 backdrop-blur-sm"
                  >
                    <motion.div
                      animate={{ scale: [1, 1.15, 1], rotate: [0, 360] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                      className="text-6xl mb-4"
                    >
                      ✏️
                    </motion.div>
                    <h3 className="text-white font-bold mb-2">Rendering Conditioning Maps</h3>
                    <p className="text-slate-500 text-xs">Generating depth matrices and SDXL diffusion layers.</p>
                  </motion.div>
                ) : activeOutput ? (
                  <motion.div
                    key={activeOutput.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="w-full h-full relative"
                    style={{ transform: `scale(${zoom})`, transition: "transform 0.2s" }}
                  >
                    {/* Before Image (Left side Sketch skeleton map) */}
                    <div className="absolute inset-0 bg-[#111118] flex items-center justify-center text-7xl font-mono text-slate-700">
                      <span>✏️ Sketch</span>
                    </div>

                    {/* After Image (Right side generated rendering overlay) */}
                    <div
                      className={`absolute inset-0 bg-gradient-to-br ${activeOutput.bgGradient} flex items-center justify-center text-7xl`}
                      style={{ clipPath: `polygon(${sliderPosition}% 0, 100% 0, 100% 100%, ${sliderPosition}% 100%)` }}
                    >
                      <span>{activeOutput.emoji}</span>
                    </div>

                    {/* Slider separator handle */}
                    <div
                      className="absolute top-0 bottom-0 w-1 bg-indigo-500 cursor-ew-resize z-20 flex items-center justify-center"
                      style={{ left: `${sliderPosition}%` }}
                    >
                      <div className="w-8 h-8 rounded-full bg-indigo-600 border-2 border-indigo-400 flex items-center justify-center shadow-lg -translate-x-1/2">
                        <span className="text-[10px] font-bold text-white">↔</span>
                      </div>
                    </div>

                    {/* Range input overlay for easy sliding */}
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={sliderPosition}
                      onChange={(e) => setSliderPosition(parseInt(e.target.value))}
                      className="absolute inset-0 opacity-0 cursor-ew-resize z-30 w-full h-full"
                    />

                    {/* Zoom & Maximize buttons at top right */}
                    <div className="absolute top-4 right-4 flex gap-2 z-40">
                      <button
                        onClick={() => setZoom(z => Math.max(0.5, z - 0.1))}
                        className="p-1.5 glass rounded-lg border border-white/10 text-slate-300 hover:text-white"
                      >
                        <ZoomOut className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => setZoom(z => Math.min(2, z + 0.1))}
                        className="p-1.5 glass rounded-lg border border-white/10 text-slate-300 hover:text-white"
                      >
                        <ZoomIn className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    {/* Info badge left bottom */}
                    <div className="absolute bottom-4 left-4 glass rounded-xl px-3 py-1.5 border border-white/5 flex items-center gap-1.5 text-[10px] text-white z-40">
                      <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                      Drag range split to compare original sketch template vs final design
                    </div>

                  </motion.div>
                ) : (
                  <div className="text-center p-8 text-slate-500">
                    <Info className="w-12 h-12 text-slate-700 mx-auto mb-3" />
                    <p className="text-sm font-semibold">Render a sketch to display before/after slider.</p>
                  </div>
                )}
              </AnimatePresence>
            </Card>

            {/* Session Preprocessing Runs */}
            <div className="flex flex-col gap-3">
              <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5 px-1">
                <History className="w-3.5 h-3.5" /> Preprocessing Run History
              </h4>

              <div className="flex flex-col gap-2">
                {history.map((run) => (
                  <button
                    key={run.id}
                    onClick={() => {
                      setActiveOutput(run);
                      setSliderPosition(50);
                    }}
                    className={`w-full flex items-center gap-4 p-3 rounded-xl border text-left transition-all ${
                      activeOutput?.id === run.id
                        ? "border-indigo-500/40 bg-indigo-500/5"
                        : "border-white/5 hover:border-white/10 hover:bg-white/2"
                    }`}
                  >
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${run.bgGradient} flex items-center justify-center text-xl flex-shrink-0`}>
                      {run.emoji}
                    </div>
                    <div className="flex-1 min-w-0 flex flex-col gap-0.5">
                      <div className="text-white font-bold text-xs">Run: {run.id}</div>
                      <div className="flex items-center gap-2 text-[10px] text-slate-500 font-medium">
                        <span>Type: {run.type}</span>
                        <span>·</span>
                        <span>Strength: {run.strength}</span>
                      </div>
                    </div>
                    <ChevronRight className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
                  </button>
                ))}
              </div>
            </div>

          </div>

        </div>

      </Section>
    </DashboardLayout>
  );
}
