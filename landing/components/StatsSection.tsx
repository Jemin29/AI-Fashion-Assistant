"use client";
import { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";

function CountUp({ end, duration = 2000, suffix = "" }: { end: number; duration?: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView) return;
    let startTime: number | null = null;
    const step = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      setCount(Math.floor(progress * end));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [inView, end, duration]);

  return <span ref={ref}>{count.toLocaleString()}{suffix}</span>;
}

const STATS = [
  {
    value: 50000,
    suffix: "+",
    label: "Designs Generated",
    sub: "Editorial fashion outputs via SDXL",
    color: "from-indigo-500 to-purple-600",
    icon: "🎨",
  },
  {
    value: 556,
    suffix: "",
    label: "Expert QA Pairs",
    sub: "Fashion knowledge base seeded pairs",
    color: "from-orange-500 to-rose-500",
    icon: "📚",
  },
  {
    value: 4,
    suffix: " Brands",
    label: "LoRA Adapters",
    sub: "Nike · Gucci · Zara · H&M fine-tuned",
    color: "from-teal-500 to-cyan-500",
    icon: "🏷️",
  },
  {
    value: 384,
    suffix: "D",
    label: "Vector Embeddings",
    sub: "Dense ChromaDB fashion index",
    color: "from-violet-500 to-indigo-600",
    icon: "🧠",
  },
  {
    value: 12,
    suffix: "+",
    label: "Trend Categories",
    sub: "Real-time velocity tracking",
    color: "from-emerald-500 to-teal-500",
    icon: "📈",
  },
  {
    value: 222,
    suffix: "",
    label: "Tests Passing",
    sub: "Full automated test suite — 100%",
    color: "from-amber-500 to-orange-500",
    icon: "✅",
  },
];

export default function StatsSection() {
  return (
    <section id="stats" className="py-32 relative overflow-hidden scroll-mt-24">
      {/* Radial gradient */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_40%_at_50%_50%,rgba(99,102,241,0.08),transparent)]" />
      <div className="absolute inset-0 grid-bg opacity-40" />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold tracking-widest text-teal-300 border border-teal-500/30 bg-teal-500/10 mb-5">
            BY THE NUMBERS
          </span>
          <h2 className="text-5xl md:text-6xl font-black text-white mb-5 tracking-tight leading-tight">
            Built for <span className="gradient-text">scale and precision</span>
          </h2>
          <p className="text-lg md:text-xl text-slate-400 max-w-xl mx-auto">
            Real metrics from a production AI pipeline — not marketing numbers.
          </p>
        </motion.div>

        {/* Stats grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {STATS.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.5 }}
              whileHover={{ scale: 1.03, y: -4 }}
              className="relative glass rounded-2xl p-7 border border-white/5 overflow-hidden group cursor-default flex flex-col h-full hover:border-white/10 transition-all duration-300"
            >
              {/* Gradient blob */}
              <div className={`absolute -top-8 -right-8 w-32 h-32 rounded-full bg-gradient-to-br ${stat.color} opacity-10 blur-2xl group-hover:opacity-20 transition-opacity`} />

              <div className="relative z-10 flex flex-col h-full">
                <div className="text-4xl mb-4">{stat.icon}</div>
                <div className={`text-4xl md:text-5xl font-black mb-2 bg-gradient-to-r ${stat.color} bg-clip-text text-transparent tracking-tight`}>
                  <CountUp end={stat.value} suffix={stat.suffix} />
                </div>
                <div className="text-base font-bold text-white mb-1">{stat.label}</div>
                <div className="text-sm text-slate-500 mt-auto leading-relaxed">{stat.sub}</div>
              </div>

              {/* Bottom accent line */}
              <div className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r ${stat.color} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
