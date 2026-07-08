"use client";
import { motion } from "framer-motion";
import { Star, Quote } from "lucide-react";

const TESTIMONIALS = [
  {
    name: "Sofia Marchetti",
    role: "Creative Director, Atelier Milano",
    avatar: "👩‍🎨",
    rating: 5,
    quote:
      "I generated 40 concept looks in a single afternoon. The LoRA brand adapters are uncanny — the Gucci aesthetic is immediately recognizable. This is the future of pre-production moodboarding.",
    highlight: "40 concept looks in one afternoon",
  },
  {
    name: "James Park",
    role: "Fashion Tech Founder, StyleOS",
    avatar: "👨‍💻",
    rating: 5,
    quote:
      "The RAG assistant answered every fabric and trend question with cited sources from the knowledge base. As a developer, the API-first design and 100% test coverage gave me immediate confidence.",
    highlight: "100% test coverage",
  },
  {
    name: "Aisha Okonkwo",
    role: "Sustainable Fashion Designer",
    avatar: "👩‍🏫",
    rating: 5,
    quote:
      "Sketch2Design transformed my rough Procreate sketches into fully realized editorial designs. The ControlNet conditioning understood my pose references perfectly. Reduced my design iteration cycle by 3x.",
    highlight: "3x faster design iteration",
  },
  {
    name: "Luca Ferretti",
    role: "Brand Strategist, Luxury Tier",
    avatar: "🧑‍💼",
    rating: 5,
    quote:
      "The Trend Explorer with velocity tracking gave us data-driven evidence for our next collection direction. The CLIP evaluation scores helped us A/B test looks quantitatively — something we've never had before.",
    highlight: "Data-driven trend forecasting",
  },
  {
    name: "Priya Nair",
    role: "Senior Stylist & Content Creator",
    avatar: "👩‍🎤",
    rating: 5,
    quote:
      "From typing 'bohemian golden-hour linen' to a gallery of 4 editorial shots in 15 seconds — the speed is mind-bending. The gallery export to Markdown for my portfolio was a perfect touch.",
    highlight: "4 editorial shots in 15 seconds",
  },
  {
    name: "Marco Silva",
    role: "AI Researcher, Fashion Institute",
    avatar: "🧑‍🔬",
    rating: 5,
    quote:
      "The Evaluation Dashboard with CLIP/FID scoring is exactly what the academic community needs. Production-grade metrics directly integrated into the creative workflow. Exceptional engineering.",
    highlight: "CLIP/FID scoring built in",
  },
];

export default function TestimonialsSection() {
  return (
    <section id="testimonials" className="py-32 relative overflow-hidden scroll-mt-20">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_100%,rgba(99,102,241,0.08),transparent)]" />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold text-yellow-300 border border-yellow-500/30 bg-yellow-500/10 mb-4">
            TESTIMONIALS
          </span>
          <h2 className="text-5xl md:text-6xl font-black text-white mb-4">
            Loved by{" "}
            <span className="gradient-text">designers & developers</span>
          </h2>
          <p className="text-xl text-slate-400 max-w-xl mx-auto">
            From fashion houses to independent creators — real feedback from real users.
          </p>
        </motion.div>

        {/* Masonry-style testimonials */}
        <div className="columns-1 md:columns-2 lg:columns-3 gap-6">
          {TESTIMONIALS.map((t, i) => (
            <motion.div
              key={t.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.5 }}
              whileHover={{ y: -4 }}
              className="break-inside-avoid glass rounded-2xl p-6 border border-white/5 hover:border-indigo-500/30 transition-all duration-300 mb-6"
            >
              {/* Quote icon */}
              <Quote className="w-6 h-6 text-indigo-400/40 mb-4" />

              {/* Stars */}
              <div className="flex gap-1 mb-4">
                {Array.from({ length: t.rating }).map((_, i) => (
                  <Star key={i} className="w-4 h-4 text-amber-400 fill-amber-400" />
                ))}
              </div>

              {/* Highlight badge */}
              <span className="inline-block px-3 py-1 rounded-full bg-indigo-600/20 text-indigo-300 text-xs font-semibold border border-indigo-500/20 mb-3">
                "{t.highlight}"
              </span>

              {/* Quote text */}
              <p className="text-slate-300 text-sm leading-relaxed mb-5">{t.quote}</p>

              {/* Author */}
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full glass border border-white/10 flex items-center justify-center text-xl">
                  {t.avatar}
                </div>
                <div>
                  <div className="text-white font-semibold text-sm">{t.name}</div>
                  <div className="text-slate-500 text-xs">{t.role}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
