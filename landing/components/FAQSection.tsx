"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown } from "lucide-react";

const FAQS = [
  {
    q: "Do I need a GPU to run the AI Fashion Studio?",
    a: "No! The studio runs in Mock Mode on any CPU — you'll get instant placeholder outputs that demonstrate all UI flows, features, and integrations. For real SDXL generation, we recommend an NVIDIA GPU with 8GB+ VRAM (RTX 3080 or above).",
  },
  {
    q: "What is LoRA and how do brand styles work?",
    a: "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adds brand-specific style weights to SDXL. We ship 4 brand adapters: Nike (sport/performance), Gucci (luxury/Italian), Zara (contemporary fast fashion), and H&M (casual/sustainable). Each adapter shifts the visual aesthetic of every generation toward that brand's design language.",
  },
  {
    q: "How does the Sketch-to-Design feature work?",
    a: "Sketch2Design uses ControlNet conditioning — you upload a reference image (edge map, pose skeleton, depth map, or rough sketch) and the model generates a fashion output that follows your structural reference while applying the SDXL aesthetic. It's ideal for turning Procreate or paper sketches into photorealistic fashion renders.",
  },
  {
    q: "What is the RAG Fashion Assistant?",
    a: "RAG (Retrieval-Augmented Generation) means the AI assistant searches a ChromaDB vector store of 556 expert fashion QA pairs before answering. This grounds responses in actual fashion knowledge — fabric properties, brand histories, style guides, trend analysis — rather than just hallucinating answers.",
  },
  {
    q: "Can I integrate the AI Studio with my existing design tools?",
    a: "Yes. The studio exposes a FastAPI REST API alongside the Gradio UI. You can call generation, RAG, recommendation, and evaluation endpoints programmatically. A Postman collection is included. Enterprise customers can request custom integrations with Figma, Slack, or brand management platforms.",
  },
  {
    q: "How are CLIP and FID scores calculated?",
    a: "CLIP score measures the semantic similarity between your text prompt and the generated image using OpenAI's CLIP model — higher is better. FID (Fréchet Inception Distance) measures how close the distribution of generated images is to real fashion photography — lower is better. Both metrics are computed in the Evaluation Dashboard after each generation run.",
  },
  {
    q: "Is my data private?",
    a: "All processing happens locally on your machine. No images or prompts are sent to external servers unless you're using a cloud GPU deployment. Your Gallery, history, and ChromaDB store are saved locally. Enterprise customers can request air-gapped on-premise deployment.",
  },
  {
    q: "How do I get started?",
    a: "Clone the repository, set up your Python environment, and run 'python week6/run.py'. The app opens at http://127.0.0.1:7860 with Mock Mode enabled so it works immediately without any GPU. Check the README for full setup instructions including Docker Compose deployment.",
  },
];

export default function FAQSection() {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  return (
    <section id="faq" className="py-32 relative overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-20" />

      <div className="relative z-10 max-w-3xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold text-indigo-300 border border-indigo-500/30 bg-indigo-500/10 mb-4">
            FAQ
          </span>
          <h2 className="text-4xl md:text-5xl font-black text-white mb-3">
            Everything you{" "}
            <span className="gradient-text">need to know</span>
          </h2>
          <p className="text-lg text-slate-400">
            Can't find your answer?{" "}
            <a href="http://127.0.0.1:7860" target="_blank" rel="noopener noreferrer" className="text-indigo-400 hover:underline">
              Ask the Fashion Assistant
            </a>
            .
          </p>
        </motion.div>

        {/* Accordion */}
        <div className="space-y-3">
          {FAQS.map((faq, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.05 }}
              className={`glass rounded-2xl border transition-all duration-200 overflow-hidden ${
                openIndex === i ? "border-indigo-500/40" : "border-white/5 hover:border-white/10"
              }`}
            >
              <button
                className="w-full flex items-center justify-between gap-4 px-6 py-5 text-left"
                onClick={() => setOpenIndex(openIndex === i ? null : i)}
              >
                <span className="font-semibold text-white text-sm leading-snug">{faq.q}</span>
                <motion.div
                  animate={{ rotate: openIndex === i ? 180 : 0 }}
                  transition={{ duration: 0.2 }}
                  className="flex-shrink-0"
                >
                  <ChevronDown className={`w-5 h-5 ${openIndex === i ? "text-indigo-400" : "text-slate-600"}`} />
                </motion.div>
              </button>

              <AnimatePresence>
                {openIndex === i && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25, ease: "easeInOut" }}
                  >
                    <div className="px-6 pb-5 text-sm text-slate-400 leading-relaxed border-t border-white/5 pt-4">
                      {faq.a}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
