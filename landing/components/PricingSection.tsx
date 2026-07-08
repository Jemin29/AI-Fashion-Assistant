"use client";
import { motion } from "framer-motion";
import { Check, Zap, Star, Building2, ArrowRight } from "lucide-react";

const PLANS = [
  {
    name: "Explorer",
    icon: Zap,
    price: "Free",
    sub: "Forever",
    description: "Perfect for trying out AI fashion generation locally.",
    color: "from-slate-600 to-slate-700",
    border: "border-white/8",
    badge: null,
    features: [
      "Text-to-Fashion (SDXL mock mode)",
      "Fashion Q&A Assistant",
      "Trend Explorer (12 categories)",
      "Basic Gallery (50 designs)",
      "JSON export",
      "Community support",
    ],
    cta: "Start Free",
    ctaStyle: "glass border border-white/15 text-white hover:border-indigo-500/50",
    href: "http://127.0.0.1:7860",
  },
  {
    name: "Studio",
    icon: Star,
    price: "$29",
    sub: "per month",
    description: "For professional designers and creators at scale.",
    color: "from-indigo-600 to-purple-600",
    border: "border-indigo-500/40",
    badge: "MOST POPULAR",
    features: [
      "Everything in Explorer, plus:",
      "Full SDXL GPU generation",
      "ControlNet Sketch2Design",
      "4 Brand LoRA adapters",
      "RAG Assistant (ChromaDB indexed)",
      "Unlimited Gallery + exports",
      "CLIP / FID evaluation dashboard",
      "Batch generation (1–4 images)",
      "Priority support",
    ],
    cta: "Start Studio",
    ctaStyle: "bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-500 hover:to-purple-500 hover:shadow-[0_0_30px_rgba(99,102,241,0.5)]",
    href: "http://127.0.0.1:7860",
  },
  {
    name: "Enterprise",
    icon: Building2,
    price: "Custom",
    sub: "Contact us",
    description: "For fashion houses, agencies, and platform integrations.",
    color: "from-amber-500 to-orange-600",
    border: "border-amber-500/30",
    badge: null,
    features: [
      "Everything in Studio, plus:",
      "Custom LoRA training on your brand",
      "White-label deployment",
      "Dedicated GPU cluster",
      "Custom RAG knowledge base",
      "SSO + Team management",
      "SLA & uptime guarantee",
      "Dedicated success engineer",
    ],
    cta: "Contact Sales",
    ctaStyle: "glass border border-amber-500/40 text-amber-300 hover:bg-amber-500/10",
    href: "mailto:studio@fashionai.com",
  },
];

export default function PricingSection() {
  return (
    <section id="pricing" className="py-32 relative overflow-hidden scroll-mt-20">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-[hsl(225,25%,6%)] via-[hsl(225,22%,8%)] to-[hsl(225,25%,6%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_70%_50%_at_50%_50%,rgba(99,102,241,0.07),transparent)]" />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold text-emerald-300 border border-emerald-500/30 bg-emerald-500/10 mb-4">
            PRICING
          </span>
          <h2 className="text-5xl md:text-6xl font-black text-white mb-4">
            Simple, transparent{" "}
            <span className="gradient-text">pricing</span>
          </h2>
          <p className="text-xl text-slate-400 max-w-xl mx-auto">
            Start for free. Scale as you create. No hidden costs.
          </p>
        </motion.div>

        {/* Plans */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {PLANS.map((plan, i) => {
            const Icon = plan.icon;
            return (
              <motion.div
                key={plan.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                whileHover={i === 1 ? { scale: 1.02 } : { y: -4 }}
                className={`relative glass rounded-3xl p-8 border ${plan.border} flex flex-col ${
                  i === 1 ? "ring-1 ring-indigo-500/30 shadow-[0_0_60px_rgba(99,102,241,0.15)]" : ""
                }`}
              >
                {/* Popular badge */}
                {plan.badge && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <span className="px-4 py-1 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-xs font-bold">
                      {plan.badge}
                    </span>
                  </div>
                )}

                {/* Icon & name */}
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${plan.color} mb-4 w-fit`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-xl font-black text-white mb-1">{plan.name}</h3>
                <p className="text-slate-500 text-sm mb-5">{plan.description}</p>

                {/* Price */}
                <div className="flex items-baseline gap-2 mb-6">
                  <span className="text-4xl font-black text-white">{plan.price}</span>
                  <span className="text-slate-500 text-sm">{plan.sub}</span>
                </div>

                {/* Features */}
                <ul className="space-y-3 mb-8 flex-1">
                  {plan.features.map((f) => (
                    <li key={f} className="flex items-start gap-3 text-sm text-slate-400">
                      <Check className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                      {f}
                    </li>
                  ))}
                </ul>

                {/* CTA */}
                <a
                  href={plan.href}
                  target={plan.href.startsWith("http") ? "_blank" : undefined}
                  rel="noopener noreferrer"
                  className={`flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm transition-all duration-300 ${plan.ctaStyle}`}
                >
                  {plan.cta}
                  <ArrowRight className="w-4 h-4" />
                </a>
              </motion.div>
            );
          })}
        </div>

        {/* Bottom note */}
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-center text-slate-600 text-sm mt-10"
        >
          All plans include the local development environment. GPU generation requires a compatible NVIDIA GPU.{" "}
          <span className="text-slate-500">Currently running in Mock Mode (CPU) for demo purposes.</span>
        </motion.p>
      </div>
    </section>
  );
}
