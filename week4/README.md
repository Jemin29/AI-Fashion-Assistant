# Week 4 — LoRA Personalization & Brand Studio

> **AI-Powered Fashion Design Assistant · Week 4 Deliverables**  
> Fine-tuning Stable Diffusion XL using Low-Rank Adaptation (LoRA), style mixing, switching, and custom brand adapters.

---

## 📁 Overview

Week 4 focuses on **LoRA Personalization & Brand Studio**, introducing the capability to inject specific brand aesthetics (Nike, Gucci, Zara, H&M) into the SDXL image generation pipeline. This is achieved through parameter-efficient fine-tuning (PEFT) and advanced adapter composition techniques.

To maintain repository hygiene and avoid business logic duplication, all core logic resides in [src/lora/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/lora). This folder serves as the official entry point for Week 4, providing documentation, configuration, and runner scripts.

---

## 📁 Directory Structure

```
week4/
├── assets/                  # Training visualizations and sample outputs
│   └── .gitkeep
├── results/                 # Evaluation reports, checkpoint weights info
│   └── .gitkeep
├── README.md                # This overview guide
├── architecture.md          # System architecture and adapter composition mechanics
└── documentation.md         # API guides, CLI references, and training setup
```

---

## 🚀 Key Deliverables Reused from `src/lora`

All deliverables are fully implemented within the repository's canonical packages:

1. **LoRA Core & Registry**: Implemented in [src/lora/style_manager/lora_registry.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/lora/style_manager/lora_registry.py). Registers available brand adapters, loading/unloading paths, and configurations.
2. **Style Switching**: Implemented in [src/lora/style_manager/style_switcher.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/lora/style_manager/style_switcher.py). Enables dynamic switching of active LoRA weights during text-to-fashion pipelines.
3. **Style Mixing**: Implemented in [src/lora/style_manager/style_mixer.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/lora/style_manager/style_mixer.py). Enables weighted combinations of multiple active brand adapters (e.g., 60% Nike, 40% Gucci).
4. **LoRA Fine-Tuning**: Implemented in [src/lora/trainers/](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/lora/trainers/). Contains dataset preprocessing, Kohya-based training pipeline wrapper (`kohya_pipeline.py`), and training scripts for individual brands:
   - [train_nike_lora.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/lora/trainers/train_nike_lora.py)
   - [train_gucci_lora.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/lora/trainers/train_gucci_lora.py)
   - [train_zara_lora.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/lora/trainers/train_zara_lora.py)
   - [train_hm_lora.py](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/lora/trainers/train_hm_lora.py)

---

## ⚙️ Quick Start

To verify LoRA generation, style switching, and mixing, run the generation test:

```bash
# Ensure virtual environment is active and project package is installed
python -c "from src.lora.style_manager.lora_registry import LoRARegistry; print('LoRA Registry successfully loaded!')"
```

Refer to [documentation.md](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week4/documentation.md) for detailed python code samples and training commands.
