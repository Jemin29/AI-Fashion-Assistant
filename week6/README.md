# Week 6 — AI Fashion Creative Studio

> **AI-Powered Fashion Design Assistant · Week 6 Graduation Module**  
> A production-quality Gradio Creative Studio integrating all Week 1-5 AI modules.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Setup](#setup)
- [Running the Studio](#running-the-studio)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Overview

Week 6 is the **Creative Studio** — a fully integrated, interactive Gradio application that brings together all five weeks of AI development into one premium-quality interface:

| Module | Week | Capability |
|--------|------|-----------|
| Data Pipeline | Week 1 | Fashion domain knowledge and schema |
| SDXL Generator | Week 2 | Text-to-fashion image generation |
| ControlNet | Week 3 | Sketch/Pose/Depth-conditioned generation |
| LoRA Personalization | Week 4 | Brand-specific fine-tuned generation |
| RAG Intelligence | Week 5 | Semantic search, Q&A, recommendations, trends |

---

## Architecture

```
week6/
├── gradio_app/       # App factory, state management
├── components/       # Reusable UI building blocks
├── pages/            # Full Gradio tab implementations (8 pages)
├── services/         # Thin adapters over src/ modules (mock-safe)
├── themes/           # Custom premium dark-mode theme + CSS
├── assets/           # Static CSS, images, fonts
├── configs/          # YAML configuration files
├── outputs/          # Generated images, reports, sketches
├── logs/             # Rotating application logs
├── tests/            # Pytest test suite
└── docs/             # Technical documentation
```

---

## Features

### 8 Interactive Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Live system status dashboard, model health, quick-start guide |
| 🎨 **Style Studio** | SDXL text-to-fashion generation with visual prompt builder |
| ✏️ **ControlNet Studio** | Upload sketch/pose/depth image → conditioned fashion output |
| 🏷️ **Brand Studio** | LoRA-powered brand generation (Nike, Gucci, Zara, H&M) + style mixing |
| 💬 **Fashion Q&A** | RAG-grounded conversational assistant with citation display |
| 📈 **Trend Explorer** | Real-time trend analysis, velocity forecasting, seasonal outlook |
| 👗 **Recommend Hub** | Personalized style + brand recommendations with user profiling |
| 📊 **Eval Dashboard** | Live RAG metrics (Hit Rate@K, MRR), generation quality charts |

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional — full mock mode for CPU-only)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jemin29/AI-Fashion-Agent.git
cd AI-Fashion-Agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install Week 6 dependencies
pip install -r week6/requirements.txt

# 4. Install the src package (Week 1-5 modules)
pip install -e .

# 5. Set up environment variables
cp week6/.env.example week6/.env
# Edit week6/.env with your settings
```

### Environment Variables

```bash
# week6/.env
FASHION_STUDIO_HOST=0.0.0.0
FASHION_STUDIO_PORT=7860
FASHION_STUDIO_MOCK_MODE=true   # Set false to use real GPU models
FASHION_STUDIO_LOG_LEVEL=INFO
HUGGINGFACE_TOKEN=your_token_here  # For downloading gated models
```

---

## Running the Studio

```bash
# Default launch (mock mode, port 7860)
python week6/run.py

# With custom host/port
python week6/run.py --host 0.0.0.0 --port 7861

# With real GPU models (requires model downloads)
python week6/run.py --no-mock

# Public share link
python week6/run.py --share
```

The studio will open at: **http://localhost:7860**

---

## Configuration

Edit `week6/configs/app_config.yaml` to control:
- Server settings (host, port, share)
- Feature flags (which pages to show)
- Mock mode toggle per service
- Output directories and retention

Edit `week6/configs/services_config.yaml` to configure:
- Model paths and HuggingFace model IDs
- ChromaDB collection names
- Embedding model selection
- LoRA adapter paths

---

## Testing

```bash
# Run full test suite
cd week6
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=term-missing

# Specific module
python -m pytest tests/test_services.py -v
```

---

## Project Structure

```
week6/
├── gradio_app/
│   ├── __init__.py
│   ├── main.py             # gr.Blocks app factory
│   └── app_state.py        # Session state management
├── components/
│   ├── header.py           # Hero banner
│   ├── metrics_panel.py    # Statistics cards
│   ├── image_gallery.py    # Image display + download
│   ├── chat_interface.py   # Stateful chat component
│   └── status_bar.py       # System health indicators
├── pages/
│   ├── home.py
│   ├── style_studio.py
│   ├── controlnet_page.py
│   ├── brand_studio.py
│   ├── fashion_qa.py
│   ├── trend_explorer.py
│   ├── recommend_hub.py
│   └── eval_dashboard.py
├── services/
│   ├── generation_service.py
│   ├── controlnet_service.py
│   ├── lora_service.py
│   ├── rag_service.py
│   ├── recommendation_service.py
│   └── trend_service.py
├── themes/
│   └── fashion_theme.py
├── assets/css/studio.css
├── configs/
│   ├── app_config.yaml
│   └── services_config.yaml
├── tests/
│   ├── conftest.py
│   ├── test_services.py
│   ├── test_pages.py
│   └── test_components.py
├── docs/
│   ├── Week6_Report.md
│   └── API_Reference.md
├── requirements.txt
├── logging_config.yaml
├── .env.example
└── run.py
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI Framework | Gradio 4.x |
| Image Generation | Stable Diffusion XL |
| Conditioning | ControlNet |
| Personalization | LoRA / PEFT |
| Knowledge Retrieval | ChromaDB + RAG |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Logging | Loguru + Rich |
| Testing | Pytest + pytest-cov |
| Config | PyYAML + Pydantic Settings |

---

## License

MIT — see [LICENSE](../LICENSE)
