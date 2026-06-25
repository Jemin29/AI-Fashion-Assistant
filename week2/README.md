# Week 2 — Text-to-Image Fashion Generation Foundation

> **AI-Powered Fashion Design Assistant · Week 2**
> Built on Stable Diffusion XL (SDXL) with clean, production-grade architecture.

---

## 📁 Project Structure

```
week2/
├── generator/                  # Core SDXL generation engine
│   ├── sdxl_generator.py       # High-level generation API
│   ├── model_manager.py        # Model loading, VRAM optimisation
│   ├── scheduler_factory.py    # 11 noise scheduler registry
│   └── image_processor.py      # Save, resize, watermark, grid
│
├── prompts/                    # Prompt engineering system
│   ├── prompt_builder.py       # Structured prompt assembly
│   ├── style_presets.py        # 9 fashion style preset registry
│   ├── negative_prompts.py     # Curated negative prompt library
│   └── prompt_validator.py     # Sanitise + token count check
│
├── evaluation/                 # Image quality evaluation
│   ├── metrics.py              # CLIP, quality checks, SSIM, PSNR
│   ├── quality_scorer.py       # Composite weighted scorer
│   └── evaluation_report.py    # JSON report builder
│
├── pipelines/                  # End-to-end generation pipelines
│   ├── base_pipeline.py        # Abstract base with lifecycle hooks
│   ├── text2image_pipeline.py  # Single-prompt pipeline
│   └── batch_pipeline.py       # Multi-prompt batch pipeline
│
├── configs/                    # YAML configuration files
│   ├── generation_config.yaml  # SDXL sampling hyperparameters
│   ├── model_config.yaml       # Model paths, dtype, VRAM tiers
│   ├── prompt_config.yaml      # Prompt templates, style presets
│   ├── evaluation_config.yaml  # Evaluation thresholds, metrics
│   └── logging_config.yaml     # Log levels, rotation, sinks
│
├── outputs/                    # Generated images & metadata
│   ├── images/                 # PNG/JPG generated images
│   ├── metadata/               # JSON metadata sidecars
│   └── evaluation_reports/     # Batch evaluation JSONs
│
├── logs/                       # Structured log files
├── notebooks/                  # Jupyter exploration notebooks
├── tests/                      # pytest test suite
│   ├── conftest.py             # Shared fixtures (no model load)
│   ├── test_prompt_builder.py  # 43 prompt system tests
│   ├── test_config_manager.py  # 20 config system tests
│   └── test_sdxl_generator.py  # 26 generator tests
│
├── docs/                       # Documentation
│   ├── architecture.md         # System architecture diagram
│   └── prompt_guide.md         # Prompt engineering best practices
│
├── config_manager.py           # Typed config loader (Pydantic v2)
├── logging_setup.py            # Multi-sink Loguru logging
├── requirements.txt            # All dependencies with versions
└── .env.example                # All environment variables documented
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# From the fashion-ai-assistant directory
cd week2

# Copy and fill in your settings
cp .env.example .env
# Edit .env — set HF_TOKEN, DEVICE, TORCH_DTYPE

# Install dependencies
pip install -r requirements.txt
```

### 2. Single Image Generation

```python
from week2.config_manager import get_config
from week2.logging_setup import setup_logging
from week2.pipelines import Text2ImagePipeline

setup_logging()
cfg      = get_config()
pipeline = Text2ImagePipeline(config=cfg)

result = pipeline.run(
    prompt = "A woman wearing an elegant red silk evening gown",
    style  = "luxury",
    gender = "women",
    season = "summer",
)
print(result)
# → PipelineRunResult(SUCCESS | generated=1 | passed=1 | time=8.3s)
```

### 3. Batch Generation

```python
from week2.pipelines import BatchPipeline

result = BatchPipeline.run_prompts(
    prompts = [
        "A man in a tailored navy blue suit",
        "A woman in a floral bohemian dress",
        "An oversized graphic streetwear hoodie",
    ],
    config = get_config(),
    style  = "formal",
)
print(f"{result.passed_evaluation}/{result.total_generated} images passed quality check")
```

### 4. Use a Generation Preset

```python
result = pipeline.run(
    prompt = "A luxury silk blouse",
    preset = "high_quality",   # 50 steps + refiner
)
```

---

## ⚙️ Configuration

All settings can be controlled via environment variables or YAML files.

### Key `.env` Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token (required for SDXL download) |
| `DEVICE` | `auto` | `auto` \| `cuda` \| `cpu` \| `mps` |
| `TORCH_DTYPE` | `float16` | `float16` \| `bfloat16` \| `float32` |
| `DEFAULT_STEPS` | `30` | Default inference steps |
| `DEFAULT_WIDTH` | `1024` | Default image width |
| `ENABLE_XFORMERS` | `true` | Memory-efficient attention |
| `ENABLE_CPU_OFFLOAD` | `false` | For low-VRAM GPUs (≤ 8 GB) |

### VRAM Requirements

| VRAM | Recommended Settings |
|------|---------------------|
| 4 GB | `enable_sequential_offload=true`, 768×768 |
| 8 GB | `enable_cpu_offload=true`, 1024×1024 |
| 12 GB | `enable_xformers=true`, 1024×1024 |
| 16 GB+ | Full quality + refiner |

---

## 🎨 Style Presets

9 pre-built fashion style presets are ready to use:

| Preset | Description |
|--------|-------------|
| `streetwear` | Urban, oversized, graphic, sneaker culture |
| `formal` | Tailored, elegant, professional |
| `casual` | Relaxed, everyday, minimalist |
| `athleisure` | Sporty-chic, performance wear |
| `bohemian` | Free-spirited, flowy, earthy |
| `luxury` | High-end designer, couture, premium |
| `minimalist` | Clean lines, neutral palette, Scandi |
| `avant_garde` | Conceptual, sculptural, experimental |
| `ethnic` | Traditional, embroidered, cultural |

---

## 🧪 Running Tests

```bash
# From fashion-ai-assistant/
python -m pytest week2/tests/ -v
# → 89 tests, all pass without GPU or real models
```

---

## 📊 Evaluation

Every generated image is automatically evaluated:

- **Resolution check** — minimum size validation
- **Black / white detection** — catches failed generations
- **CLIP similarity** — text-image alignment scoring (if enabled)
- **JSON report** — saved to `outputs/evaluation_reports/`

---

## 📝 Logging

Logs are written to `week2/logs/` with automatic rotation:

| Sink | Level | File |
|------|-------|------|
| Console | INFO | `stderr` |
| General | DEBUG | `week2_YYYY-MM-DD.log` |
| Errors | ERROR | `errors_YYYY-MM-DD.log` |
| Generation | INFO | `generation_YYYY-MM-DD.log` |

---

## 🔗 Integration with Week 1

```python
# Load a record from Week 1 dataset
from data_pipeline import load_fashion_dataset   # Week 1 module

records = load_fashion_dataset("processed/final_fashion_dataset.json")

# Generate an image from a Week 1 record
pipeline = Text2ImagePipeline(config=get_config())
result   = pipeline.run_from_record(records[0])
```

---

## 🚧 Week 3 Roadmap (NOT in Week 2)

- ControlNet for pose/structure conditioning
- LoRA adapter fine-tuning on fashion datasets
- IP-Adapter for reference image conditioning
- REST API for serving the generation pipeline
- Web interface (Gradio / Streamlit)

---

## 📦 Dependencies

Key packages (see `requirements.txt` for pinned versions):

- `diffusers >= 0.27` — SDXL pipeline
- `transformers >= 4.38` — CLIP text encoders
- `accelerate >= 0.28` — Device management
- `torch >= 2.2` — PyTorch backend
- `Pillow >= 10.2` — Image I/O
- `pydantic >= 2.6` — Config validation
- `loguru >= 0.7` — Structured logging
- `open-clip-torch >= 2.24` — CLIP evaluation
