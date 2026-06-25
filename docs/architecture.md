# Architecture — Week 2: Text-to-Image Fashion Generation

## Overview

Week 2 builds a complete **Text-to-Image Generation Foundation** on top of the
Week 1 data pipeline. It uses **Stable Diffusion XL (SDXL)** as the core
generative model, wrapped in clean, testable layers.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User / Application                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Pipeline Layer  │
                    │                  │
                    │ Text2ImagePipeline│
                    │  BatchPipeline   │
                    └────────┬─────────┘
                    ┌────────▼────────┐
         ┌──────────┤  Prompt Layer   ├──────────┐
         │          │                 │           │
         │    PromptBuilder   PromptValidator     │
         │    StylePresets    NegativePrompts     │
         └──────────┴────────┬────────┴───────────┘
                             │
                    ┌────────▼────────┐
                    │  Generator Layer │
                    │                  │
                    │  SDXLGenerator   │
                    │  ModelManager    │
                    │ SchedulerFactory │
                    │  ImageProcessor  │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │         SDXL Model           │
              │  (stabilityai/sdxl-base-1.0)│
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │ Evaluation Layer │
                    │                  │
                    │  QualityScorer   │
                    │    metrics.py    │
                    │ EvaluationReport │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │           Outputs            │
              │  outputs/images/*.png        │
              │  outputs/metadata/*.json     │
              │  outputs/evaluation_reports/ │
              └──────────────────────────────┘
```

---

## Layer Descriptions

### Config Layer (`config_manager.py`)
- Pydantic v2 models for all config sections
- YAML file loading with env override merging
- Singleton `get_config()` with LRU caching

### Logging Layer (`logging_setup.py`)
- Multi-sink Loguru setup (console + file + error + generation)
- Stdlib intercept so third-party libraries use the same pipeline
- `generation_logger()` context manager for tagged events

### Generator Layer (`generator/`)
| Module | Responsibility |
|--------|---------------|
| `model_manager.py` | Lazy model loading, xformers/offload, teardown |
| `scheduler_factory.py` | Registry of 11 noise schedulers |
| `sdxl_generator.py` | High-level `generate()` API returning `GenerationResult` |
| `image_processor.py` | Save, resize, watermark, grid, sidecar metadata |

### Prompts Layer (`prompts/`)
| Module | Responsibility |
|--------|---------------|
| `prompt_builder.py` | Assemble structured prompts from components |
| `style_presets.py` | Registry of 9 fashion style presets |
| `negative_prompts.py` | Curated negative prompt library |
| `prompt_validator.py` | Sanitise, check tokens, block NSFW |

### Evaluation Layer (`evaluation/`)
| Module | Responsibility |
|--------|---------------|
| `metrics.py` | CLIP similarity, quality checks, SSIM, PSNR |
| `quality_scorer.py` | Composite weighted scorer + batch scoring |
| `evaluation_report.py` | JSON report builder with summary stats |

### Pipeline Layer (`pipelines/`)
| Module | Responsibility |
|--------|---------------|
| `base_pipeline.py` | Abstract base with lifecycle hooks |
| `text2image_pipeline.py` | Full single-prompt generation pipeline |
| `batch_pipeline.py` | Multi-prompt batch pipeline with progress |

---

## Data Flow: Single Generation

```
Input prompt (str)
    │
    ▼
PromptValidator.validate()          ← Check: empty? NSFW? token count?
    │
    ▼
PromptBuilder.build()               ← Add: style tags, quality boosters,
    │                                         gender/season modifiers
    ▼
SDXLGenerator.generate()            ← SDXL inference, seed management
    │
    ├─── base pipeline call (scheduler, steps, guidance, size)
    └─── [optional] refiner pass
    │
    ▼
ImageProcessor.save_image()         ← PNG/JPG to disk
ImageProcessor.save_metadata_sidecar() ← JSON sidecar
    │
    ▼
QualityScorer.score()               ← Resolution, black/white, CLIP
    │
    ▼
EvaluationReport.save()             ← JSON report
    │
    ▼
PipelineRunResult                   ← Returned to caller
```

---

## Configuration Hierarchy

```
.env  →  pydantic-settings (Week2EnvSettings)
              │
              ▼
configs/generation_config.yaml   ┐
configs/model_config.yaml        ├──→  Week2Config (merged)
configs/prompt_config.yaml       │
configs/evaluation_config.yaml   ┘
```

Precedence (highest to lowest): **env vars > YAML file > Pydantic defaults**

---

## Memory Optimisation Levels

| Level | Setting | VRAM Required |
|-------|---------|---------------|
| None | Default (xformers only) | ~10 GB |
| CPU Offload | `enable_cpu_offload=true` | ~6 GB |
| Sequential Offload | `enable_sequential_offload=true` | ~4 GB |
| Attention Slicing | `enable_attention_slicing=true` | ~3 GB |

---

## Week 3 Boundary

> **Week 2 does NOT implement:**
> - ControlNet conditioning
> - LoRA adapter loading/merging
> - IP-Adapter
> - Real-time streaming generation
> - REST API / web server
> - Database storage of results
