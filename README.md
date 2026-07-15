# 🧵 AI-Powered Fashion Design Assistant

> **Week 1 — Fashion Domain Research & Dataset Curation**  
> Production-grade data engineering pipeline for large-scale fashion AI.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [File Descriptions](#-file-descriptions)
- [Quick Start](#-quick-start)
- [Data Pipeline](#-data-pipeline)
- [Configuration](#-configuration)
- [Running Tests](#-running-tests)
- [Week 1 Deliverables](#-week-1-deliverables)
- [Roadmap](#-roadmap)

---

## 🎯 Project Overview

This repository implements **Week 1** of an AI-Powered Fashion Design Assistant.
The goal of this week is to establish the complete data engineering foundation:
researching fashion datasets, building an ingestion pipeline, preprocessing images,
validating data quality, and generating rich embeddings-based metadata.

### What we're building

| Layer | Technology | Purpose |
|---|---|---|
| Dataset Ingestion | h5py, Pillow | Stream FashionGen (HDF5) & DeepFashion (folders) |
| Preprocessing | Pillow, OpenCV | Resize, quality filter, normalize images |
| Validation | Pydantic, Great Expectations | Schema + statistical quality gates |
| Metadata Generation | CLIP, SentenceTransformers | Visual + text embeddings |
| Storage | Parquet, ChromaDB | Efficient columnar + vector storage |
| Testing | pytest | 30+ unit tests with mocked datasets |

---

## 🏗 Architecture

The AI-Powered Fashion Design Assistant uses a modular multi-tier architecture separating core generative models, data structures, and the service routing layer:

1. **Next.js Landing Page**: The main public portal of the application, serving as the visual gate. It routes users dynamically to either the local/staging backend API or the interactive **Open Studio** UI, based on environment configuration (`NEXT_PUBLIC_STUDIO_URL`).
2. **Gradio Creative Studio (Open Studio UI)**: Located in [`apps/gradio_app/app.py`](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/apps/gradio_app/app.py), this is the front-facing UI that serves as the creative design board (sketch control, LoRA style mixing, multi-stage prompts).
3. **FastAPI Serving Backend**: Located in [`week7/backend/main.py`](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/week7/backend/main.py), it exposes production-ready REST endpoints, API routes, security middleware, and task queues.
4. **Celery Task Queue & Redis Cache**: Coordinates long-running generative model tasks asynchronously (background jobs) to keep the REST API responsive.
5. **ChromaDB & Vector Search Layer**: Handles multi-modal search querying (using CLIP embeddings) to support retrieval-augmented generation (RAG) and fashion design recommendations.
6. **Core Engines (Canonical Packages)**: Located under [`src/`](file:///c:/Users/HP/Desktop/AI%20Fashion%20Agent/fashion-ai-assistant/src/), it houses the core algorithmic code:
   - `src/generation/`: SDXL diffusion generator engine.
   - `src/lora/`: Style registries, mixing, and parameter-efficient adapters.
   - `src/controlnet/`: Preprocessing and conditioning models.
   - `src/rag/`: Retrieval-augmented semantic agent.
   - `src/recommendations/`: Collaborative recommendation logic.
   - `src/trends/`: Trend metrics analysis.
7. **Versioned Milestone Iterations**:
   - `week6/services/`: Hosts the core service logic wrappers and adapter layers actively utilized by the `week7` FastAPI router.
   - `week2/` and `week3/`: Expose utility configuration parsing schemas and sketch preprocessing hooks utilized dynamically by the `src/` modules.
   - `week8/`: Exposes DevOps and containerization configurations (`docker-compose`, Dockerfiles).
   - `archive/` (e.g. `archive/week4`, `archive/week5`): Legacy milestone directories that have been fully superseded by the canonical implementations in `src/` and are archived for repo cleanliness.

---

## 📦 Datasets

### 1. FashionGen
| Property | Value |
|---|---|
| Total Images | 293,000 |
| Resolution | 256 × 256 px (RGB) |
| Format | HDF5 (.h5) |
| Annotations | Natural-language descriptions, category, sub-category, gender |
| Access | Registration required at [fashion-gen.com](https://fashion-gen.com) |

**Download guide:**
```bash
python data_pipeline/ingestion/download_fashiongen.py
```

### 2. DeepFashion
| Property | Value |
|---|---|
| Total Images | 800,000+ |
| Categories | 50 fine-grained fashion categories |
| Attributes | 1,000 binary clothing attributes |
| Annotations | BBox, 6 landmarks, train/val/test splits |
| Access | [liuziwei7.github.io/projects/DeepFashion](https://liuziwei7.github.io/projects/DeepFashion.html) |

**Kaggle download (quick access):**
```bash
kaggle datasets download -d nguyngiabol/colorful-fashion-dataset-for-object-detection
```

---

## 📁 Project Structure

```
fashion-ai-assistant/
│
├── apps/                             # Interactive creative UI apps
│   └── gradio_app/                   # Gradio Open Studio UI front-facing portal
│       └── app.py                    # Main Gradio application runner
│
├── src/                              # Core canonical package modules
│   ├── data/                         # Data ingestion and verification pipelines
│   │   ├── ingestion/                # Stream HDF5 and image folders ingesters
│   │   ├── preprocessing/            # Image resizing and normalization
│   │   ├── validation/               # Statistical and schema quality gates
│   │   └── metadata_generation/      # Multimodal visual & text embedding extraction
│   ├── generation/                   # Stable Diffusion XL models and generators
│   ├── lora/                         # style registries and adapter mixing logic
│   ├── controlnet/                   # Preprocessors and sketch conditioning systems
│   └── rag/                          # Retrieval-Augmented Generation agent
│
├── week6/                            # Milestone service layer (actively used)
│   └── services/                     # Business logic and adapters for fastapi/celery
│
├── week7/                            # Active fastapi server and Celery background workers
│   └── backend/                      # fastapi routes, dependencies, schemas, entry points
│
├── week8/                            # Deployment, DevOps and docker compose configurations
│
├── datasets/                         # Datasets storage (gitignored)
│
├── landing/                          # Public landing page (Next.js & Vanilla CSS)
│
├── tests/                            # 2,900+ test cases covering all modules
│
├── archive/                          # Superseded milestone directories (week4, week5)
│
├── configs/                          # YAML/YAML-overridden configuration schemas
│
├── setup.py                          # Editable package installer metadata
└── requirements.txt                  # Consolidated python dependency manifests
```

---

## 📄 File Descriptions

### Core Pipeline & Engine Modules

| File | Purpose |
|---|---|
| [`src/data/ingestion/`](src/data/ingestion/) | Ingests raw fashion metadata, handles HDF5 bytes parsing, and streams parsed design records lazily. |
| [`src/data/preprocessing/`](src/data/preprocessing/) | Normalizes image pixel channels, handles size conversions, and quarantined corrupt records. |
| [`src/data/validation/`](src/data/validation/) | Coordinates strict Pydantic validation checks on schema attributes. |
| [`src/generation/`](src/generation/) | Hosts base generative Diffusion models and configures schedulers. |
| [`src/lora/`](src/lora/) | Manages styled design blending and adapter mixing parameters. |
| [`src/controlnet/`](src/controlnet/) | Performs outline conditioning maps generation and ControlNet model routing. |

### Configuration Files

| File | Purpose |
|---|---|
| [`configs/settings.yaml`](configs/settings.yaml) | Master configuration: dataset paths, pipeline parameters, schema definition, logging, compute settings. |
| [`.env.example`](.env.example) | Template for all environment variables: API keys, dataset paths, model names, device settings. |

---

## ⚙️ Running in Real Mode vs. Mock Mode

The system defaults to `GLOBAL_MOCK=True` everywhere to facilitate fast testing and application logic development without requiring expensive GPU hardware resources.

### 1. Mock Mode (`GLOBAL_MOCK=True`)
* **Hardware Requirements**: None (runs on standard CPU/laptop).
* **Use Case**: UI prototyping, frontend landing page integration, mock API testing, local development.
* **Behavior**:
  - The image generation pipelines return pre-configured color gradient placeholder image results immediately.
  - The vector databases (ChromaDB & FAISS) run in fallback mode with deterministic mock embeddings.
  - The RAG Fashion Assistant returns responses grounded in local mock data without calling remote sentence-transformer encoders.

### 2. Real Mode (`GLOBAL_MOCK=False`)
* **Hardware Requirements**: A CUDA-capable GPU with at least **12GB VRAM** (e.g., NVIDIA RTX 3080/4070 or better).
* **Dependencies**: Requires all packages listed in `requirements.txt` to be installed (specifically `diffusers`, `peft`, `accelerate`, `controlnet-aux`, `invisible-watermark`, `wandb`, and `xformers`).
* **Activation**:
  - For the **Gradio app** (`week6/run.py`):
    ```bash
    python week6/run.py --no-mock
    ```
    Alternatively, set `FASHION_STUDIO_MOCK_MODE=false` in `week6/.env`.
  - For the **FastAPI Backend**:
    Set `model__global_mock=False` in `week7/backend/configs/.env` or set the environment variable:
    ```bash
    export model__global_mock=False
    ```

---

## ⚡ Quick Start

### 1. Clone and enter the project
```bash
git clone <your-repo-url>
cd fashion-ai-assistant
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies
```bash
# Core dependencies only
pip install -r requirements.txt

# Or as editable package (recommended for development)
pip install -e ".[dev]"
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env with your Kaggle credentials and dataset paths
```

### 5. Download datasets
```bash
# FashionGen — follow instructions printed by script
python data_pipeline/ingestion/download_fashiongen.py

# DeepFashion — via Kaggle API
kaggle datasets download -d nguyngiabol/colorful-fashion-dataset-for-object-detection
unzip colorful-fashion-dataset-for-object-detection.zip -d datasets/deepfashion/
```

### 6. Run the EDA notebook
```bash
pip install jupytext jupyter
jupytext --to notebook notebooks/01_dataset_exploration.py
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

---

## 🔄 Data Pipeline

### Minimal pipeline example
```python
from data_pipeline.ingestion import FashionGenIngester
from data_pipeline.preprocessing import FashionPreprocessor
from data_pipeline.validation import DataValidator
from data_pipeline.metadata_generation import MetadataGenerator

# 1. Ingest
ingester = FashionGenIngester("datasets/fashiongen/fashiongen_256_256_train.h5")

# 2. Preprocess
preprocessor = FashionPreprocessor(output_dir="datasets/processed")

# 3. Validate
validator = DataValidator()

# 4. Generate metadata
generator = MetadataGenerator(output_dir="datasets/metadata")

# Stream and process
BATCH_SIZE = 100
batch = []

for record in ingester.stream(max_items=1000):
    result = preprocessor.process(record)

    if result.success:
        report = validator.validate_record(result.metadata)
        if report.is_valid:
            batch.append(result.metadata)

    if len(batch) >= BATCH_SIZE:
        generator.generate_batch(batch)
        batch.clear()

# Flush remaining records
if batch:
    generator.generate_batch(batch)

# Persist to Parquet
generator.flush_to_parquet("fashiongen_train_meta.parquet")
```

---

## ⚙️ Configuration

All parameters are controlled via **three sources** (env file overrides YAML / Next.js defaults):

| Parameter | YAML Key | .env Variable / Config | Default / Description |
|---|---|---|---|
| FashionGen HDF5 path | `datasets.fashiongen.hdf5_file` | `FASHIONGEN_HDF5_PATH` | HDF5 file path |
| Target image size | `pipeline.preprocessing.target_size` | `TARGET_IMAGE_SIZE` | Preprocessor target dimensions |
| Compute device | `compute.device` | `DEVICE` | Compute target (cuda, cpu) |
| CLIP model | `pipeline.metadata_generation.clip_model` | `CLIP_MODEL_NAME` | Model naming tag |
| Batch size | `pipeline.metadata_generation.batch_size` | `EMBED_BATCH_SIZE` | Embedding batch slice size |
| Log level | `logging.level` | `LOG_LEVEL` | App logger console verbosity |
| Gradio Studio URL | - | `NEXT_PUBLIC_STUDIO_URL` | Gradio server URL (`http://127.0.0.1:7860`) |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Verbose with coverage report
pytest -v --cov=data_pipeline --cov-report=html

# Run a specific test file
pytest tests/test_preprocessing.py -v

# Run only fast unit tests (skip slow/integration)
pytest -m "not slow and not integration"
```

**Expected output (all tests should pass without dataset files):**
```
tests/test_ingestion.py    .........   9 passed
tests/test_preprocessing.py ...........  11 passed
tests/test_validation.py   .............  13 passed

============= 33 passed in 4.2s =============
```

---

## ✅ Week 1 Deliverables

| Deliverable | Status |
|---|---|
| ✅ Professional project structure | Complete |
| ✅ `fashiongen_ingester.py` — HDF5 streaming | Complete |
| ✅ `deepfashion_ingester.py` — annotation parsing | Complete |
| ✅ `image_preprocessor.py` — resize, filter, normalize | Complete |
| ✅ `data_validator.py` — multi-layer quality gates | Complete |
| ✅ `metadata_generator.py` — CLIP + text embeddings | Complete |
| ✅ `configs/settings.yaml` — master configuration | Complete |
| ✅ `requirements.txt` — pinned dependencies | Complete |
| ✅ `.env.example` — credential template | Complete |
| ✅ `tests/` — 33 unit tests | Complete |
| ✅ `notebooks/01_dataset_exploration.py` — EDA | Complete |
| ✅ `README.md` — full documentation | Complete |

---

## 🗺 Roadmap

| Week | Focus |
|---|---|
| **Week 1** ✅ | Fashion Domain Research & Dataset Curation |
| Week 2 | Full Pipeline Run — End-to-end ingestion & metadata generation |
| Week 3 | Generative Model Architecture Design (Diffusion / GAN) |
| Week 4 | Model Training — Conditional image generation |
| Week 5 | Fine-tuning & Style Transfer |
| Week 6 | Retrieval-Augmented Generation (RAG) for fashion |
| Week 7 | API Development & Serving Layer |
| Week 8 | UI/UX & Full Product Integration |

---

## 📥 Full Dataset Download and Execution Guide

To reproduce the ingestion pipeline at full scale with the complete datasets, follow these steps:

### 1. FashionGen Dataset (~15 GB)
1. Register for free academic access at [fashion-gen.com](https://fashion-gen.com/).
2. Once approved, download:
   - `fashiongen_256_256_train.h5` (~13 GB)
   - `fashiongen_256_256_val.h5` (~1.7 GB)
3. Move both `.h5` files directly into:
   ```bash
   datasets/fashiongen/
   ```
4. Run the full-scale ingestion pipeline:
   ```bash
   python src/data/ingestion/fashiongen_loader.py
   ```

### 2. DeepFashion Dataset (~30 GB)
1. Request access and download the **Category and Attribute Prediction Benchmark** from the [official DeepFashion page](https://liuziwei7.github.io/projects/DeepFashion.html).
2. Alternatively, download the colorful-fashion dataset via Kaggle:
   ```bash
   kaggle datasets download -d nguyngiabol/colorful-fashion-dataset-for-object-detection
   ```
3. Extract the downloaded archives so your workspace structure matches:
   ```
   datasets/deepfashion/
   ├── img/               ← Nested clothing category images
   ├── Anno/
   │   ├── list_category_cloth.txt
   │   ├── list_category_img.txt
   │   ├── list_attr_cloth.txt
   │   ├── list_attr_img.txt
   │   ├── list_bbox.txt
   │   └── list_landmarks.txt
   └── Eval/
       └── list_eval_partition.txt
   ```
4. Run the full-scale ingestion pipeline:
   ```bash
   python src/data/ingestion/deepfashion_loader.py --split train
   ```

---

## 📚 References

- [FashionGen: The Generative Fashion Dataset and Challenge](https://arxiv.org/abs/1806.08317) — Rostamzadeh et al., 2018
- [DeepFashion: Powering Robust Clothes Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) — Liu et al., CVPR 2016
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al., OpenAI 2021
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Reimers & Gurevych, 2019

---

*Built with ❤️ by the Fashion AI Team — Week 1, June 2026*
