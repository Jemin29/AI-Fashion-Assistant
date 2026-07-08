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

```
Raw Datasets (HDF5 / Image Folders)
        │
        ▼
┌─────────────────────┐
│   Data Ingesters    │  ← fashiongen_ingester.py
│   (Stream records)  │  ← deepfashion_ingester.py
└────────┬────────────┘
         │  FashionRecord dicts
         ▼
┌─────────────────────┐
│  Image Preprocessor │  ← image_preprocessor.py
│  resize · filter    │
│  normalize · save   │
└────────┬────────────┘
         │  ProcessingResult dicts
         ▼
┌─────────────────────┐
│   Data Validator    │  ← data_validator.py
│  schema · dims      │
│  file · text checks │
└────────┬────────────┘
         │  Validated records
         ▼
┌─────────────────────┐
│ Metadata Generator  │  ← metadata_generator.py
│  CLIP embeddings    │
│  text embeddings    │
│  Parquet / JSONL    │
└─────────────────────┘
         │
         ▼
  datasets/metadata/
  datasets/processed/
  outputs/chroma_db/
```

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
├── datasets/                         # Raw & processed dataset files
│   ├── fashiongen/                   # FashionGen HDF5 files (gitignored)
│   ├── deepfashion/                  # DeepFashion images + annotations
│   ├── processed/                    # Preprocessed output images
│   └── metadata/                     # Parquet + JSONL metadata files
│
├── data_pipeline/                    # Core data engineering package
│   ├── __init__.py                   # Package root
│   ├── ingestion/
│   │   ├── __init__.py               # Exports: FashionGenIngester, DeepFashionIngester
│   │   ├── fashiongen_ingester.py    # HDF5 streaming ingester
│   │   ├── deepfashion_ingester.py   # Image-folder ingester with annotation parsing
│   │   └── download_fashiongen.py    # Download guide script
│   ├── preprocessing/
│   │   ├── __init__.py               # Exports: FashionPreprocessor
│   │   └── image_preprocessor.py    # Resize, quality filter, normalize, save
│   ├── validation/
│   │   ├── __init__.py               # Exports: DataValidator
│   │   └── data_validator.py        # Schema + statistical quality gates
│   └── metadata_generation/
│       ├── __init__.py               # Exports: MetadataGenerator
│       └── metadata_generator.py    # CLIP + text embeddings + Parquet export
│
├── configs/
│   ├── settings.yaml                 # Master config: datasets, pipeline, compute
│   └── logging_config.yaml          # Loguru multi-sink logging config
│
├── notebooks/
│   └── 01_dataset_exploration.py    # Week 1 EDA notebook (jupytext format)
│
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py            # 9 unit tests for both ingesters
│   ├── test_preprocessing.py        # 11 unit tests for image preprocessor
│   └── test_validation.py           # 13 unit tests for data validator
│
├── logs/                            # Rotating log files (gitignored)
├── outputs/                         # Model outputs, plots, embeddings
├── docs/                            # Extended documentation
│
├── .env.example                     # Environment variable template
├── .gitignore                       # Excludes datasets, credentials, builds
├── requirements.txt                 # Full dependency list with pinned versions
├── setup.py                         # Editable package install configuration
└── pytest.ini                       # Pytest discovery and marker configuration
```

---

## 📄 File Descriptions

### Core Pipeline Modules

| File | Purpose |
|---|---|
| [`fashiongen_ingester.py`](data_pipeline/ingestion/fashiongen_ingester.py) | Streams records from FashionGen HDF5 files. Handles byte decoding, lazy loading, and key validation. |
| [`deepfashion_ingester.py`](data_pipeline/ingestion/deepfashion_ingester.py) | Parses all DeepFashion annotation files at init, then streams image + metadata records. |
| [`image_preprocessor.py`](data_pipeline/preprocessing/image_preprocessor.py) | Resizes to 256×256, detects blur/low-contrast, saves JPEG, and returns ImageNet-normalized float32 tensors. |
| [`data_validator.py`](data_pipeline/validation/data_validator.py) | Validates records against a schema: required fields, allowed categoricals, image dimensions, file existence, and description quality. |
| [`metadata_generator.py`](data_pipeline/metadata_generation/metadata_generator.py) | Computes CLIP image embeddings and SentenceTransformer text embeddings; saves as `.npy` and exports metadata to Parquet. |

### Configuration Files

| File | Purpose |
|---|---|
| [`configs/settings.yaml`](configs/settings.yaml) | Master configuration: dataset paths, pipeline parameters, schema definition, logging, compute settings. |
| [`configs/logging_config.yaml`](configs/logging_config.yaml) | Loguru sink definitions: colorized console + rotating file logs + error-only file. |
| [`.env.example`](.env.example) | Template for all environment variables: API keys, dataset paths, model names, device settings. |

### Test Files

| File | Coverage |
|---|---|
| [`tests/test_ingestion.py`](tests/test_ingestion.py) | FashionGen byte decoding, record schema, HDF5 structure validation; DeepFashion annotation loading. |
| [`tests/test_preprocessing.py`](tests/test_preprocessing.py) | Success/failure paths, output shapes, MD5 determinism, normalization layout (CHW), config dataclass. |
| [`tests/test_validation.py`](tests/test_validation.py) | Required fields, categorical values, dimensions, file checks, description length, batch aggregation. |

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

## 📚 References

- [FashionGen: The Generative Fashion Dataset and Challenge](https://arxiv.org/abs/1806.08317) — Rostamzadeh et al., 2018
- [DeepFashion: Powering Robust Clothes Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) — Liu et al., CVPR 2016
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al., OpenAI 2021
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Reimers & Gurevych, 2019

---

*Built with ❤️ by the Fashion AI Team — Week 1, June 2026*
