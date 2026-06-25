# Week 3 — Style Control with ControlNet (Setup & Architecture)

> **AI-Powered Fashion Design Assistant · Week 3**  
> Foundational setup for spatial and structure conditioning using ControlNet with Stable Diffusion XL (SDXL).

---

## 📁 Project Structure

```
week3/
├── configs/                  # YAML configuration files
│   ├── model_config.yaml     # SDXL checkpoint paths & device runtimes
│   └── controlnet_config.yaml# ControlNet model IDs, weights & scales
│
├── controlnet/               # Core ControlNet model loader & manager
│   └── __init__.py           # Lazy loading stubs
│
├── preprocessors/            # Image feature extraction modules
│   └── __init__.py           # Canny, Openpose, Depth map module stubs
│
├── datasets/                 # Datasets and reference poses folder
│   └── .gitkeep
│
├── evaluation/               # Metrics and pose alignment scoring
│   └── __init__.py
│
├── pipelines/                # ControlNet-aware generation pipelines
│   └── __init__.py
│
├── outputs/                  # Generated designs & extracted control maps
│   └── .gitkeep
│
├── logs/                     # Multi-sink log files
│   └── .gitkeep
│
├── tests/                    # ControlNet unit tests
│   └── __init__.py
│
├── docs/                     # Visual design diagrams & setup instructions
│   └── .gitkeep
│
├── config_manager.py         # Typed Pydantic v2 configurations manager
├── logging_setup.py          # Loguru logging setup supporting rotated sinks
├── requirements.txt          # Python packages with pinned dependencies
└── README.md                 # This guide
```

---

## 🚀 Environment Setup

### 1. Prerequisite Packages
Ensure you install the required packages in your Python virtual environment:
```bash
# Navigate to the week3 directory
cd week3

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Settings (`.env`)
Create a `.env` file in the project root folder or set environment variables:
```bash
# Device configuration (auto | cuda | mps | cpu)
FASHION_CONFIG__MODEL__RUNTIME__DEVICE=cuda

# HuggingFace token (for model downloading)
HF_TOKEN=your_huggingface_write_token

# Output and logging overrides (optional)
FASHION_CONFIG__OUTPUT_DIR=week3/outputs
FASHION_CONFIG__LOG_DIR=week3/logs
```

---

## ⚙️ ControlNet Overview

ControlNet is a neural network architecture designed to add spatial conditioning controls to text-to-image diffusion models. For fashion generation, this allows the assistant to maintain structured outlines (Canny edges), body poses (Openpose), or clothing depth (Depth map) while changing colors, materials, styles, or backgrounds.

### Supported Conditionings:

1. **Canny Edge Detection**:
   * **Purpose**: Capture the exact silhouette of collars, pockets, buttons, and stitching lines.
   * **Model**: `diffusers/controlnet-canny-sdxl-1.0`
   * **Input**: Grayscale edge map.

2. **Openpose Detection**:
   * **Purpose**: Control the model's stance, hands, and facial direction, ensuring catalog-ready poses.
   * **Model**: `diffusers/controlnet-openpose-sdxl-1.0`
   * **Input**: Pose skeleton joints.

3. **Depth Map Estimation**:
   * **Purpose**: Manage the 3D drapes and wrinkles of fabrics, clothing folds, and layers.
   * **Model**: `diffusers/controlnet-depth-sdxl-1.0`
   * **Input**: Depth gradient map (lighter = closer, darker = further).

---

## ⚙️ Configuration Properties

The Pydantic configuration (`config_manager.py`) maps the following default settings:

| YAML Key | Environment Override | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `controlnet.enabled` | `FASHION_CONFIG__CONTROLNET__ENABLED` | `true` | Enable/Disable ControlNet |
| `controlnet.model_id` | `FASHION_CONFIG__CONTROLNET__MODEL_ID` | `diffusers/controlnet-canny-sdxl-1.0` | Default ControlNet model ID |
| `controlnet.conditioning_scale` | `FASHION_CONFIG__CONTROLNET__CONDITIONING_SCALE` | `0.8` | Influence weight (0.0 to 1.0) |
| `controlnet.control_guidance_start` | `FASHION_CONFIG__CONTROLNET__CONTROL_GUIDANCE_START` | `0.0` | Timestep start range |
| `controlnet.control_guidance_end` | `FASHION_CONFIG__CONTROLNET__CONTROL_GUIDANCE_END` | `1.0` | Timestep end range |
| `preprocessors.canny.resolution` | `FASHION_CONFIG__PREPROCESSORS__CANNY__RESOLUTION` | `1024` | Resolution size |

---

## 📝 Logging System

Logging is fully centralized. When `setup_logging()` is invoked, logs are mapped to:
* **Console logs**: Standard output for generation updates.
* **`logs/week3_general.log`**: Standard operational logs.
* **`logs/errors.log`**: Unhandled errors and stacks.
* **`logs/controlnet.log`**: dedicated logging sink filtered specifically for ControlNet preprocessor actions, weights caching, and model loading.
