# Week 4 — API & Usage Documentation

This document describes how to programmatically interact with the LoRA personalization engine, register custom adapters, perform style switching/mixing, and run fine-tuning pipelines.

---

## 🛠️ Python API Reference

### 1. Load and Register Adapters

```python
from src.lora.style_manager.lora_registry import LoRARegistry

# Initialize registry and load configuration
registry = LoRARegistry(config_path="configs/model_config.yaml")

# Get list of registered brand adapters
available = registry.list_adapters()
print(f"Registered adapters: {available}")
# Output: ['nike', 'gucci', 'zara', 'hm']
```

### 2. Live Style Switching

Use the `StyleSwitcher` wrapper to hot-swap style adapters on an active pipeline:

```python
from src.lora.style_manager.style_switcher import StyleSwitcher

# Instantiate switcher on existing SDXL pipeline
switcher = StyleSwitcher(pipeline)

# Enable Gucci adapter
switcher.apply_style("gucci", scale=0.8)

# Run generation
image_gucci = pipeline(prompt="A professional business blazer, editorial portrait").images[0]

# Instantly switch to Nike adapter
switcher.apply_style("nike", scale=0.9)

# Run generation
image_nike = pipeline(prompt="A running jacket, sportswear photoshoot").images[0]
```

### 3. Multi-LoRA Style Mixing

Compose multiple adapters to create hybrid brand collections:

```python
from src.lora.style_manager.style_mixer import StyleMixer

mixer = StyleMixer(pipeline)

# Mix Nike and Gucci aesthetics with custom weights
# 70% sporty, 30% luxury
mixer.apply_mix(
    adapters=["nike", "gucci"],
    weights=[0.7, 0.3]
)

# Run generation
mixed_image = pipeline(prompt="Hybrid luxury track jacket, studio fashion photography").images[0]
```

---

## 🏋️ Fine-Tuning Setup & Execution

Custom adapters are trained on dataset collections matching the schema defined in Week 1.

### 1. Preparing Training Metadata
Generate processed images and tag sidecars (matching `.json` or `.txt` captions) in the training folder:
```
training_dataset/
├── img1.png
├── img1.txt    --> "nike style sport tech jacket, highly detailed"
├── img2.png
└── img2.txt    --> "nike style performance trousers, white background"
```

### 2. Launching Training Scripts
We provide standalone training wrappers for each standard brand under `src/lora/trainers/`:

```bash
# Train Nike adapter
python src/lora/trainers/train_nike_lora.py --dataset_dir path/to/dataset --output_dir path/to/output

# Train Gucci adapter
python src/lora/trainers/train_gucci_lora.py --dataset_dir path/to/dataset --output_dir path/to/output
```

### Key Training Hyperparameters:
- **Base Model**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Resolution**: `1024`
- **Network Rank (Dimension)**: `16`
- **Network Alpha**: `32`
- **Learning Rate**: `1e-4` (U-Net) / `5e-5` (Text Encoders)
- **Batch Size**: `2` (per GPU)
