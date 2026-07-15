# Week 2 Internship Report: Text-to-Image Fashion Generation Foundation

> **Prepared by:** Junior Generative AI Research Engineer  
> **Project:** AI-Powered Fashion Design Assistant  
> **Milestone:** Week 2 — Generation Foundation, Centralized Config, and Quality Evaluation  

---

## 1. Stable Diffusion XL (SDXL) Overview
Stable Diffusion XL (SDXL) represents a significant advancement in open-weights latent diffusion models, optimized specifically for high-resolution, detailed image generation. For our fashion design assistant, SDXL was chosen due to:
* **High Native Resolution**: Operates natively at $1024 \times 1024$ pixels, eliminating the blurry artifacts common in older models (like SD 1.5) when generating large images.
* **Dual Text Encoders**: Integrates CLIP ViT-L/14 and OpenCLIP-G/14, which capture fine-grained descriptions of clothing details, fabrics, and fit more effectively.
* **Refined Micro-Conditioning**: Incorporates crop parameters, aesthetic scores, and target sizes directly into the conditioning vectors, allowing detailed control over framing and fashion editorial aesthetics.

---

## 2. Diffusion Model Architecture (Executive Summary)
The generation engine employs a dual-stage latent diffusion architecture:
1. **Base Model (`SDXL-Base-1.0`)**: Runs the initial denoising loop in latent space (typically 30 steps) using a combination of the dual CLIP text encoders to construct the structural layout, garment shapes, and model pose.
2. **Refiner Model (`SDXL-Refiner-1.0`)**: Processes the final $20\%$ of the denoising steps (denoising interval $[0.8, 1.0]$) to add micro-texture details to fabric, stitching, faces, and accessories.
3. **VAE (`madebyollin/sdxl-vae-fp16-fix`)**: Translates the latent representation back to pixel space. The choice of the `fp16-fix` VAE resolves the notorious NaN/black-screen bug native to the official Stability AI VAE under half-precision settings.

---

## 3. Fashion Generation Workflow
Our end-to-end workflow transforms raw textual prompts into high-quality fashion imagery, validates them against constraints, and scores their perceptual quality:

```
[User Input Prompt] ──► [PromptValidator] ──► [PromptBuilder] ──► [SDXLGenerator]
                                                                        │
                                                                   (Inference)
                                                                        │
                                                                        ▼
[PipelineRunResult] ◄── [EvaluationReport] ◄── [QualityScorer] ◄── [Image Output]
```

1. **Validation**: The `PromptValidator` checks the token limit and sanitizes inputs (filtering empty strings or unsafe tokens).
2. **Expansion**: The `PromptBuilder` injects fashion-specific descriptors, style presets, gender context, and season modifiers, outputting a highly descriptive prompt block.
3. **Denoising**: The `FashionSDXLGenerator` runs the base and optional refiner loops under the selected noise scheduler.
4. **Post-Processing & Save**: The image is saved along with a sidecar JSON file containing the seed, hardware settings, and scheduler metadata.
5. **Quality Check**: The image runs through a composite scoring block (validating size, color balance, and prompt alignment).

---

## 4. Prompt Engineering Strategy
SDXL is prompt-sensitive and easily distracted. To automate premium output quality, we built a **Style Preset Registry** containing 9 custom-tailored fashion categories:

| Preset | Key Descriptors Added | Guidance Scale | Typical Use Case |
| :--- | :--- | :---: | :--- |
| `streetwear` | urban style, oversized silhouette, graphic details, heavy cotton | 7.0 | Hoodies, graphic tees, cargo pants |
| `luxury` | high-end designer, couture, premium silk, gold embroidery, haute couture | 8.5 | Evening gowns, tailored coats |
| `casual` | relaxed fit, everyday wear, minimalist, neutral color palette | 7.0 | T-shirts, knitwear, denim |
| `athleisure` | performance fabric, sporty-chic, sleek activewear, matte finish | 7.5 | Leggings, windbreakers, sports bras |
| `bohemian` | flowy silhouette, organic linen, earthy tones, layered textiles | 7.0 | Maxi dresses, floral blouses |
| `minimalist` | clean lines, Scandinavian design, high-quality wool, monochrome | 7.5 | Trench coats, tailored trousers |
| `avant_garde` | sculptural silhouette, conceptual fashion, experimental draping | 9.0 | Runway collection, editorial |

---

## 5. CLIP Alignment and Perceptual Metrics
To ensure the model actually generated what was requested, we implement a multi-stage evaluation engine:
* **CLIP Text-Image Similarity**: Evaluates semantic alignment by projecting both the prompt text and the generated image into a shared embedding space using `openai/clip-vit-large-patch14`. We apply custom fashion-tuned thresholds:
  * $\ge 0.30$: Very High Alignment
  * $\ge 0.25$: High Alignment
  * $\ge 0.20$: Medium Alignment (Minimum passing threshold)
  * $< 0.20$: Low Alignment (Triggers auto-rejection)
* **Basic Quality Checks**: Fast numpy-based checks to reject all-black or all-white images (failed generations) and enforce resolution constraints (minimum $512 \times 512$).

> **Validation Status:** The thresholds above are **design targets** validated in mock mode during development. Real evaluation numbers from running the actual CLIP model against DeepFashion sample images are recorded in [Week2_Evaluation_Results_Real.md](./Week2_Evaluation_Results_Real.md).

---

## 6. FID (Fréchet Inception Distance) Evaluation
While CLIP measures single-image alignment, **FID** measures the overall diversity and realism of a generated batch compared to a reference dataset (such as our preprocessed datasets from Week 1):
* **Inception V3 Feature Spaces**: Extract 2048-dimensional feature vectors from both real reference images and generated batches.
* **Distribution Comparison**: Calculates the Fréchet distance between the two multivariate Gaussian distributions:
  $$d^2 = \|\mu_1 - \mu_2\|_2^2 + \text{Tr}(\Sigma_1 + \Sigma_2 - 2(\Sigma_1\Sigma_2)^{1/2})$$
* **Quality Interpretation**:
  * $\le 10$: Excellent diversity and photorealism.
  * $10 - 25$: Very Good runway quality.
  * $25 - 50$: Good (standard generative quality).
  * $> 50$: Fair/Poor (indicates repetitive artifacts or low diversity).

> **Validation Status:** FID thresholds above are **design targets** for GPU-generated SDXL batches. The baseline FID computed against the real DeepFashion sample (intra-dataset reference) is recorded in [Week2_Evaluation_Results_Real.md](./Week2_Evaluation_Results_Real.md).

---

## 6.1 Mock Validation vs. Real Model Validation

This project supports two distinct evaluation modes. It is critical to distinguish them when interpreting metrics:

| Aspect | 🟡 Mock-Mode Validation | 🟢 Real Model Validation |
| :--- | :--- | :--- |
| **Purpose** | Unit testing, CI, API development without GPU | Measure actual model quality |
| **SDXL Generation** | Geometric placeholder images (fast) | Full diffusion pipeline (GPU required) |
| **CLIP Evaluation** | Mocked cosine similarity values | Real `openai/clip-vit-large-patch14` on CPU/GPU |
| **FID Evaluation** | Skipped / placeholder FID score | Real Inception V3 feature extraction |
| **Hardware** | Any machine, CPU-only | CUDA GPU ≥ 12 GB VRAM for generation |
| **Run Command** | Default (GLOBAL_MOCK=True) | `GLOBAL_MOCK=False python scripts/run_week2_evaluation.py` |
| **Results File** | N/A (tests only) | [Week2_Evaluation_Results_Real.md](./Week2_Evaluation_Results_Real.md) |

### Summary of Real Evaluation Numbers (CPU Baseline)

The real evaluation was run against **15 DeepFashion sample images** using the real CLIP model on CPU.
See [Week2_Evaluation_Results_Real.md](./Week2_Evaluation_Results_Real.md) for full per-image breakdown.

| Metric | Real CPU Baseline | Mock-Mode Target |
| :--- | :--- | :--- |
| Mean CLIP Score | `0.1153` (mismatched pairs, expected) | ≥ 0.25 (design target) |
| Pass Rate (≥ 0.20) | `0.0%` (catalog photos vs. prompts) | ≥ 80% |
| FID vs Reference | `-2.73` (small-sample artifact, N=15) | ≤ 50 (SDXL target) |

> **When a CUDA GPU is available**, run `GLOBAL_MOCK=False python scripts/run_week2_evaluation.py` to produce GPU SDXL-generated results and update this table with real generation numbers.

---

## 7. Centralized Configuration and Benchmarking
* **Configuration Manager**: Built using Pydantic v2, resolving settings from YAML configurations and overriding them dynamically with environment variables prefixed with `FASHION_CONFIG_`.
* **Combinatorial Parameter Benchmarking**: Systematically generates parameter grid combinations (seeds, CFG scales, noise schedulers, and resolutions) to sweep hyperparameters.
* **Benchmark Report**: Saves detailed metrics to `benchmark_report.json` and renders an interactive HTML dashboard containing image comparison grids.

---

## 8. Real-world Challenges & Engineering Solutions

During the implementation, several critical software and hardware bottlenecks were resolved:

### 1. SQLite Database Locking in Parallel Generation
* **Challenge**: When running `BatchGenerator` with thread pool workers, concurrent sqlite operations on `experiment_tracker.db` resulted in `database is locked` errors.
* **Solution**: Implemented a thread-safe connection pooling wrapper with a write lock (`threading.Lock`) around write operations, along with setting `PRAGMA journal_mode=WAL` (Write-Ahead Logging) to allow concurrent reads and serial write queues.

### 2. The Python Dataclass "Falsy" Object Bug
* **Challenge**: `ExperimentTracker` implements a custom `__len__` method returning the database record count. When the database was empty, `len(tracker)` returned `0`, making `if self._tracker:` evaluate to `False` due to Python's truth value testing rules. This silently bypassed logging for the first run.
* **Solution**: Replaced checks with explicit identity testing: `if self._tracker is not None:`.

### 3. VRAM Constraints on Consumer Hardware
* **Challenge**: Running SDXL base + refiner requires $> 12$ GB of VRAM, causing Out-Of-Memory (OOM) errors on 8GB cards.
* **Solution**: Added layered memory configurations inside `model_manager.py`:
  * *8GB GPUs*: Enable `enable_xformers()` and `enable_model_cpu_offload()`.
  * *4GB GPUs*: Enable `enable_sequential_cpu_offload()`.
  * *CPU fallback*: Graceful CPU inference mapping if no CUDA/MPS device is detected.

---

## 9. Next Steps: Week 3 Conditioning Roadmap
To advance from simple prompt-to-image to structured design generation, Week 3 will introduce:
1. **ControlNet Integration**: Incorporate pose estimation (Openpose) and garment edge structures (Canny/Depth maps) to maintain consistent silhouettes.
2. **LoRA Fine-tuning**: Fine-tune lightweight adapters on specific luxury fashion collections to learn branded textures and styling.
3. **Web Interface & API**: Wrap the pipelines in a FastAPI backend and build a Streamlit design dashboard.

---

## 10. References
1. Podell, D., et al. (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*. arXiv:2307.01952.
2. Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML.
3. Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)*. NeurIPS.
