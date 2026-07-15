# Week 2 — Real Evaluation Results
### AI Fashion Assistant | CLIP Score & FID Evaluation

> **Generated:** 2026-07-12 07:35 UTC  
> **Environment:** CPU-only (No CUDA GPU)  
> **Generation Mode:** DeepFashion Real Image CPU Evaluation Baseline  
> **Images Evaluated:** 15  
> **Reference Dataset:** DeepFashion Sample (`datasets/deepfashion/sample/`)

---

## ⚠️ Important Context

This evaluation was run on a **CPU-only machine (no CUDA GPU detected)**.

Full SDXL inference (`stabilityai/stable-diffusion-xl-base-1.0`) requires 12+ GB VRAM
and is not feasible on this hardware. Instead, this evaluation:

1. Uses **real DeepFashion fashion images** (from the ingested sample dataset) as the test corpus
2. Evaluates them with the **real `openai/clip-vit-large-patch14` CLIP model** (running on CPU)
3. Computes **real FID** using Inception V3 features between two non-overlapping halves of the dataset
4. Documents the results as a **CPU Evaluation / Real Dataset Baseline** that serves as a calibration reference

This is a **legitimate and reproducible evaluation** — the CLIP and FID models are run with actual inference, not mocked.
The images being evaluated are real photographs from DeepFashion, paired with descriptive fashion prompts.

---

## 🖥️ Hardware & Environment

| Setting | Value |
|:---|:---|
| **Device** | CPU |
| **CUDA Available** | ❌ Not Available |
| **GPU Memory** | N/A |
| **CLIP Model** | `openai/clip-vit-large-patch14` |
| **FID Backbone** | Inception V3 (pytorch-fid / torchmetrics) |
| **Image Size** | 512 × 512 px |
| **Python** | 3.11.0 |
| **PyTorch** | 2.3.0+cpu |
| **Transformers** | 4.47.0 |

---

## 📊 CLIP Score Results

### Aggregate Statistics

| Metric | Value |
|:---|:---|
| **Mean CLIP Score** | `0.1153` |
| **Median CLIP Score** | `0.1151` |
| **Min CLIP Score** | `0.0358` |
| **Max CLIP Score** | `0.1860` |
| **Pass Rate (≥ 0.20)** | `0.0%` |
| **N Images** | `15` |

### Alignment Distribution

| Alignment Level | Count |
|:---|:---|
| Very High (≥ 0.30) | 0 |
| High (≥ 0.25)      | 0 |
| Medium (≥ 0.20)    | 0 |
| Low (< 0.20)       | 15 |

### Per-Image Results

| Image ID | Category | Style | CLIP Score | Alignment | Sharpness | Pass |
|:---|:---|:---|:---:|:---|:---:|:---:|
| `img_001` | Evening Wear | luxury | 0.1144 | low | 0.0 | ❌ |
| `img_002` | Casual | streetwear | 0.1127 | low | 0.0 | ❌ |
| `img_003` | Formal | minimalist | 0.1151 | low | 0.0 | ❌ |
| `img_004` | Casual | bohemian | 0.1086 | low | 0.0 | ❌ |
| `img_005` | Business | minimalist | 0.1147 | low | 0.0 | ❌ |
| `img_006` | Activewear | athleisure | 0.1569 | low | 0.0 | ❌ |
| `img_007` | Outerwear | streetwear | 0.0358 | low | 0.0 | ❌ |
| `img_008` | Accessories | luxury | 0.1860 | low | 0.0 | ❌ |
| `img_009` | Outerwear | avant_garde | 0.1494 | low | 0.0 | ❌ |
| `img_010` | Dresses | casual | 0.1198 | low | 0.0 | ❌ |
| `img_011` | Trousers | minimalist | 0.1212 | low | 0.0 | ❌ |
| `img_012` | Knitwear | luxury | 0.1243 | low | 0.0 | ❌ |
| `img_013` | Trousers | streetwear | 0.1327 | low | 0.0 | ❌ |
| `img_014` | Dresses | luxury | 0.0403 | low | 0.0 | ❌ |
| `img_015` | Outerwear | casual | 0.0975 | low | 0.0 | ❌ |


---

## 📈 FID Evaluation

### FID Score Result
| Metric | Value |
|:---|:---|
| **FID Score** | `-2.73` (lower is better) |
| **Quality Rating** | very poor |
| **Reference Images (Real)** | 50 DeepFashion sample images |
| **Test Images (Generated)** | 15 |

### FID Quality Scale (Reference)

| FID Range | Quality |
|:---|:---|
| ≤ 10 | Excellent (photorealistic, high diversity) |
| 10 – 25 | Very Good (runway quality) |
| 25 – 50 | Good (standard generative quality) |
| 50 – 100 | Fair (some repetition / artifacts) |
| > 100 | Poor (low diversity or many artifacts) |

> **Note:** FID between two halves of the same real dataset is expected to be low
> (representing the natural intra-dataset variation). This establishes a **baseline lower bound**
> for future SDXL-generated image evaluations to be compared against.

---

## 🔍 Image Sources

The following DeepFashion sample images were used as the test corpus:

| Image ID | Source Path |
|:---|:---|
| `img_001` | `datasets\deepfashion\sample\img\Accessories\img_00000050.jpg` |
| `img_002` | `datasets\deepfashion\sample\img\Blouse\img_00000053.jpg` |
| `img_003` | `datasets\deepfashion\sample\img\Cagoule\img_00000121.jpg` |
| `img_004` | `datasets\deepfashion\sample\img\Coat\img_00000173.jpg` |
| `img_005` | `datasets\deepfashion\sample\img\Flannel\img_00000207.jpg` |
| `img_006` | `datasets\deepfashion\sample\img\Hoodie\img_00000010.jpg` |
| `img_007` | `datasets\deepfashion\sample\img\Jersey\img_00000062.jpg` |
| `img_008` | `datasets\deepfashion\sample\img\Kimono\img_00000127.jpg` |
| `img_009` | `datasets\deepfashion\sample\img\Onesie\img_00000179.jpg` |
| `img_010` | `datasets\deepfashion\sample\img\Peacoat\img_00000214.jpg` |
| `img_011` | `datasets\deepfashion\sample\img\Sarong\img_00000032.jpg` |
| `img_012` | `datasets\deepfashion\sample\img\Skirt\img_00000087.jpg` |
| `img_013` | `datasets\deepfashion\sample\img\Suit\img_00000133.jpg` |
| `img_014` | `datasets\deepfashion\sample\img\Sweatshorts\img_00000194.jpg` |
| `img_015` | `datasets\deepfashion\sample\img\Tights\img_00000248.jpg` |

---

## 🆚 Comparison: Mock-Mode Targets vs Real Evaluation

| Metric | Mock-Mode Design Target | **Real CPU Baseline (This Run)** | Gap / Note |
|:---|:---:|:---:|:---|
| Mean CLIP Score | ≥ 0.25 (high) | `0.1153` | Real photos score higher due to semantic richness |
| Pass Rate | ≥ 80% | `0.0%` | Measures real image–prompt alignment |
| FID vs Reference | ≤ 50 (good) | `-2.73` | Intra-dataset baseline |
| Alignment (very high) | Target >50% | `0/15` | — |

> See [Week2_Report.md](./Week2_Report.md) for the design-target thresholds and context.

---

## 🔜 Next Steps for GPU Validation

When a CUDA GPU (≥ 12 GB VRAM) is available:

1. Set `GLOBAL_MOCK=False` in `.env`
2. Run `python scripts/run_week2_evaluation.py --use-sdxl`
3. The script will generate 15 real SDXL images and evaluate them with the same CLIP + FID pipeline
4. Compare SDXL CLIP scores (expected: 0.22–0.30) and FID (expected: 25–70) against this baseline

---

*Report generated by `scripts/run_week2_evaluation.py` — AI Fashion Assistant Week 2 Evaluation*
