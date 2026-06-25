# Week 3 Technical Documentation: Controllable Fashion Generation

This document provides a deep technical dive into the architecture, workflows, and strategies implemented during Week 3 to establish a robust ControlNet-based fashion generation system.

---

## 1. ControlNet Architecture

Our system leverages the **ControlNet** architecture to augment a pre-trained Stable Diffusion model with spatial conditioning. 

*   **Mechanism:** ControlNet locks the original parameters of the Stable Diffusion model and creates a trainable copy of the encoding layers. The conditioning image (e.g., a sketch) is processed through these copied layers and added to the locked model's hidden states using "zero convolutions" ($1 \times 1$ convolutions initialized to zero). This ensures that the structural conditioning does not catastrophically forget the base model's pre-trained knowledge.
*   **Implementation:** We utilize the `diffusers` library, specifically `StableDiffusionControlNetPipeline`. The core engine (`FashionControlNetEngine`) handles model loading, memory management (e.g., CPU offloading, xFormers memory efficient attention if available), and tensor preparation.

## 2. Sketch-to-Design Workflow

The **Sketch-to-Design** pipeline (`sketch2design.py`) enables fashion designers to convert rough outlines into photorealistic garment renderings.

*   **Pre-processing:** Input images are routed through `utils/sketch_processor.py`. The processor normalizes the image, applies adaptive thresholding to isolate critical lines, and optionally dilates the lines to ensure the ControlNet can perceive them effectively.
*   **Conditioning:** The processed sketch is converted into a tensor and passed to a ControlNet trained on Canny edges or HED boundaries.
*   **Execution:** The `Sketch2Design` class orchestrates the generation, pairing the sketch with textual prompts from our domain-specific prompt library (e.g., `prompts/dresses.txt`). It utilizes negative prompting to avoid sketch-like artifacts in the final output (e.g., `negative_prompt="sketch, drawing, line art, monochrome"`).

## 3. Pose Control Workflow

The **Pose2Fashion** pipeline (`pose2fashion.py`) is designed for model rendering and lookbook generation, allowing garments to be visualized on specific human poses.

*   **Conditioning (OpenPose):** We utilize an OpenPose-based ControlNet. The input is an image containing an OpenPose skeleton (often generated via an auxiliary pose estimator).
*   **Workflow:** The pipeline maps the skeleton to the spatial canvas. Textual prompts guide the physical appearance of the model and the drape of the garment. This ensures that the generated fashion item accurately respects the anatomy and limb occlusion dictated by the input pose.

## 4. Depth Control Workflow

The **Depth2Fashion** pipeline (`depth2fashion.py`) is utilized for generating garments with complex 3D structures, volumetric drapery, or specific spatial relationships in a scene.

*   **Conditioning (MiDaS/Depth):** The input is a depth map, where pixel intensity correlates with spatial proximity to the camera.
*   **Workflow:** This workflow is highly effective for generating outerwear, layered clothing, and accessories. The ControlNet interprets the depth map to maintain realistic volume and lighting falloff on the generated fabrics.

## 5. Multi-Control Pipeline

To support advanced use cases, we engineered the `MultiControlPipeline` (`multi_control_pipeline.py`). This pipeline can accept lists of ControlNet models and conditioning images simultaneously.

*   **Weighted Controls:** We can blend conditioning modalities (e.g., `[Sketch, Pose]`) by assigning distinct scaling factors (`controlnet_conditioning_scale=[0.7, 0.4]`). This allows a designer to enforce a specific garment structure via sketch while loosely guiding the mannequin's stance via pose.

## 6. Fashion Sketch Dataset Design

To support future fine-tuning of custom fashion-specific ControlNets, we architected a standardized **Fashion Sketch Dataset** schema.

*   **Data Pairs:** The dataset requires thousands of high-quality pairs consisting of `(Target Image, Conditioning Image, Text Prompt)`.
*   **Target Image:** High-resolution, professional photographs of garments (sourced from datasets like DeepFashion or FashionGen).
*   **Conditioning Image:** Synthetically generated Canny edges, HED boundaries, or simulated hand-drawn sketches derived from the Target Images using specialized filters.
*   **Text Prompt:** Highly descriptive captions (e.g., "A macro shot of a red silk evening gown with asymmetrical ruffles, white background, studio lighting").

## 7. Fine-Tuning Strategy

If the out-of-the-box ControlNet models prove insufficient for specific fashion tasks, we have defined a fine-tuning strategy:

1.  **Dataset Preparation:** Utilize the aforementioned Fashion Sketch Dataset.
2.  **Model Initialization:** Initialize the ControlNet with weights from an existing model (e.g., `lllyasviel/sd-controlnet-canny`) to accelerate convergence.
3.  **Training Objectives:** Lock the base Stable Diffusion UNet. Train only the ControlNet copies and zero convolutions using a standard noise prediction objective.
4.  **Hyperparameters:** Utilize a low learning rate (e.g., $1e-5$) to prevent divergence, train with batch sizes optimized for available VRAM, and use gradient accumulation to simulate larger batches.
5.  **Validation:** Intermittently evaluate on a held-out set using our `comparison_engine.py` to track SSIM and Edge Preservation improvements over epochs.

---
*End of Technical Documentation*
