"""
week3/controlnet/controlnet_trainer.py
======================================
ControlNet Fine-Tuning Framework for Fashion Sketches.
AI-Powered Fashion Design Assistant — Week 3.

Provides a unified trainer class using HuggingFace Diffusers and Accelerate
to fine-tune Stable Diffusion XL (SDXL) ControlNet on paired fashion design and sketch images.

=============================================================================
TRAINING INSTRUCTIONS
=============================================================================
Prerequisites:
  $ pip install torch torchvision diffusers transformers accelerate loguru pydantic pyyaml

Usage (Python API):
  ```python
  from datasets.fashion_sketch_dataset import FashionSketchDataset
  from src.controlnet.controlnet.controlnet_trainer import FashionControlNetTrainer
  from src.utils.config_manager import get_config

  # 1. Load config and datasets
  cfg = get_config()
  train_ds = FashionSketchDataset(
      manifest_path="datasets/processed/final_fashion_dataset.json",
      design_dir="datasets/processed",
      split="train"
  )
  val_ds = FashionSketchDataset(
      manifest_path="datasets/processed/final_fashion_dataset.json",
      design_dir="datasets/processed",
      split="val"
  )

  # 2. Initialize and run trainer
  trainer = FashionControlNetTrainer(
      config=cfg,
      train_dataset=train_ds,
      val_dataset=val_ds,
      batch_size=1,
      learning_rate=1e-5,
      num_train_epochs=5,
      mixed_precision="fp16",
      output_dir="week3/outputs/trainer"
  )
  trainer.train()
  ```

CLI Training via Accelerate (Multi-GPU/Distributed):
  1. Configure your environment:
     $ accelerate config
  2. Launch your script using the config:
     $ accelerate launch train_controlnet.py
"""

from __future__ import annotations

import os
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader

# ── Lazy imports to allow importing without full environment configured ────────
diffusers = None
transformers = None
accelerate = None


class MockControlNet(nn.Module):
    """A lightweight mock model representing SDXL-ControlNet for dry-run testing."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.linear = nn.Linear(10, 10)
        self.dummy_param = nn.Parameter(torch.randn(1))

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Dict[str, torch.Tensor],
        controlnet_cond: torch.Tensor,
        return_dict: bool = False
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        # Compute dynamic dummy output
        res = self.conv(controlnet_cond) + noisy_latents.mean(dim=1, keepdim=True)
        # Mimic SDXL down-block residual shapes: list of down block residuals
        # SDXL down blocks residuals are typically list of tensors
        down_residuals = [res * 0.1] * 9
        mid_residual = res * 0.05
        return (down_residuals, mid_residual) if not return_dict else {
            "down_block_res_samples": down_residuals,
            "mid_block_res_sample": mid_residual
        }


class FashionControlNetTrainer:
    """
    Orchestrates the fine-tuning of SDXL ControlNet on fashion sketch datasets.
    Wraps HuggingFace Diffusers models and manages training states, checkpointing,
    and validation generation.
    """

    def __init__(
        self,
        config: Any = None,
        train_dataset: Any = None,
        val_dataset: Any = None,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        num_train_epochs: int = 5,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "fp16",
        output_dir: Union[str, Path] = "week3/outputs/trainer",
        validation_steps: int = 100,
        checkpointing_steps: int = 500,
        max_train_steps: Optional[int] = None,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize the FashionControlNetTrainer.
        """
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision.lower()
        self.output_dir = Path(output_dir).resolve()
        self.validation_steps = validation_steps
        self.checkpointing_steps = checkpointing_steps
        self.max_train_steps = max_train_steps
        self.dry_run = dry_run

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_dependencies()

    def _load_dependencies(self) -> None:
        """Dynamically import Diffusers and Transformers libraries."""
        global diffusers, transformers, accelerate
        try:
            import diffusers as _diffusers
            import transformers as _transformers
            import accelerate as _accelerate
            
            diffusers = _diffusers
            transformers = _transformers
            accelerate = _accelerate
        except ImportError as exc:
            if not self.dry_run:
                raise ImportError(
                    f"Required libraries are missing. Run: pip install diffusers transformers accelerate. Details: {exc}"
                )
            else:
                logger.warning(f"Imports failed ({exc}), but dry_run is active. Proceeding with dummy imports.")

    def train(self) -> Dict[str, Any]:
        """
        Execute the main fine-tuning loop.
        """
        logger.info("Initializing ControlNet trainer framework...")
        t_start = time.perf_counter()

        # 1. Setup Accelerator
        from accelerate import Accelerator
        acc = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            project_dir=str(self.output_dir)
        )

        logger.info(f"Accelerator configured | device={acc.device} | mixed_precision={acc.mixed_precision}")

        # 2. Build or Load Models
        if self.dry_run:
            logger.info("Running in DRY-RUN mode. Instantiating mock layers...")
            controlnet = MockControlNet()
            optimizer = torch.optim.AdamW(controlnet.parameters(), lr=self.learning_rate)
            # Create mock pipeline helper structures
            noise_scheduler = None
            vae = None
            unet = None
        else:
            controlnet, optimizer, noise_scheduler, vae, unet = self._build_real_models()

        # 3. Create DataLoader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

        # 4. Prepare everything via Accelerator
        if self.dry_run:
            controlnet, optimizer, train_dataloader = acc.prepare(
                controlnet, optimizer, train_dataloader
            )
        else:
            controlnet, optimizer, train_dataloader, unet, vae = acc.prepare(
                controlnet, optimizer, train_dataloader, unet, vae
            )

        # 5. Training Stats & Schedulers
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
            self.max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        else:
            self.num_train_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)

        logger.info(
            f"Training Stats | epochs={self.num_train_epochs} | max_steps={self.max_train_steps} "
            f"| batch_size={self.batch_size} | total_train_samples={len(self.train_dataset)}"
        )

        global_step = 0
        total_loss = 0.0

        controlnet.train()

        # 6. Main Training Loop
        for epoch in range(self.num_train_epochs):
            logger.info(f"Starting Epoch [{epoch+1}/{self.num_train_epochs}]")
            epoch_loss = 0.0
            
            for step, batch in enumerate(train_dataloader):
                with acc.accumulate(controlnet):
                    if self.dry_run:
                        sketch = batch["conditioning_pixel_values"].to(acc.device)
                        B, C, H, W = sketch.shape
                        noisy_latents = torch.randn((B, 3, H, W), device=acc.device)
                        timesteps = torch.zeros(B, device=acc.device).long()
                        
                        down_res, mid_res = controlnet(
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            encoder_hidden_states=None,
                            added_cond_kwargs=None,
                            controlnet_cond=sketch
                        )
                        
                        # Calculate dummy MSE loss on mock parameters
                        loss = (down_res[0].mean() + mid_res.mean()) * 0.0 + controlnet.dummy_param.pow(2).mean()
                    else:
                        loss = self._compute_sdxl_loss(batch, controlnet, noise_scheduler, vae, unet, acc.device)

                    acc.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                # Track steps
                if acc.sync_gradients:
                    global_step += 1
                    total_loss += loss.item()
                    epoch_loss += loss.item()

                    # Checkpoint saving
                    if global_step % self.checkpointing_steps == 0:
                        self._save_checkpoint(acc, controlnet, global_step)

                    # Validation image generation
                    if global_step % self.validation_steps == 0:
                        self._run_validation(controlnet, global_step, acc.device)

                if global_step >= self.max_train_steps:
                    break

            avg_epoch_loss = epoch_loss / max(1, step)
            logger.info(f"Epoch {epoch+1} completed | Avg Loss: {avg_epoch_loss:.5f}")
            if global_step >= self.max_train_steps:
                logger.info("Reached maximum training steps. Exiting training loop.")
                break

        # 7. Save Final Model
        self._save_checkpoint(acc, controlnet, global_step, is_final=True)

        elapsed = round(time.perf_counter() - t_start, 2)
        logger.success(f"Fine-tuning complete | total_steps={global_step} | final_loss={total_loss/max(1, global_step):.5f} | elapsed_time={elapsed}s")

        return {
            "global_step": global_step,
            "average_loss": total_loss / max(1, global_step),
            "elapsed_time_s": elapsed,
            "output_dir": self.output_dir
        }

    # ── Auxiliary & Helper Methods ────────────────────────────────────────────

    def _collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch examples into stacked PyTorch tensors."""
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        prompts = [example["prompt"] for example in examples]
        
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "prompts": prompts
        }

    def _build_real_models(self) -> Tuple[Any, Any, Any, Any, Any]:
        """Loads and prepares real SDXL and ControlNet weights."""
        from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
        
        # Resolve config overrides or fallback to defaults
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        controlnet_id = "diffusers/controlnet-canny-sdxl-1.0"
        if self.config:
            if getattr(self.config, "model", None) and getattr(self.config.model, "base", None):
                base_model_id = self.config.model.base.repo_id or base_model_id
            if getattr(self.config, "controlnet", None) and getattr(self.config.controlnet, "model_id", None):
                controlnet_id = self.config.controlnet.model_id or controlnet_id

        logger.info(f"Loading SDXL base model: {base_model_id}...")
        logger.info(f"Loading ControlNet weights: {controlnet_id}...")

        # Load schedulers and sub-models
        noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
        controlnet = ControlNetModel.from_pretrained(controlnet_id)

        # Freeze unneeded model layers (ControlNet fine-tuning only trains controlnet)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        controlnet.requires_grad_(True)

        # Setup Optimizer
        optimizer = torch.optim.AdamW(
            controlnet.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )

        return controlnet, optimizer, noise_scheduler, vae, unet

    def _compute_sdxl_loss(
        self,
        batch: Dict[str, Any],
        controlnet: Any,
        noise_scheduler: Any,
        vae: Any,
        unet: Any,
        device: torch.device
    ) -> torch.Tensor:
        """Runs the forward pass through the diffusion models and returns loss."""
        # 1. Encode target design images to VAE latent space
        pixel_values = batch["pixel_values"].to(device)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # 2. Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        
        # 3. Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 4. Extract prompt condition embeddings
        # For simplicity in this unified implementation, we mimic text encoder calls.
        # Ideally, SDXL uses dual encoders; here we simulate text embedding tensor dimensions [B, seq_len, 2048]
        # and pooled projection embeddings [B, 1280] for SDXL inputs.
        B = latents.shape[0]
        prompt_embeds = torch.zeros((B, 77, 2048), device=device)
        pooled_prompt_embeds = torch.zeros((B, 1280), device=device)
        time_ids = torch.zeros((B, 6), device=device) # SDXL size details

        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": time_ids
        }

        # 5. Get ControlNet conditioning output
        sketch = batch["conditioning_pixel_values"].to(device)
        
        down_block_res, mid_block_res = controlnet(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            controlnet_cond=sketch,
            return_dict=False
        )

        # 6. Predict noise via UNet
        model_pred = unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_res,
            mid_block_additional_residual=mid_block_res,
            return_dict=False
        )[0]

        # 7. Compute Loss
        return torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

    def _save_checkpoint(self, accelerator: Any, controlnet: Any, step: int, is_final: bool = False) -> None:
        """Serializes ControlNet weights and writes state dictionaries to disk."""
        unwrapped_model = accelerator.unwrap_model(controlnet)
        
        if is_final:
            folder_name = "final_controlnet"
        else:
            folder_name = f"checkpoint-{step}"

        save_path = self.output_dir / folder_name
        save_path.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            # Save simple mock weights dict
            torch.save(unwrapped_model.state_dict(), save_path / "mock_model.bin")
            logger.info(f"Saved mock checkpoint directory: {save_path}")
        else:
            try:
                # Save standard HuggingFace ControlNet format
                unwrapped_model.save_pretrained(save_path)
                logger.info(f"Saved ControlNet weights checkpoint directory: {save_path}")
            except Exception as exc:
                logger.warning(f"Could not save pretrained weights: {exc}. Saving via torch.save fallback.")
                torch.save(unwrapped_model.state_dict(), save_path / "diffusion_pytorch_model.bin")

    def _run_validation(self, controlnet: Any, step: int, device: torch.device) -> None:
        """Executes a validation generation run to output design samples for visual auditing."""
        logger.info(f"Running validation generation audit at step {step}...")
        
        val_output_dir = self.output_dir / "validation"
        val_output_dir.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            # Generate a simple mock validation image (solid gray with text)
            time.sleep(0.1)
            img = Image.new("RGB", (256, 256), color=(100, 100, 105))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.text((128, 128), f"STEP {step}\nVALIDATION", fill=(255, 255, 255), anchor="mm")
            
            out_path = val_output_dir / f"step_{step}.png"
            img.save(out_path, format="PNG")
            logger.info(f"Validation mock image saved to: {out_path}")
        else:
            # Real validation generation requires loading full pipeline.
            # To conserve GPU memory, we wrap it in torch.no_grad and cpu-offloading if possible.
            if not self.val_dataset:
                logger.debug("Validation dataset missing. Skipping validation audit run.")
                return

            try:
                from diffusers import StableDiffusionXLControlNetPipeline
                
                # Fetch a sample from validation dataset
                sample = self.val_dataset[0]
                sketch_tensor = sample["conditioning_pixel_values"] # [3, H, W]
                prompt = sample["prompt"]

                # Convert tensor back to PIL
                import torchvision.transforms as T
                sketch_img = T.ToPILImage()(sketch_tensor)

                base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                if self.config and getattr(self.config, "model", None) and getattr(self.config.model, "base", None):
                    base_model_id = self.config.model.base.repo_id or base_model_id

                # Load pipeline temporarily
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    base_model_id,
                    controlnet=accelerator.unwrap_model(controlnet),
                    torch_dtype=torch.float16 if self.mixed_precision != "no" else torch.float32
                ).to(device)

                # Generate
                with torch.no_grad():
                    gen_img = pipe(
                        prompt=prompt,
                        image=sketch_img,
                        num_inference_steps=20
                    ).images[0]

                out_path = val_output_dir / f"step_{step}.png"
                gen_img.save(out_path, format="PNG")
                logger.success(f"Validation design generated and saved to: {out_path}")

                # Clean up to free memory
                del pipe
                torch.cuda.empty_cache()

            except Exception as err:
                logger.warning(f"Validation inference run encountered an error: {err}. Skipping visual audit.")
