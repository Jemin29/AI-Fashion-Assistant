"""
tests/test_controlnet_real_path.py
===================================
Integration tests verifying the real ControlNet SDXL generation path.
Automatically skips if no CUDA GPU is detected.
"""

import os
import sys
from pathlib import Path
import pytest
from PIL import Image
import torch

# Add repository root to system path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.controlnet.controlnet.controlnet_engine import FashionControlNetEngine


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Real ControlNet SDXL generation requires a CUDA GPU."
)
def test_real_controlnet_generation_canny():
    """
    Test real Canny ControlNet generation path with Stable Diffusion XL.
    Verifies that models are loaded and forward pass runs successfully on GPU.
    """
    engine = FashionControlNetEngine(mock=False)
    
    # Create simple binary edge image (simulated Canny edge map)
    control_image = Image.new("RGB", (512, 512), color=(0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(control_image)
    draw.rectangle([100, 100, 412, 412], outline=(255, 255, 255), width=3)
    
    # Execute generation with minimal steps for speed
    result = engine.generate(
        prompt="A sleek red silk dress, fashion catalog photography, studio background",
        control_image=control_image,
        mode="canny",
        conditioning_scale=0.7,
        num_inference_steps=5,  # Minimal steps for quick verification
        guidance_scale=5.0,
        negative_prompt="blurry, low quality",
        seed=42
    )
    
    assert result["success"] is True
    assert result["image"] is not None
    assert isinstance(result["image"], Image.Image)
    assert result["image"].size == (512, 512)
    
    # Quick visual content check: ensure it is not a flat black or flat color image
    import numpy as np
    img_arr = np.array(result["image"])
    assert np.std(img_arr) > 1.0, "Generated image is flat/empty"
