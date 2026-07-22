import sys
import os
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Detect if torch is installed and if CUDA is available
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

from src.generation.generator.sdxl_generator import FashionSDXLGenerator

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU not detected")
def test_sdxl_real_generation_integration(tmp_path):
    print("CUDA GPU detected! Running real SDXL base-1.0 inference integration test...")
    
    # Initialize generator with real mode (global_mock=False) on cuda device
    generator = FashionSDXLGenerator(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda",
        torch_dtype="float16",
        output_dir=tmp_path / "outputs",
        global_mock=False,
    )
    
    # Load model
    generator.load_model()
    assert generator.is_loaded is True
    
    # Generate image using standard preset
    result = generator.generate_image(
        prompt="A minimalist black turtleneck sweater, professional fashion design studio shot",
        width=512,
        height=512,
        num_inference_steps=15,  # low steps for faster integration test
        seed=42,
    )
    
    # Validate result
    assert result.success is True, f"Generation failed: {result.error}"
    assert len(result.images) == 1
    
    img = result.images[0]
    assert img.width == 512
    assert img.height == 512
    
    # Convert image to numpy array to assert it's not a flat placeholder color
    arr = np.array(img)
    std_dev = np.std(arr)
    print(f"Generated image standard deviation: {std_dev}")
    
    # Flat color would have standard deviation of 0. Real generation should have substantial variance.
    assert std_dev > 5.0, "Generated image is a flat placeholder color!"
    
    # Save output
    saved_paths = generator.save_output(result)
    assert len(saved_paths) == 1
    assert saved_paths[0].exists()
    assert saved_paths[0].suffix == ".png"
