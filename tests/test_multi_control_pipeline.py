"""
week3/tests/test_multi_control_pipeline.py
=========================================
Unit tests for the MultiControlFashionPipeline.
Validates loading, single/multi-conditioning, scale adjustments, and sidecar savings.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.controlnet.pipelines.multi_control_pipeline import MultiControlFashionPipeline
from src.controlnet.controlnet.controlnet_engine import GenerationOutput


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

@pytest.fixture
def mock_inputs():
    """Creates mock inputs for sketch, pose, and depth testing."""
    # Solid gray representations
    return {
        "sketch": Image.new("RGB", (128, 128), color=(220, 220, 220)),
        "pose": Image.new("RGB", (128, 128), color=(180, 180, 180)),
        "depth": Image.new("RGB", (128, 128), color=(150, 150, 150))
    }


@pytest.fixture
def pipeline():
    """Return a MultiControlFashionPipeline instance forced to mock mode."""
    return MultiControlFashionPipeline(mock=True)


# =============================================================================
# ── Test Suite
# =============================================================================

class TestMultiControlFashionPipeline:

    def test_init_propagates_mock_flag(self):
        pipe = MultiControlFashionPipeline(mock=True)
        assert pipe.mock is True
        assert pipe.engine.mock is True
        assert pipe.pose_orchestrator.mock is True
        assert pipe.depth_orchestrator.mock is True

    def test_raises_value_error_on_no_inputs(self, pipeline):
        with pytest.raises(ValueError, match="At least one conditioning image"):
            pipeline.generate(
                prompt="A casual wool sweater",
                sketch_image=None,
                pose_image=None,
                depth_image=None
            )

    def test_generate_single_condition_sketch_mock(self, pipeline, mock_inputs):
        res = pipeline.generate(
            prompt="A minimalist black silk dress",
            sketch_image=mock_inputs["sketch"],
            sketch_scale=0.75,
            style="minimalist",
            seed=101
        )
        
        assert isinstance(res, GenerationOutput)
        assert res.success is True
        assert res.seed == 101
        
        # Verify metadata details
        assert "preprocessor" in res.metadata
        preproc = res.metadata["preprocessor"]
        assert preproc["method"] == "multi_control"
        assert preproc["active_types"] == ["sketch"]
        assert preproc["conditioning_scales"] == [0.75]
        
        # Check preprocessed maps
        assert "preprocessed_sketch" in res.metadata
        assert isinstance(res.metadata["preprocessed_sketch"], Image.Image)

    def test_generate_full_multi_condition_mock(self, pipeline, mock_inputs):
        res = pipeline.generate(
            prompt="Luxury oversized puffer coat with gold trims",
            sketch_image=mock_inputs["sketch"],
            pose_image=mock_inputs["pose"],
            depth_image=mock_inputs["depth"],
            sketch_scale=0.45,
            pose_scale=0.85,
            depth_scale=0.25,
            style="luxury",
            seed=202
        )
        
        assert res.success is True
        assert res.seed == 202
        
        preproc = res.metadata["preprocessor"]
        assert preproc["active_types"] == ["sketch", "pose", "depth"]
        assert preproc["conditioning_scales"] == [0.45, 0.85, 0.25]
        
        # Check all preprocessed maps are populated
        assert "preprocessed_sketch" in res.metadata
        assert "preprocessed_pose" in res.metadata
        assert "preprocessed_depth" in res.metadata
        
        assert isinstance(res.metadata["preprocessed_sketch"], Image.Image)
        assert isinstance(res.metadata["preprocessed_pose"], Image.Image)
        assert isinstance(res.metadata["preprocessed_depth"], Image.Image)

    def test_save_results_creates_multiple_files(self, pipeline, mock_inputs):
        res = pipeline.generate(
            prompt="Relaxed fit cotton jeans",
            sketch_image=mock_inputs["sketch"],
            pose_image=mock_inputs["pose"],
            depth_image=mock_inputs["depth"],
            sketch_scale=0.5,
            pose_scale=0.7,
            depth_scale=0.4,
            seed=999
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepaths = pipeline.save_results(res, output_dir=tmpdir)
            
            # Outputs: 1 design image + 3 sidecar condition maps = 4 total image paths!
            assert len(filepaths) == 4
            
            # Check design image
            design_path = filepaths[0]
            assert design_path.exists()
            assert "sketch" not in design_path.name
            assert "pose" not in design_path.name
            assert "depth" not in design_path.name
            
            # Check sidecars
            sketch_path = filepaths[1]
            assert sketch_path.exists()
            assert sketch_path.name == f"{design_path.stem}_sketch_map.png"
            
            pose_path = filepaths[2]
            assert pose_path.exists()
            assert pose_path.name == f"{design_path.stem}_pose_map.png"
            
            depth_path = filepaths[3]
            assert depth_path.exists()
            assert depth_path.name == f"{design_path.stem}_depth_map.png"
            
            # Find the JSON metadata file
            json_paths = list(Path(tmpdir).glob("*.json"))
            assert len(json_paths) == 1
            
            # Load and verify JSON contents
            with open(json_paths[0], "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["prompt"] == res.prompt
            assert meta["generation"]["seed"] == 999
            
            # Preprocessor list
            active = meta["preprocessor"]["active_types"]
            assert active == ["sketch", "pose", "depth"]
            scales = meta["preprocessor"]["conditioning_scales"]
            assert scales == [0.5, 0.7, 0.4]

    def test_unload_models(self, pipeline):
        # Trigger model load first
        pipeline.engine.load_models("sketch")
        assert pipeline.engine._current_type == "sketch"
        
        # Trigger unload
        pipeline.engine.unload()
        assert pipeline.engine._current_type is None

    def test_save_results_empty_or_failed(self, pipeline):
        failed_res = GenerationOutput(success=False, images=[])
        filepaths = pipeline.save_results(failed_res)
        assert filepaths == []

    def test_config_device_auto_detection(self):
        # Create a mock config with device set to auto
        from unittest.mock import MagicMock
        mock_cfg = MagicMock()
        mock_cfg.model.runtime.device = "auto"
        mock_cfg.model.runtime.torch_dtype = "float32"
        
        pipe = MultiControlFashionPipeline(config=mock_cfg, mock=True)
        assert pipe.engine.device == "cpu"

    def test_generate_prompt_builder_exception(self, pipeline, mock_inputs):
        from unittest.mock import MagicMock
        bad_builder = MagicMock()
        bad_builder.build.side_effect = Exception("Failed to build prompt")
        pipeline.prompt_builder = bad_builder
        
        res = pipeline.generate(
            prompt="A sleek jacket",
            sketch_image=mock_inputs["sketch"],
            style="luxury"
        )
        assert res.success is True
        assert res.prompt == "A sleek jacket"


