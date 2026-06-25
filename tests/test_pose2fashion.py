"""
week3/tests/test_pose2fashion.py
================================
Unit tests for Pose2Fashion orchestrator.
Validates loading, single-image generation, batch loops, skeleton checkers, and output saving.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.controlnet.controlnet.pose2fashion import Pose2Fashion
from src.controlnet.controlnet.controlnet_engine import GenerationOutput


# =============================================================================
# ── Fixtures & Helpers
# =============================================================================

@pytest.fixture
def mock_human_photo():
    """Create a 128x128 solid gray PIL image representing a human model photo."""
    return Image.new("RGB", (128, 128), color=(200, 200, 200))


@pytest.fixture
def orchestrator():
    """Return a Pose2Fashion instance forced into mock mode."""
    return Pose2Fashion(mock=True)


# =============================================================================
# ── Test Suite
# =============================================================================

class TestPose2Fashion:

    def test_init_propagates_mock_flag(self):
        p2f = Pose2Fashion(mock=True)
        assert p2f.mock is True
        assert p2f.engine.mock is True

    def test_preprocess_pose_generates_skeleton(self, orchestrator, mock_human_photo):
        pose_map = orchestrator.preprocess_pose(mock_human_photo)
        assert isinstance(pose_map, Image.Image)
        assert pose_map.size == (128, 128)
        assert pose_map.mode == "RGB"
        
        # Skeleton map should have a black background (mean luma close to 0)
        assert orchestrator._is_skeleton_image(pose_map) is True

    def test_is_skeleton_image_checks_non_skeleton(self, orchestrator, mock_human_photo):
        # A solid light-gray photo is not a skeleton map
        assert orchestrator._is_skeleton_image(mock_human_photo) is False

    def test_generate_fashion_mock(self, orchestrator, mock_human_photo):
        res = orchestrator.generate_fashion(
            pose_image=mock_human_photo,
            prompt="A blue denim jacket",
            style="casual",
            conditioning_scale=0.90,
            seed=77
        )
        assert isinstance(res, GenerationOutput)
        assert res.success is True
        assert res.control_type == "pose"
        assert res.seed == 77
        assert len(res.images) == 1
        assert res.images[0].size == (128, 128)
        
        # Verify metadata keys
        assert "preprocessor" in res.metadata
        assert res.metadata["preprocessor"]["method"] == "openpose"
        assert res.metadata["preprocessor"]["style_preset"] == "casual"
        assert "preprocessed_pose" in res.metadata
        assert isinstance(res.metadata["preprocessed_pose"], Image.Image)

    def test_generate_fashion_detects_pre_extracted_skeleton(self, orchestrator, mock_human_photo):
        # 1. Preprocess first to get a skeleton map
        skeleton = orchestrator.preprocess_pose(mock_human_photo)
        
        # 2. Feed the skeleton directly. It should bypass preprocess_pose loop
        # We verify by checking that it generates cleanly
        res = orchestrator.generate_fashion(
            pose_image=skeleton,
            prompt="A white cotton tee",
            style="casual"
        )
        assert res.success is True
        assert res.control_type == "pose"

    def test_generate_batch_mock(self, orchestrator, mock_human_photo):
        poses = [mock_human_photo, mock_human_photo]
        prompts = ["a velvet blazer", "a silk gown"]
        styles = ["formal", "luxury"]
        
        outputs = orchestrator.generate_batch(
            pose_images=poses,
            prompts=prompts,
            styles=styles,
            seed=500
        )
        
        assert len(outputs) == 2
        assert outputs[0].success is True
        assert outputs[0].seed == 500
        assert outputs[1].success is True
        assert outputs[1].seed == 501

    def test_save_results_creates_three_files(self, orchestrator, mock_human_photo):
        res = orchestrator.generate_fashion(
            pose_image=mock_human_photo,
            prompt="A summer linen dress",
            style="casual",
            seed=888
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepaths = orchestrator.save_results(res, output_dir=tmpdir)
            
            # Returns paths for: design image + preprocessed pose map
            assert len(filepaths) == 2
            
            # Check design image
            design_path = filepaths[0]
            assert design_path.exists()
            assert "pose_map" not in design_path.name
            
            # Check pose map image
            pose_map_path = filepaths[1]
            assert pose_map_path.exists()
            assert pose_map_path.name == f"{design_path.stem}_pose_map.png"
            
            # Find JSON metadata
            json_paths = list(Path(tmpdir).glob("*.json"))
            assert len(json_paths) == 1
            
            # Load and verify JSON contents
            with open(json_paths[0], "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["prompt"] == res.prompt
            assert meta["generation"]["seed"] == 888
            assert meta["preprocessor"]["method"] == "openpose"

    def test_save_results_empty_or_failed(self, orchestrator):
        failed_res = GenerationOutput(success=False, images=[])
        filepaths = orchestrator.save_results(failed_res)
        assert filepaths == []

    def test_generate_batch_padding_and_null_styles(self, orchestrator, mock_human_photo):
        poses = [mock_human_photo, mock_human_photo]
        prompts = ["a velvet blazer"] # Only 1 prompt provided
        
        outputs = orchestrator.generate_batch(
            pose_images=poses,
            prompts=prompts,
            styles=None, # null styles list
            seed=500
        )
        assert len(outputs) == 2
        assert outputs[1].prompt == "a velvet blazer"

    def test_save_results_save_exception(self, orchestrator, mock_human_photo):
        res = orchestrator.generate_fashion(
            pose_image=mock_human_photo,
            prompt="A summer linen dress",
            seed=888
        )
        # Mock preprocessed_pose save method to throw an exception
        from unittest.mock import MagicMock
        bad_img = MagicMock()
        bad_img.save.side_effect = Exception("Disk full")
        res.metadata["preprocessed_pose"] = bad_img
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepaths = orchestrator.save_results(res, output_dir=tmpdir)
            assert len(filepaths) == 1

    def test_is_skeleton_image_exception(self, orchestrator):
        # Pass something that isn't a PIL Image to raise an exception inside converter
        assert orchestrator._is_skeleton_image(None) is False

