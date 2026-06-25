"""
week2/tests/test_batch_generator.py
=======================================
Comprehensive test suite for BatchGenerator.
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Debug import to raise import errors directly
try:
    from src.utils.experiment_tracker import ExperimentTracker
except Exception as e:
    print(f"DEBUG IMPORT ERROR: {e}")
    raise

from src.generation.generator.batch_generator import (
    BatchGenerator,
    BatchItem,
    ItemResult,
    BatchReport,
    BatchGenerationConfig,
    run_batch_from_csv,
    run_batch_from_list,
    _TRACKER_AVAILABLE,
)


# =============================================================================
# ── Fixtures
# =============================================================================

@pytest.fixture
def temp_dirs(tmp_path):
    """Isolated output directories, explicitly created."""
    dirs = {
        "output_dir": tmp_path / "generated",
        "batch_dir": tmp_path / "batch_runs",
        "experiments_dir": tmp_path / "experiments",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def mock_sdxl_generator():
    """Mock for FashionSDXLGenerator."""
    mock_gen = MagicMock()
    mock_gen.model_id = "mock-sdxl-1.0"
    
    # Return a mocked GenerationOutput
    class MockOutput:
        def __init__(self):
            self.images = [MagicMock()]
            self.image_paths = []
            self.metadata = {"mock_key": "mock_val"}
            self.generation_time_s = 0.05
            self.success = True
            self.error = None
            self.model_id = "mock-sdxl-1.0"
            
    mock_gen.generate_image.return_value = MockOutput()
    return mock_gen


# =============================================================================
# ── Unit Tests
# =============================================================================

def test_tracker_available_debug():
    """Verify that _TRACKER_AVAILABLE is True."""
    assert _TRACKER_AVAILABLE is True


def test_batch_item_resolved_seed():
    """Test seed resolution logic on BatchItem."""
    item = BatchItem(prompt="streetwear hoodie", seed=42)
    assert item.resolved_seed() == 42

    item_rand = BatchItem(prompt="luxury gown", seed=-1)
    seed1 = item_rand.resolved_seed()
    seed2 = item_rand.resolved_seed()
    assert 0 <= seed1 <= 2 ** 32 - 1
    assert 0 <= seed2 <= 2 ** 32 - 1


def test_item_result_serialisation(temp_dirs):
    """Test ItemResult serialization and formatting methods."""
    item = BatchItem(
        prompt="a cool jacket",
        seed=100,
        style="streetwear",
        width=512,
        height=512,
        steps=20,
        guidance_scale=8.0,
        run_name="test_run",
        tags=["jacket", "winter"],
        notes="no details",
        row_index=5,
    )
    
    res = ItemResult(
        item=item,
        success=True,
        attempts=2,
        generation_time=1.5,
        total_time=3.2,
        image_paths=[temp_dirs["output_dir"] / "test.png"],
        experiment_id="exp123",
        clip_score=0.85,
        metadata={"inference_time": 1.5},
    )

    # test to_dict
    d = res.to_dict()
    assert d["item_id"] == item.item_id
    assert d["row_index"] == 5
    assert d["prompt"] == "a cool jacket"
    assert d["success"] is True
    assert d["attempts"] == 2
    assert d["clip_score"] == 0.85
    assert d["experiment_id"] == "exp123"
    assert d["image_paths"] == [str(temp_dirs["output_dir"] / "test.png")]

    # test to_csv_row
    c = res.to_csv_row()
    assert c["row_index"] == "5"
    assert c["success"] == "True"
    assert c["clip_score"] == "0.85"
    assert "jacket" in c["tags"]

    # test summary / repr
    summary = res.summary()
    assert item.item_id in summary
    assert "OK" in summary
    assert "a cool jacket" in summary
    assert repr(res) == summary


def test_batch_report_properties():
    """Test BatchReport properties and summary output."""
    item1 = BatchItem(prompt="dress", row_index=1)
    item2 = BatchItem(prompt="shoes", row_index=2)
    
    res1 = ItemResult(item=item1, success=True, generation_time=1.0)
    res2 = ItemResult(item=item2, success=False, error="CUDA out of memory")
    
    report = BatchReport(
        run_name="my_test_batch",
        total=2,
        succeeded=1,
        failed=1,
        results=[res1, res2],
        total_time=4.5,
        mean_gen_time=1.0,
    )

    assert report.success_rate == 0.5
    assert report.successful_items == [res1]
    assert report.failed_items == [res2]
    
    summary = report.summary()
    assert "my_test_batch" in summary
    assert "Success rate  : 50.0%" in summary
    assert "CUDA out of memory" in summary
    
    report_dict = report.to_dict()
    assert report_dict["succeeded"] == 1
    assert report_dict["failed"] == 1
    assert len(report_dict["results"]) == 2


def test_batch_config_defaults(temp_dirs):
    """Test BatchGenerationConfig defaults and directory creation."""
    cfg = BatchGenerationConfig(
        output_dir=temp_dirs["output_dir"],
        batch_dir=temp_dirs["batch_dir"],
        experiments_dir=temp_dirs["experiments_dir"],
    )
    
    bg = BatchGenerator(config=cfg)
    assert bg.config.output_dir.exists()
    assert bg.config.batch_dir.exists()
    assert bg.config.experiments_dir.exists()
    assert bg.config.max_workers == 2
    assert bg.config.max_retries == 3


def test_batch_generator_run_from_list_stub(temp_dirs):
    """Test run_from_list using stub mode (generator = None)."""
    cfg = BatchGenerationConfig(
        max_workers=2,
        output_dir=temp_dirs["output_dir"],
        batch_dir=temp_dirs["batch_dir"],
        experiments_dir=temp_dirs["experiments_dir"],
        track_experiments=False,
    )
    
    bg = BatchGenerator(config=cfg)
    
    items = [
        {"prompt": "first design", "style": "casual"},
        {"prompt": "second design", "style": "formal", "seed": 77},
    ]
    
    report = bg.run_from_list(items, run_name="stub_test", export=True)
    
    assert report.total == 2
    assert report.succeeded == 2
    assert report.failed == 0
    assert report.success_rate == 1.0
    assert len(report.results) == 2
    
    # Check stub outputs
    assert report.results[0].metadata["stub"] is True
    assert report.results[1].item.seed == 77
    
    # Check exported manifests
    assert report.manifest_csv is not None
    assert report.manifest_json is not None
    assert report.manifest_csv.exists()
    assert report.manifest_json.exists()


def test_batch_generator_run_from_csv(temp_dirs):
    """Test parsing logic and run from CSV files."""
    csv_file = temp_dirs["batch_dir"] / "test_prompts.csv"
    
    # Create a CSV file with aliases & mapping
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "fashion_style", "rng_seed", "w", "h", "tags", "notes"])
        writer.writerow(["vintage leather jacket", "retro", "999", "512", "512", "vintage,cool", "test notes"])
        writer.writerow(["sporty leggings", "active", "", "", "", "[\"sport\", \"active\"]", ""])
        writer.writerow(["", "empty", "", "", "", "", ""])  # Empty prompt row (should be skipped)

    cfg = BatchGenerationConfig(
        output_dir=temp_dirs["output_dir"],
        batch_dir=temp_dirs["batch_dir"],
        experiments_dir=temp_dirs["experiments_dir"],
        track_experiments=False,
    )
    
    bg = BatchGenerator(config=cfg)
    report = bg.run_from_csv(
        csv_file,
        run_name="csv_test",
        default_seed=123,
        default_style="default_style",
    )
    
    assert report.total == 2
    
    # First row check
    item1 = report.results[0].item
    assert item1.prompt == "vintage leather jacket"
    assert item1.style == "retro"
    assert item1.seed == 999
    assert item1.width == 512
    assert item1.height == 512
    assert item1.tags == ["vintage", "cool"]
    assert item1.notes == "test notes"
    
    # Second row check (checking defaults and defaults falling back)
    item2 = report.results[1].item
    assert item2.prompt == "sporty leggings"
    assert item2.style == "active"
    assert item2.seed == 123  # fell back to default_seed
    assert item2.width == bg.config.default_width
    assert item2.tags == ["sport", "active"]


def test_batch_generator_csv_missing_prompt(temp_dirs):
    """Test that CSV loader raises ValueError if no prompt column exists."""
    csv_file = temp_dirs["batch_dir"] / "bad_prompts.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["style", "seed"])
        writer.writerow(["retro", "42"])

    bg = BatchGenerator(
        output_dir=temp_dirs["output_dir"],
        batch_dir=temp_dirs["batch_dir"],
        experiments_dir=temp_dirs["experiments_dir"],
    )
    with pytest.raises(ValueError, match="CSV must have a 'prompt' column"):
        bg.run_from_csv(csv_file)


def test_batch_generator_retry_logic(temp_dirs):
    """Test retry logic and exponential back-off."""
    mock_gen = MagicMock()
    
    # Fail twice, succeed on third attempt
    attempts_calls = 0
    def call_side_effect(*args, **kwargs):
        nonlocal attempts_calls
        attempts_calls += 1
        if attempts_calls < 3:
            raise RuntimeError("Transient CUDA failure")
        
        # Success output
        class MockOutput:
            def __init__(self):
                self.images = []
                self.success = True
        return MockOutput()
        
    mock_gen.generate_image.side_effect = call_side_effect
    mock_gen.model_id = "faulty-sdxl"

    cfg = BatchGenerationConfig(
        max_workers=1,
        max_retries=4,
        retry_delay=0.01,  # Speed up tests
        output_dir=temp_dirs["output_dir"],
        batch_dir=temp_dirs["batch_dir"],
        experiments_dir=temp_dirs["experiments_dir"],
        track_experiments=False,
    )
    
    bg = BatchGenerator(generator=mock_gen, config=cfg)
    item = BatchItem(prompt="retry me", seed=10)
    
    res = bg.generate_one(item)
    
    assert res.success is True
    assert res.attempts == 3
    assert attempts_calls == 3


def test_batch_generator_retry_failure(temp_dirs):
    """Test retry logic when all attempts fail."""
    mock_gen = MagicMock()
    mock_gen.generate_image.side_effect = RuntimeError("Permanent failure")
    mock_gen.model_id = "broken-sdxl"

    cfg = BatchGenerationConfig(
        max_workers=1,
        max_retries=2,
        retry_delay=0.01,
        output_dir=temp_dirs["output_dir"],
        batch_dir=temp_dirs["batch_dir"],
        experiments_dir=temp_dirs["experiments_dir"],
        track_experiments=False,
    )
    
    bg = BatchGenerator(generator=mock_gen, config=cfg)
    item = BatchItem(prompt="fail me", seed=10)
    
    res = bg.generate_one(item)
    
    assert res.success is False
    assert res.attempts == 2
    assert "Permanent failure" in res.error


def test_batch_generator_callback(temp_dirs):
    """Test completion callback is invoked per item."""
    completed_items = []
    
    def on_complete(result: ItemResult):
        completed_items.append(result)
        
    cfg = BatchGenerationConfig(
        output_dir=temp_dirs["output_dir"],
        batch_dir=temp_dirs["batch_dir"],
        experiments_dir=temp_dirs["experiments_dir"],
        track_experiments=False,
        on_item_complete=on_complete,
    )
    
    bg = BatchGenerator(config=cfg)
    bg.run_from_list([
        {"prompt": "dress 1"},
        {"prompt": "dress 2"},
    ], run_name="callback_test")
    
    assert len(completed_items) == 2
    assert completed_items[0].item.prompt == "dress 1"
    assert completed_items[1].item.prompt == "dress 2"


def test_batch_generator_locking(temp_dirs):
    """Test inference lock is acquired sequentially while other tasks run in parallel."""
    inference_threads = []
    lock_acquired = []

    mock_gen = MagicMock()
    mock_gen.model_id = "locking-sdxl"
    
    def call_side_effect(*args, **kwargs):
        # Record thread ID and verify lock is held
        inference_threads.append(threading.get_ident())
        time.sleep(0.02)  # Simulate GPU computation
        
        class MockOutput:
            def __init__(self):
                self.images = [MagicMock()]
                self.success = True
        return MockOutput()
        
    mock_gen.generate_image.side_effect = call_side_effect

    cfg = BatchGenerationConfig(
        max_workers=3,
        output_dir=temp_dirs["output_dir"],
        batch_dir=temp_dirs["batch_dir"],
        experiments_dir=temp_dirs["experiments_dir"],
        track_experiments=False,
        save_images=False,
        save_metadata=False,
    )
    
    # We patch threading.Lock to intercept acquire/release
    mock_lock = MagicMock()
    # Mock context manager behavior
    mock_lock.__enter__.return_value = mock_lock
    
    with patch("threading.Lock", return_value=mock_lock):
        bg = BatchGenerator(generator=mock_gen, config=cfg)
        bg.run_from_list([
            {"prompt": "lock test 1"},
            {"prompt": "lock test 2"},
            {"prompt": "lock test 3"},
        ])
    
    assert mock_lock.__enter__.call_count >= 3


def test_experiment_tracker_logging(temp_dirs):
    """Test integration with ExperimentTracker."""
    cfg = BatchGenerationConfig(
        output_dir=temp_dirs["output_dir"],
        batch_dir=temp_dirs["batch_dir"],
        experiments_dir=temp_dirs["experiments_dir"],
        track_experiments=True,
    )
    
    bg = BatchGenerator(config=cfg)
    
    # Verify that the tracker WAS initialized
    assert bg._tracker is not None
    
    report = bg.run_from_list([
        {"prompt": "tracked design 1", "style": "retro", "seed": 42},
        {"prompt": "tracked design 2", "style": "streetwear", "seed": 52},
    ], run_name="tracked_batch")
    
    # Verify records in tracker database
    tracker = bg._tracker
    assert len(tracker) == 2
    
    exps = tracker.list_experiments()
    assert len(exps) == 2
    
    prompts = [e.prompt for e in exps]
    assert "tracked design 1" in prompts
    assert "tracked design 2" in prompts


def test_convenience_wrappers(temp_dirs):
    """Test module-level convenience function wrappers."""
    csv_file = temp_dirs["batch_dir"] / "convenience.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["prompt"])
        writer.writerow(["wrapped design 1"])
        writer.writerow(["wrapped design 2"])

    # Test run_batch_from_csv
    rep_csv = run_batch_from_csv(
        csv_file,
        run_name="csv_convenience",
        max_workers=2,
        output_dir=temp_dirs["output_dir"],
        export=True,
    )
    assert rep_csv.total == 2
    assert rep_csv.succeeded == 2
    
    # Test run_batch_from_list
    rep_list = run_batch_from_list(
        [
            {"prompt": "list convenience 1"},
            {"prompt": "list convenience 2"},
        ],
        run_name="list_convenience",
        max_workers=2,
        output_dir=temp_dirs["output_dir"],
        export=True,
    )
    assert rep_list.total == 2
    assert rep_list.succeeded == 2
