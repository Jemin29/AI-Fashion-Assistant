"""
Week 6 Test Suite — Export Manager.
Verifies image formats conversion, metadata CSV/JSON logs, report formatting,
conversation history exporting, ZIP archiving, and health status checks.
"""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
import pytest
from PIL import Image

from week6.services.export_manager import ExportManager, _EXPORTS_DIR


@pytest.fixture
def manager() -> ExportManager:
    """Fixture returning ExportManager instance."""
    return ExportManager(mock_mode=False)


@pytest.fixture
def dummy_img() -> Image.Image:
    """Fixture returning a 100x100 white RGBA image."""
    return Image.new("RGBA", (100, 100), color=(255, 255, 255, 128))


def test_export_image_pil_to_png(manager, dummy_img) -> None:
    """Test exporting a PIL RGBA image to PNG."""
    res = manager.export_image(dummy_img, format_type="PNG", filename="test_pil.png")
    assert res.is_ok
    out_path = Path(res.data)
    assert out_path.exists()
    assert out_path.suffix == ".png"

    # Verify image integrity
    with Image.open(out_path) as img:
        assert img.size == (100, 100)
    
    # Cleanup
    out_path.unlink()


def test_export_image_pil_to_jpeg(manager, dummy_img) -> None:
    """Test converting and exporting PIL RGBA image to JPEG."""
    res = manager.export_image(dummy_img, format_type="JPEG", quality=90, filename="test_pil.jpg")
    assert res.is_ok
    out_path = Path(res.data)
    assert out_path.exists()
    assert out_path.suffix == ".jpg"

    # Verify image is RGB after conversion (no alpha in JPEG)
    with Image.open(out_path) as img:
        assert img.mode == "RGB"

    # Cleanup
    out_path.unlink()


def test_export_image_invalid_inputs(manager) -> None:
    """Test validation errors for formats and missing files."""
    # Invalid format
    res = manager.export_image("some_path.png", format_type="TIFF")
    assert not res.is_ok
    assert "format_type" in res.error

    # File path does not exist
    res = manager.export_image("nonexistent_design.png", format_type="PNG")
    assert not res.is_ok
    assert "exist" in res.error


def test_export_metadata_json_and_csv(manager) -> None:
    """Test exporting metadata dictionary to JSON and CSV formats."""
    meta = {
        "prompt": "streetwear oversized hoodie",
        "cfg": 8.5,
        "seed": 1002,
        "steps": 25,
    }

    # JSON
    res_json = manager.export_metadata(meta, format_type="JSON", filename="test_meta.json")
    assert res_json.is_ok
    json_path = Path(res_json.data)
    assert json_path.exists()
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["seed"] == 1002

    # CSV
    res_csv = manager.export_metadata(meta, format_type="CSV", filename="test_meta.csv")
    assert res_csv.is_ok
    csv_path = Path(res_csv.data)
    assert csv_path.exists()
    
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == ["Parameter", "Value"]
    assert ["seed", "1002"] in rows

    # Cleanup
    json_path.unlink()
    csv_path.unlink()


def test_export_recommendations(manager) -> None:
    """Test exporting style recommendations report to CSV, JSON, and MD Lookbook."""
    recs = [
        {"style": "Cyber Gorpcore", "confidence": 0.92, "season": "SS27"},
        {"style": "Minimalist Linen", "confidence": 0.88, "season": "SS27"}
    ]

    # JSON
    res = manager.export_recommendation_report(recs, format_type="JSON", filename="recs.json")
    assert res.is_ok
    p = Path(res.data)
    assert p.exists()
    p.unlink()

    # CSV
    res = manager.export_recommendation_report(recs, format_type="CSV", filename="recs.csv")
    assert res.is_ok
    p = Path(res.data)
    assert p.exists()
    with open(p, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert "confidence" in rows[0]
    p.unlink()

    # MD
    res = manager.export_recommendation_report(recs, format_type="MD", filename="recs.md")
    assert res.is_ok
    p = Path(res.data)
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "# 👗 AI Fashion Studio — Recommendation Report" in content
    assert "Cyber Gorpcore" in content
    p.unlink()


def test_export_chat_history(manager) -> None:
    """Test exporting chat conversation logs."""
    history = [
        ("What is linen fabric?", "Linen is a natural fiber woven from flax..."),
        ("Is it durable?", "Yes, linen is highly durable and breathable...")
    ]

    # MD Q&A report
    res = manager.export_chat_history(history, format_type="MD", filename="chat.md")
    assert res.is_ok
    p = Path(res.data)
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "🗣️ **USER**" in content
    assert "🤖 **ASSISTANT**" in content
    assert "natural fiber" in content
    p.unlink()

    # CSV Q&A report
    res = manager.export_chat_history(history, format_type="CSV", filename="chat.csv")
    assert res.is_ok
    p = Path(res.data)
    assert p.exists()
    p.unlink()


def test_export_batch_zip(manager, dummy_img) -> None:
    """Test packaging batch image, metadata, and report records into a ZIP."""
    # Write a temporary image and metadata first
    img_path = _EXPORTS_DIR / "temp_design.png"
    dummy_img.save(img_path)
    
    meta = {"prompt": "zipped design", "steps": 30}
    meta_path = _EXPORTS_DIR / "temp_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    records = [
        {"type": "image", "source": str(img_path), "filename": "concept.png"},
        {"type": "metadata", "source": meta, "filename": "meta.json"},
        {"type": "report", "source": "# Markdown Report content", "filename": "summary.md"},
    ]

    res = manager.export_batch_zip(records, archive_name="studio_export.zip")
    assert res.is_ok
    zip_path = Path(res.data)
    assert zip_path.exists()

    # Inspect ZIP file contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "concept.png" in namelist
        assert "meta.json" in namelist
        assert "summary.md" in namelist

        # Verify metadata is correct JSON inside zip
        meta_content = json.loads(zf.read("meta.json").decode("utf-8"))
        assert meta_content["steps"] == 30

    # Cleanup
    img_path.unlink()
    meta_path.unlink()
    zip_path.unlink()


def test_export_manager_health(manager) -> None:
    """Test health check probe."""
    check = manager.health_check()
    assert check["status"] == "ok"
    assert "readable/writable" in check["message"]
