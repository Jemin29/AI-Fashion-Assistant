"""
week5/tests/test_portfolio_generator.py
======================================
Unit tests for the Portfolio Generator.
Verifies clean execution of portfolio asset creation, PIL card rendering, and JSON/HTML outputs.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path
import tempfile
import pytest

from PIL import Image
import portfolio_generator


def test_get_font():
    """Verify get_font returns a valid font descriptor without raising exceptions."""
    font = portfolio_generator.get_font(size=14, bold=False)
    assert font is not None
    
    font_bold = portfolio_generator.get_font(size=18, bold=True)
    assert font_bold is not None


def test_draw_wrapped_text():
    """Test text wrapping helper handles normal text and long tokens."""
    img = Image.new("RGB", (300, 200))
    draw = portfolio_generator.ImageDraw.Draw(img)
    font = portfolio_generator.get_font()
    
    y = portfolio_generator.draw_wrapped_text(
        draw=draw,
        text="This is a test of the text wrapping utility within the portfolio card drawer.",
        x=10,
        y=10,
        max_width=100,
        font=font,
        fill_color=(255, 255, 255)
    )
    assert y > 10


def test_draw_scenario_card():
    """Verify draw_scenario_card successfully renders a styled PNG image."""
    metrics = {
        "latency": 0.045,
        "hit_rate": 1.0,
        "grounding": 0.95,
        "relevance": 0.85,
        "confidence": 0.90
    }
    
    # Render style recommendation card
    img = portfolio_generator.draw_scenario_card(
        title="Style Recommendation Engine",
        query="streetwear in black color",
        response=" street wear is a style of casual clothing.",
        metrics=metrics,
        theme_color1=(236, 72, 153),
        theme_color2=(244, 114, 182)
    )
    
    assert img.size == (900, 550)


def test_data_extraction_helpers():
    """Test data extraction helpers parse various response types safely."""
    # Dict response
    res_dict = {
        "response": "Advice text",
        "data": {
            "retrieved_items": [{"id": "doc1", "content": "hello"}],
            "styles": ["streetwear"]
        }
    }
    assert portfolio_generator.get_retrieved_items(res_dict) == [{"id": "doc1", "content": "hello"}]
    assert portfolio_generator.get_recommendations(res_dict) == ["streetwear"]

    # List response inside data
    res_list = {
        "response": " advice text",
        "data": [{"id": "trend_1", "name": "Minimalism"}]
    }
    assert portfolio_generator.get_retrieved_items(res_list) == [{"id": "trend_1", "name": "Minimalism"}]
    assert portfolio_generator.get_recommendations(res_list) == ["Minimalism"]

    # Missing keys response
    res_empty = {"response": "empty"}
    assert portfolio_generator.get_retrieved_items(res_empty) == []
    assert portfolio_generator.get_recommendations(res_empty) == []


def test_portfolio_generator_subprocess():
    """Verify executing portfolio_generator.py in a subprocess creates all assets and outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Run portfolio generator pointing to the temp directory
        cmd = [
            sys.executable,
            "portfolio_generator.py",
            "--output-dir", str(tmp_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"
        
        # Check that evaluation_summary.json exists and is valid
        summary_file = tmp_path / "evaluation_summary.json"
        assert summary_file.exists()
        with open(summary_file, "r") as f:
            data = json.load(f)
        
        assert "scenarios" in data
        assert "summary" in data
        assert len(data["scenarios"]) == 5
        
        # Check that HTML dashboard exists
        dashboard_file = tmp_path / "index.html"
        assert dashboard_file.exists()
        
        # Check that PNG screenshots exist under images/
        images_dir = tmp_path / "images"
        assert images_dir.exists()
        
        expected_images = [
            "style_recommendation_card.png",
            "brand_recommendation_card.png",
            "trend_forecasting_card.png",
            "semantic_search_card.png",
            "fashion_qa_card.png"
        ]
        for img_name in expected_images:
            img_file = images_dir / img_name
            assert img_file.exists(), f"Missing image asset: {img_name}"
            # Check image can be opened
            with Image.open(img_file) as img:
                assert img.size == (900, 550)


def test_main_cli_argument_mocking(monkeypatch):
    """Test calling portfolio_generator.main() directly by mocking CLI arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Mock sys.argv to mock arguments parsed by argparse
        monkeypatch.setattr(sys, "argv", ["portfolio_generator.py", "--output-dir", str(tmp_path)])
        
        exit_code = portfolio_generator.main()
        assert exit_code == 0
        
        # Check index.html exists
        assert (tmp_path / "index.html").exists()
        assert (tmp_path / "evaluation_summary.json").exists()
