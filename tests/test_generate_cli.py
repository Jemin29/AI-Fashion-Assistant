"""
week2/tests/test_generate_cli.py
==================================
Tests for the week2/generate.py CLI entry point.

All tests are mock-based — no model loading or GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.generation.generate import (
    build_parser,
    build_prompts,
    print_sizes,
    print_schedulers,
    print_styles,
)
from src.generation.generator.sdxl_generator import SIZE_PRESETS, SCHEDULER_MAP


# =============================================================================
# ── Parser Tests
# =============================================================================

class TestParser:

    @pytest.fixture
    def parser(self):
        return build_parser()

    def test_default_prompt_is_none(self, parser):
        args = parser.parse_args([])
        assert args.prompt is None

    def test_prompt_flag(self, parser):
        args = parser.parse_args(["--prompt", "a red dress"])
        assert args.prompt == "a red dress"

    def test_short_prompt_flag(self, parser):
        args = parser.parse_args(["-p", "a blue coat"])
        assert args.prompt == "a blue coat"

    def test_default_steps(self, parser):
        args = parser.parse_args([])
        assert args.steps == 30

    def test_steps_override(self, parser):
        args = parser.parse_args(["--steps", "50"])
        assert args.steps == 50

    def test_default_guidance(self, parser):
        args = parser.parse_args([])
        assert args.guidance == 7.5

    def test_guidance_override(self, parser):
        args = parser.parse_args(["--guidance", "9.0"])
        assert args.guidance == 9.0

    def test_default_seed(self, parser):
        args = parser.parse_args([])
        assert args.seed == -1

    def test_seed_override(self, parser):
        args = parser.parse_args(["--seed", "42"])
        assert args.seed == 42

    def test_default_num_images(self, parser):
        args = parser.parse_args([])
        assert args.num_images == 1

    def test_num_images_override(self, parser):
        args = parser.parse_args(["--n", "4"])
        assert args.num_images == 4

    def test_default_size(self, parser):
        args = parser.parse_args([])
        assert args.size == "square_1024"

    def test_size_override(self, parser):
        args = parser.parse_args(["--size", "portrait_1024"])
        assert args.size == "portrait_1024"

    def test_invalid_size_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--size", "invalid_size_xyz"])

    def test_default_format(self, parser):
        args = parser.parse_args([])
        assert args.format == "png"

    def test_format_jpg(self, parser):
        args = parser.parse_args(["--format", "jpg"])
        assert args.format == "jpg"

    def test_default_device(self, parser):
        args = parser.parse_args([])
        assert args.device == "auto"

    def test_device_cpu(self, parser):
        args = parser.parse_args(["--device", "cpu"])
        assert args.device == "cpu"

    def test_default_dtype(self, parser):
        args = parser.parse_args([])
        assert args.dtype == "float16"

    def test_refiner_false_by_default(self, parser):
        args = parser.parse_args([])
        assert args.refiner is False

    def test_refiner_flag(self, parser):
        args = parser.parse_args(["--refiner"])
        assert args.refiner is True

    def test_low_vram_flag(self, parser):
        args = parser.parse_args(["--low-vram"])
        assert args.low_vram is True

    def test_sequential_offload_flag(self, parser):
        args = parser.parse_args(["--sequential-offload"])
        assert args.sequential_offload is True

    def test_list_sizes_flag(self, parser):
        args = parser.parse_args(["--list-sizes"])
        assert args.list_sizes is True

    def test_list_schedulers_flag(self, parser):
        args = parser.parse_args(["--list-schedulers"])
        assert args.list_schedulers is True

    def test_list_styles_flag(self, parser):
        args = parser.parse_args(["--list-styles"])
        assert args.list_styles is True

    def test_info_flag(self, parser):
        args = parser.parse_args(["--info"])
        assert args.info is True

    def test_no_metadata_flag(self, parser):
        args = parser.parse_args(["--no-metadata"])
        assert args.no_metadata is True

    def test_prefix_default(self, parser):
        args = parser.parse_args([])
        assert args.prefix == "fashion"

    def test_prefix_override(self, parser):
        args = parser.parse_args(["--prefix", "luxury"])
        assert args.prefix == "luxury"

    def test_default_scheduler(self, parser):
        args = parser.parse_args([])
        assert args.scheduler == "euler"

    def test_invalid_scheduler_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--scheduler", "bad_scheduler"])


# =============================================================================
# ── Print / Info Helpers
# =============================================================================

class TestInfoHelpers:

    def test_print_sizes_does_not_crash(self, capsys):
        print_sizes()
        captured = capsys.readouterr()
        assert "square_1024" in captured.out

    def test_print_sizes_lists_all_presets(self, capsys):
        print_sizes()
        captured = capsys.readouterr()
        for name in SIZE_PRESETS:
            assert name in captured.out

    def test_print_schedulers_does_not_crash(self, capsys):
        print_schedulers()
        captured = capsys.readouterr()
        assert "euler" in captured.out

    def test_print_schedulers_lists_all(self, capsys):
        print_schedulers()
        captured = capsys.readouterr()
        for name in SCHEDULER_MAP:
            assert name in captured.out

    def test_print_styles_does_not_crash(self, capsys):
        print_styles()
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# =============================================================================
# ── build_prompts()
# =============================================================================

class TestBuildPrompts:

    def _args(self, **kwargs):
        """Namespace-like object."""
        parser = build_parser()
        defaults = parser.parse_args([])
        for k, v in kwargs.items():
            setattr(defaults, k, v)
        return defaults

    def test_single_prompt(self):
        args = self._args(prompt="a red dress", batch_file=None, style=None, negative=None)
        pairs = build_prompts(args)
        assert len(pairs) == 1
        assert pairs[0][0] == "a red dress"

    def test_negative_auto_built_when_none(self):
        args = self._args(prompt="a dress", batch_file=None, style=None, negative=None)
        pairs = build_prompts(args)
        neg = pairs[0][1]
        assert "blurry" in neg    # from get_fashion_negative()

    def test_explicit_negative_used(self):
        args = self._args(prompt="a dress", batch_file=None, style=None, negative="ugly")
        pairs = build_prompts(args)
        assert pairs[0][1] == "ugly"

    def test_style_tags_appended_to_positive(self):
        args = self._args(prompt="a shirt", batch_file=None, style="streetwear", negative=None)
        pairs = build_prompts(args)
        # Should have streetwear tags appended
        assert len(pairs[0][0]) > len("a shirt")
        assert "a shirt" in pairs[0][0]

    def test_unknown_style_returns_prompt_unchanged(self):
        args = self._args(prompt="a coat", batch_file=None, style=None, negative=None)
        pairs = build_prompts(args)
        assert pairs[0][0] == "a coat"

    def test_no_prompt_no_file_exits(self):
        args = self._args(prompt=None, batch_file=None, style=None, negative=None)
        with pytest.raises(SystemExit):
            build_prompts(args)

    def test_batch_file_nonexistent_exits(self, tmp_path):
        args = self._args(
            prompt=None,
            batch_file=tmp_path / "nonexistent.txt",
            style=None, negative=None,
        )
        with pytest.raises(SystemExit):
            build_prompts(args)

    def test_batch_file_loaded(self, tmp_path):
        f = tmp_path / "prompts.txt"
        f.write_text("a red dress\na blue hoodie\n# comment line\n\na white shirt\n")
        args = self._args(prompt=None, batch_file=f, style=None, negative=None)
        pairs = build_prompts(args)
        # Comment and blank lines stripped
        assert len(pairs) == 3
        prompts = [p for p, _ in pairs]
        assert "a red dress" in prompts
        assert "a blue hoodie" in prompts
        assert "a white shirt" in prompts

    def test_batch_file_and_prompt_combined(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("a dress\n")
        args = self._args(prompt="a suit", batch_file=f, style=None, negative=None)
        pairs = build_prompts(args)
        prompts = [p for p, _ in pairs]
        assert "a dress" in prompts
        assert "a suit" in prompts


# =============================================================================
# ── main() Integration Tests (mocked)
# =============================================================================

class TestMain:

    def test_list_sizes_exits_zero(self):
        from src.generation.generate import main
        with patch("sys.argv", ["generate.py", "--list-sizes"]):
            code = main()
        assert code == 0

    def test_list_schedulers_exits_zero(self):
        from src.generation.generate import main
        with patch("sys.argv", ["generate.py", "--list-schedulers"]):
            code = main()
        assert code == 0

    def test_list_styles_exits_zero(self):
        from src.generation.generate import main
        with patch("sys.argv", ["generate.py", "--list-styles"]):
            code = main()
        assert code == 0

    def test_info_exits_zero(self, tmp_path):
        from src.generation.generate import main
        with patch("sys.argv", [
            "generate.py", "--info", "--output-dir", str(tmp_path),
        ]):
            code = main()
        assert code == 0

    def test_successful_generation_returns_zero(self, tmp_path):
        from src.generation.generate import main
        from src.generation.generator.sdxl_generator import GenerationOutput
        try:
            from PIL import Image
            fake_img = Image.new("RGB", (64, 64), (0, 0, 0))
        except ImportError:
            pytest.skip("Pillow not installed")

        mock_output = GenerationOutput(
            images    = [fake_img],
            image_ids = ["TEST_001"],
            prompt    = "a dress",
            success   = True,
        )

        with (
            patch("sys.argv", [
                "generate.py",
                "--prompt", "a dress",
                "--device", "cpu",
                "--steps", "2",
                "--size", "square_512",
                "--output-dir", str(tmp_path),
            ]),
            patch("src.generation.generate.FashionSDXLGenerator") as MockGen,
        ):
            instance = MockGen.return_value
            instance.load_model.return_value = None
            instance.generate_image.return_value = mock_output
            instance.save_output.return_value = [tmp_path / "test.png"]
            instance.unload_model.return_value = None
            instance.get_info.return_value = {}

            code = main()

        assert code == 0

    def test_load_failure_returns_one(self, tmp_path):
        from src.generation.generate import main
        with (
            patch("sys.argv", [
                "generate.py",
                "--prompt", "a dress",
                "--device", "cpu",
                "--output-dir", str(tmp_path),
            ]),
            patch("src.generation.generate.FashionSDXLGenerator") as MockGen,
        ):
            instance = MockGen.return_value
            instance.load_model.side_effect = RuntimeError("model not found")
            code = main()

        assert code == 1
