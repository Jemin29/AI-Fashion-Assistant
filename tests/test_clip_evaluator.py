"""
week2/tests/test_clip_evaluator.py
=====================================
Comprehensive test suite for CLIPEvaluator.

Strategy
--------
All tests run in "stub mode" (no GPU / no transformers required):
  - Mock _compute_similarity() to inject controlled similarity values
  - Test all data structures, metrics, ranking, comparison, aggregate
  - Test graceful degradation path (no model loaded)
  - Test edge cases (empty inputs, None images, single image)
  - Test module-level convenience functions

Coverage targets
----------------
  - CLIPScore dataclass
  - RankedImage dataclass
  - ComparisonResult dataclass
  - CLIPEvaluator.__init__
  - CLIPEvaluator.load_clip (stub and mocked)
  - CLIPEvaluator.evaluate
  - CLIPEvaluator.rank_images
  - CLIPEvaluator.compare_generations
  - CLIPEvaluator.evaluate_batch
  - CLIPEvaluator.aggregate
  - CLIPEvaluator._classify_alignment
  - CLIPEvaluator._alignment_to_score
  - CLIPEvaluator._resolve_device
  - CLIPEvaluator.unload
  - Module-level: load_clip, evaluate, rank_images, compare_generations
  - ALIGNMENT_THRESHOLDS, DEFAULT_MODEL_ID, DEFAULT_MIN_SIMILARITY
"""

from __future__ import annotations

import sys
import uuid
import json
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import numpy as np
from PIL import Image as PILImage

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.week2_clip_evaluator import (
    CLIPEvaluator,
    CLIPScore,
    RankedImage,
    ComparisonResult,
    ALIGNMENT_THRESHOLDS,
    DEFAULT_MODEL_ID,
    DEFAULT_MIN_SIMILARITY,
    load_clip,
    evaluate,
    rank_images,
    compare_generations,
)


# =============================================================================
# ── Fixtures
# =============================================================================

def _make_image(mean: int = 128, size: int = 64) -> PILImage.Image:
    """Create a solid-colour PIL image for testing."""
    arr = np.full((size, size, 3), mean, dtype=np.uint8)
    return PILImage.fromarray(arr)

@pytest.fixture
def img():
    return _make_image(128)

@pytest.fixture
def img_red():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = 200   # red channel
    return PILImage.fromarray(arr)

@pytest.fixture
def img_blue():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 2] = 200   # blue channel
    return PILImage.fromarray(arr)

@pytest.fixture
def evaluator():
    """CLIPEvaluator in stub mode (no model loaded)."""
    return CLIPEvaluator(device="cpu", min_similarity=DEFAULT_MIN_SIMILARITY)

@pytest.fixture
def mock_evaluator(evaluator):
    """CLIPEvaluator with _compute_similarity mocked to return 0.28."""
    evaluator._loaded = True
    with patch.object(evaluator, "_compute_similarity", return_value=0.28):
        yield evaluator

@pytest.fixture
def images(img, img_red, img_blue):
    return [img, img_red, img_blue]

PROMPT = "a black streetwear hoodie on an urban street"
PROMPT_LUXURY = "an emerald green silk evening gown on a fashion runway"


# =============================================================================
# ── Module Constants
# =============================================================================

class TestModuleConstants:

    def test_default_model_id(self):
        assert DEFAULT_MODEL_ID == "openai/clip-vit-large-patch14"

    def test_default_min_similarity(self):
        assert 0.0 < DEFAULT_MIN_SIMILARITY < 1.0

    def test_alignment_thresholds_keys(self):
        expected = {"very high", "high", "medium", "low"}
        assert set(ALIGNMENT_THRESHOLDS.keys()) == expected

    def test_alignment_thresholds_ordered(self):
        # very high > high > medium > low
        assert ALIGNMENT_THRESHOLDS["very high"] > ALIGNMENT_THRESHOLDS["high"]
        assert ALIGNMENT_THRESHOLDS["high"] > ALIGNMENT_THRESHOLDS["medium"]
        assert ALIGNMENT_THRESHOLDS["medium"] > ALIGNMENT_THRESHOLDS["low"]

    def test_alignment_thresholds_in_zero_one_range(self):
        for label, val in ALIGNMENT_THRESHOLDS.items():
            assert 0.0 <= val <= 1.0, f"{label} threshold {val} out of range"


# =============================================================================
# ── CLIPScore Dataclass
# =============================================================================

class TestCLIPScore:

    def test_default_instantiation(self):
        sc = CLIPScore()
        assert sc.clip_score == 0.0
        assert sc.alignment  == "low"
        assert sc.passed     is False

    def test_custom_values(self):
        sc = CLIPScore(
            image_id      = "test_001",
            clip_score    = 0.92,
            alignment     = "very high",
            semantic_score= 1.0,
            ranking_score = 0.95,
            passed        = True,
        )
        assert sc.clip_score    == 0.92
        assert sc.alignment     == "very high"
        assert sc.semantic_score== 1.0
        assert sc.passed        is True

    def test_to_dict_returns_dict(self):
        sc = CLIPScore(image_id="x", clip_score=0.92, alignment="high", passed=True)
        d  = sc.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_required_keys(self):
        sc = CLIPScore(image_id="x", clip_score=0.92, alignment="high", passed=True)
        d  = sc.to_dict()
        for key in ["clip_score", "alignment", "semantic_score", "ranking_score",
                    "passed", "image_id", "prompt_preview", "model", "device",
                    "eval_ms", "error"]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_clip_score_matches_spec(self):
        """Output matches the spec: {'clip_score': 0.92, 'alignment': 'high', ...}"""
        sc = CLIPScore(image_id="x", clip_score=0.92, alignment="high", passed=True)
        d  = sc.to_dict()
        assert d["clip_score"] == 0.92
        assert d["alignment"]  == "high"

    def test_to_dict_is_json_serialisable(self):
        sc = CLIPScore(
            image_id    = "test",
            clip_score  = 0.75,
            alignment   = "high",
            eval_ms     = 12.5,
        )
        json.dumps(sc.to_dict())   # Must not raise

    def test_repr_contains_image_id(self):
        sc = CLIPScore(image_id="abc123", clip_score=0.5)
        assert "abc123" in repr(sc)

    def test_repr_contains_score(self):
        sc = CLIPScore(image_id="x", clip_score=0.876)
        assert "0.876" in repr(sc)

    def test_auto_generated_image_id(self):
        sc1 = CLIPScore()
        sc2 = CLIPScore()
        assert sc1.image_id != sc2.image_id

    def test_clip_score_rounded_in_to_dict(self):
        sc = CLIPScore(clip_score=0.123456789)
        d  = sc.to_dict()
        assert d["clip_score"] == 0.1235   # rounded to 4dp


# =============================================================================
# ── RankedImage Dataclass
# =============================================================================

class TestRankedImage:

    def test_instantiation(self):
        ri = RankedImage(
            rank=1, image_id="a", clip_score=0.31,
            alignment="very high", ranking_score=1.0, passed=True,
        )
        assert ri.rank == 1
        assert ri.passed is True

    def test_to_dict_keys(self):
        ri = RankedImage(rank=2, image_id="b", clip_score=0.25,
                         alignment="high", ranking_score=0.5, passed=True)
        d  = ri.to_dict()
        for key in ["rank", "image_id", "clip_score", "alignment", "ranking_score", "passed"]:
            assert key in d

    def test_to_dict_json_serialisable(self):
        ri = RankedImage(rank=1, image_id="x", clip_score=0.3,
                         alignment="high", ranking_score=0.9, passed=True)
        json.dumps(ri.to_dict())


# =============================================================================
# ── ComparisonResult Dataclass
# =============================================================================

class TestComparisonResult:

    def test_instantiation(self):
        cr = ComparisonResult(
            prompt_preview="a navy suit",
            mean_score_a=0.28, mean_score_b=0.25,
            winner="A", delta=0.03,
        )
        assert cr.winner == "A"
        assert cr.delta  == pytest.approx(0.03)

    def test_to_dict_keys(self):
        cr = ComparisonResult(
            prompt_preview="test", mean_score_a=0.3, mean_score_b=0.2,
            winner="A", delta=0.1,
        )
        d = cr.to_dict()
        for key in ["prompt_preview", "mean_score_a", "mean_score_b",
                    "winner", "delta", "scores_a", "scores_b"]:
            assert key in d

    def test_to_dict_json_serialisable(self):
        cr = ComparisonResult(
            prompt_preview="test", mean_score_a=0.3, mean_score_b=0.2,
            winner="B", delta=-0.1,
        )
        json.dumps(cr.to_dict())


# =============================================================================
# ── CLIPEvaluator — Initialisation
# =============================================================================

class TestCLIPEvaluatorInit:

    def test_default_instantiation(self):
        ev = CLIPEvaluator()
        assert ev is not None

    def test_device_auto_resolves(self):
        ev = CLIPEvaluator(device="auto")
        assert ev.device in ("cpu", "cuda")

    def test_device_cpu_explicit(self):
        ev = CLIPEvaluator(device="cpu")
        assert ev.device == "cpu"

    def test_model_id_set(self):
        ev = CLIPEvaluator(model_id="openai/clip-vit-base-patch32")
        assert ev.model_id == "openai/clip-vit-base-patch32"

    def test_min_similarity_set(self):
        ev = CLIPEvaluator(min_similarity=0.35)
        assert ev.min_similarity == 0.35

    def test_is_loaded_false_initially(self, evaluator):
        assert evaluator.is_loaded is False

    def test_model_none_initially(self, evaluator):
        assert evaluator.model is None

    def test_processor_none_initially(self, evaluator):
        assert evaluator.processor is None

    def test_repr(self, evaluator):
        r = repr(evaluator)
        assert "CLIPEvaluator" in r
        assert "not loaded" in r

    def test_repr_loaded(self, evaluator):
        evaluator._loaded = True
        r = repr(evaluator)
        assert "loaded" in r


# =============================================================================
# ── CLIPEvaluator.load_clip()
# =============================================================================

class TestLoadClip:

    def test_returns_bool(self, evaluator):
        result = evaluator.load_clip()
        assert isinstance(result, bool)

    def test_returns_false_when_transformers_unavailable(self, evaluator):
        with patch("src.evaluation.week2_clip_evaluator._HF_AVAILABLE", False):
            result = evaluator.load_clip(force=True)
        assert result is False

    def test_idempotent_when_loaded(self, evaluator):
        evaluator._loaded = True
        result = evaluator.load_clip()
        assert result is True

    def test_force_reloads(self, evaluator):
        evaluator._loaded = True
        with patch("src.evaluation.week2_clip_evaluator._HF_AVAILABLE", False):
            result = evaluator.load_clip(force=True)
        # Even with HF unavailable, force=True should attempt and return False
        assert result is False

    def test_load_failure_does_not_raise(self, evaluator):
        with patch("src.evaluation.week2_clip_evaluator._HF_AVAILABLE", True):
            with patch("src.evaluation.week2_clip_evaluator.CLIPProcessor") as mp:
                mp.from_pretrained.side_effect = RuntimeError("test error")
                result = evaluator.load_clip(force=True)
        assert result is False

    def test_load_sets_loaded_flag(self, evaluator):
        mock_model     = MagicMock()
        mock_processor = MagicMock()
        with patch("src.evaluation.week2_clip_evaluator._HF_AVAILABLE", True):
            with patch("src.evaluation.week2_clip_evaluator.CLIPModel") as mm:
                with patch("src.evaluation.week2_clip_evaluator.CLIPProcessor") as mp:
                    mm.from_pretrained.return_value = mock_model
                    mock_model.to.return_value = mock_model
                    mp.from_pretrained.return_value = mock_processor
                    result = evaluator.load_clip(force=True)
        if result:
            assert evaluator._loaded is True


# =============================================================================
# ── CLIPEvaluator.evaluate() — Stub Mode
# =============================================================================

class TestEvaluateStub:
    """Tests when model is NOT loaded (stub mode)."""

    @pytest.fixture(autouse=True)
    def mock_hf_unavailable(self):
        with patch("src.evaluation.week2_clip_evaluator._HF_AVAILABLE", False):
            yield

    def test_returns_clip_score(self, evaluator, img):
        result = evaluator.evaluate(img, PROMPT)
        assert isinstance(result, CLIPScore)

    def test_stub_clip_score_zero(self, evaluator, img):
        result = evaluator.evaluate(img, PROMPT)
        assert result.clip_score == 0.0

    def test_stub_alignment_low(self, evaluator, img):
        result = evaluator.evaluate(img, PROMPT)
        assert result.alignment == "low"

    def test_stub_passed_false(self, evaluator, img):
        result = evaluator.evaluate(img, PROMPT)
        assert result.passed is False

    def test_stub_error_non_empty(self, evaluator, img):
        result = evaluator.evaluate(img, PROMPT)
        assert result.error != ""

    def test_custom_image_id(self, evaluator, img):
        result = evaluator.evaluate(img, PROMPT, image_id="custom_id_001")
        assert result.image_id == "custom_id_001"

    def test_prompt_preview_truncated(self, evaluator, img):
        long_prompt = "a " * 200
        result = evaluator.evaluate(img, long_prompt)
        assert len(result.prompt_preview) <= 80

    def test_to_dict_output_format(self, evaluator, img):
        """Verify the exact output schema matches the spec."""
        d = evaluator.evaluate(img, PROMPT).to_dict()
        assert "clip_score" in d
        assert "alignment"  in d
        assert isinstance(d["clip_score"], float)
        assert isinstance(d["alignment"],  str)

    def test_eval_ms_nonnegative(self, evaluator, img):
        result = evaluator.evaluate(img, PROMPT)
        assert result.eval_ms >= 0.0


# =============================================================================
# ── CLIPEvaluator.evaluate() — Mocked Mode (simulated real scores)
# =============================================================================

class TestEvaluateMocked:
    """Tests with _compute_similarity mocked to return controlled values."""

    @pytest.mark.parametrize("sim,expected_alignment", [
        (0.35, "very high"),
        (0.30, "very high"),
        (0.27, "high"),
        (0.25, "high"),
        (0.22, "medium"),
        (0.20, "medium"),
        (0.15, "low"),
        (0.00, "low"),
    ])
    def test_alignment_classification(self, evaluator, img, sim, expected_alignment):
        with patch.object(evaluator, "_compute_similarity", return_value=sim):
            evaluator._loaded = True
            result = evaluator.evaluate(img, PROMPT)
        assert result.alignment == expected_alignment

    @pytest.mark.parametrize("sim,expected_passed", [
        (0.30, True),
        (0.20, True),   # At threshold
        (0.19, False),
        (0.00, False),
    ])
    def test_passed_threshold(self, evaluator, img, sim, expected_passed):
        with patch.object(evaluator, "_compute_similarity", return_value=sim):
            evaluator._loaded = True
            result = evaluator.evaluate(img, PROMPT)
        assert result.passed == expected_passed

    def test_clip_score_matches_similarity(self, evaluator, img):
        with patch.object(evaluator, "_compute_similarity", return_value=0.28):
            evaluator._loaded = True
            result = evaluator.evaluate(img, PROMPT)
        assert result.clip_score == pytest.approx(0.28, abs=1e-4)

    def test_semantic_score_for_high(self, evaluator, img):
        with patch.object(evaluator, "_compute_similarity", return_value=0.27):
            evaluator._loaded = True
            result = evaluator.evaluate(img, PROMPT)
        assert result.semantic_score == 0.75

    def test_semantic_score_for_very_high(self, evaluator, img):
        with patch.object(evaluator, "_compute_similarity", return_value=0.35):
            evaluator._loaded = True
            result = evaluator.evaluate(img, PROMPT)
        assert result.semantic_score == 1.00

    def test_no_error_on_success(self, evaluator, img):
        with patch.object(evaluator, "_compute_similarity", return_value=0.28):
            evaluator._loaded = True
            result = evaluator.evaluate(img, PROMPT)
        assert result.error == ""

    def test_model_field_set(self, evaluator, img):
        with patch.object(evaluator, "_compute_similarity", return_value=0.28):
            evaluator._loaded = True
            result = evaluator.evaluate(img, PROMPT)
        assert result.model == evaluator.model_id

    def test_device_field_set(self, evaluator, img):
        evaluator.device = "cpu"
        with patch.object(evaluator, "_compute_similarity", return_value=0.25):
            evaluator._loaded = True
            result = evaluator.evaluate(img, PROMPT)
        assert result.device == "cpu"

    def test_to_dict_matches_spec(self, evaluator, img):
        """The to_dict() must produce {'clip_score': X, 'alignment': Y, ...}"""
        with patch.object(evaluator, "_compute_similarity", return_value=0.28):
            evaluator._loaded = True
            d = evaluator.evaluate(img, PROMPT, image_id="spec_test").to_dict()
        assert d["clip_score"] == pytest.approx(0.28, abs=0.001)
        assert d["alignment"]  == "high"
        assert d["passed"]     is True
        assert d["image_id"]   == "spec_test"
        assert d["error"]      == ""


# =============================================================================
# ── CLIPEvaluator.rank_images()
# =============================================================================

class TestRankImages:

    def test_returns_list(self, evaluator, images):
        result = evaluator.rank_images(images, PROMPT)
        assert isinstance(result, list)

    def test_length_matches_input(self, evaluator, images):
        result = evaluator.rank_images(images, PROMPT)
        assert len(result) == len(images)

    def test_returns_ranked_image_objects(self, evaluator, images):
        result = evaluator.rank_images(images, PROMPT)
        for ri in result:
            assert isinstance(ri, RankedImage)

    def test_ranks_are_sequential(self, evaluator, images):
        result = evaluator.rank_images(images, PROMPT)
        ranks  = [ri.rank for ri in result]
        assert ranks == list(range(1, len(images) + 1))

    def test_empty_input_returns_empty(self, evaluator):
        result = evaluator.rank_images([], PROMPT)
        assert result == []

    def test_single_image_rank_one(self, evaluator, img):
        result = evaluator.rank_images([img], PROMPT)
        assert len(result) == 1
        assert result[0].rank == 1

    def test_custom_image_ids(self, evaluator, images):
        ids    = ["id_a", "id_b", "id_c"]
        result = evaluator.rank_images(images, PROMPT, image_ids=ids)
        result_ids = {ri.image_id for ri in result}
        assert result_ids == set(ids)

    def test_descending_order_by_default(self, evaluator, images):
        sims = [0.10, 0.30, 0.20]
        call_count = [0]
        def mock_sim(image, prompt):
            idx = call_count[0]
            call_count[0] += 1
            return sims[idx % len(sims)]
        with patch.object(evaluator, "_compute_similarity", side_effect=mock_sim):
            evaluator._loaded = True
            result = evaluator.rank_images(images, PROMPT)
        # Rank 1 should be highest score (0.30)
        assert result[0].clip_score >= result[1].clip_score

    def test_to_dict_per_item(self, evaluator, images):
        result = evaluator.rank_images(images, PROMPT)
        for ri in result:
            d = ri.to_dict()
            assert "rank"         in d
            assert "clip_score"   in d
            assert "alignment"    in d
            assert "ranking_score"in d
            assert "passed"       in d

    def test_ranking_score_range(self, evaluator, images):
        result = evaluator.rank_images(images, PROMPT)
        for ri in result:
            assert 0.0 <= ri.ranking_score <= 1.0

    def test_with_mocked_distinct_scores(self, evaluator, images):
        sims = [0.10, 0.30, 0.22]
        call_count = [0]
        def mock_sim(image, prompt):
            idx = call_count[0] % len(sims)
            call_count[0] += 1
            return sims[idx]
        with patch.object(evaluator, "_compute_similarity", side_effect=mock_sim):
            evaluator._loaded = True
            result = evaluator.rank_images(images, PROMPT)
        assert result[0].clip_score == max(sims)


# =============================================================================
# ── CLIPEvaluator.compare_generations()
# =============================================================================

class TestCompareGenerations:

    def test_returns_comparison_result(self, evaluator, img, img_red):
        result = evaluator.compare_generations([img], [img_red], PROMPT)
        assert isinstance(result, ComparisonResult)

    def test_winner_a_when_a_higher(self, evaluator, img, img_red):
        def mock_sim_factory(score):
            def sim(image, prompt):
                return score
            return sim
        # Patch to return 0.30 for first call (A) and 0.20 for second (B)
        sims = [0.30, 0.20]
        call_count = [0]
        def mock_sim(image, prompt):
            idx = call_count[0] % len(sims)
            call_count[0] += 1
            return sims[idx]
        with patch.object(evaluator, "_compute_similarity", side_effect=mock_sim):
            evaluator._loaded = True
            result = evaluator.compare_generations([img], [img_red], PROMPT)
        assert result.winner == "A"
        assert result.delta  > 0

    def test_winner_b_when_b_higher(self, evaluator, img, img_red):
        sims = [0.20, 0.30]
        call_count = [0]
        def mock_sim(image, prompt):
            idx = call_count[0] % len(sims)
            call_count[0] += 1
            return sims[idx]
        with patch.object(evaluator, "_compute_similarity", side_effect=mock_sim):
            evaluator._loaded = True
            result = evaluator.compare_generations([img], [img_red], PROMPT)
        assert result.winner == "B"
        assert result.delta  < 0

    def test_winner_tie_when_equal(self, evaluator, img):
        with patch.object(evaluator, "_compute_similarity", return_value=0.25):
            evaluator._loaded = True
            result = evaluator.compare_generations([img], [img], PROMPT)
        assert result.winner == "tie"

    def test_scores_a_populated(self, evaluator, img, img_red, img_blue):
        result = evaluator.compare_generations([img, img_red], [img_blue], PROMPT)
        assert len(result.scores_a) == 2

    def test_scores_b_populated(self, evaluator, img, img_red):
        result = evaluator.compare_generations([img], [img_red, img_red], PROMPT)
        assert len(result.scores_b) == 2

    def test_prompt_preview_in_result(self, evaluator, img):
        result = evaluator.compare_generations([img], [img], PROMPT)
        assert result.prompt_preview == PROMPT[:80]

    def test_to_dict_keys(self, evaluator, img):
        result = evaluator.compare_generations([img], [img], PROMPT)
        d      = result.to_dict()
        for key in ["prompt_preview", "mean_score_a", "mean_score_b",
                    "winner", "delta", "scores_a", "scores_b"]:
            assert key in d

    def test_to_dict_json_serialisable(self, evaluator, img):
        result = evaluator.compare_generations([img], [img], PROMPT)
        json.dumps(result.to_dict())

    def test_custom_ids(self, evaluator, img, img_red):
        result = evaluator.compare_generations(
            [img], [img_red], PROMPT,
            ids_a=["a1"], ids_b=["b1"],
        )
        assert result.scores_a[0].image_id == "a1"
        assert result.scores_b[0].image_id == "b1"

    def test_empty_sets_both(self, evaluator):
        result = evaluator.compare_generations([], [], PROMPT)
        assert result.winner in ("tie", "A", "B")
        assert result.mean_score_a == 0.0
        assert result.mean_score_b == 0.0


# =============================================================================
# ── CLIPEvaluator.evaluate_batch()
# =============================================================================

class TestEvaluateBatch:

    def test_returns_list(self, evaluator, images):
        result = evaluator.evaluate_batch(images, PROMPT)
        assert isinstance(result, list)

    def test_length_matches_input(self, evaluator, images):
        result = evaluator.evaluate_batch(images, PROMPT)
        assert len(result) == len(images)

    def test_all_clip_scores(self, evaluator, images):
        result = evaluator.evaluate_batch(images, PROMPT)
        for sc in result:
            assert isinstance(sc, CLIPScore)

    def test_single_shared_prompt(self, evaluator, images):
        result = evaluator.evaluate_batch(images, prompts=PROMPT)
        assert len(result) == len(images)

    def test_per_image_prompts(self, evaluator, images):
        prompts = [PROMPT, PROMPT_LUXURY, "a casual white t-shirt"]
        result  = evaluator.evaluate_batch(images, prompts=prompts)
        assert len(result) == 3

    def test_prompts_length_mismatch_raises(self, evaluator, images):
        with pytest.raises(ValueError, match="length"):
            evaluator.evaluate_batch(images, prompts=["only one prompt"])

    def test_custom_image_ids(self, evaluator, images):
        ids    = ["batch_0", "batch_1", "batch_2"]
        result = evaluator.evaluate_batch(images, PROMPT, image_ids=ids)
        result_ids = [sc.image_id for sc in result]
        assert result_ids == ids

    def test_empty_batch_returns_empty(self, evaluator):
        result = evaluator.evaluate_batch([], PROMPT)
        assert result == []

    def test_each_result_has_correct_id(self, evaluator, images):
        ids    = [f"img_{i}" for i in range(len(images))]
        result = evaluator.evaluate_batch(images, PROMPT, image_ids=ids)
        for sc, expected_id in zip(result, ids):
            assert sc.image_id == expected_id


# =============================================================================
# ── CLIPEvaluator.aggregate()
# =============================================================================

class TestAggregate:

    def _make_scores(self, clip_scores: list, passed: list = None) -> list:
        if passed is None:
            passed = [True] * len(clip_scores)
        return [
            CLIPScore(
                image_id    = f"img_{i}",
                clip_score  = s,
                alignment   = CLIPEvaluator._classify_alignment(s),
                passed      = p,
            )
            for i, (s, p) in enumerate(zip(clip_scores, passed))
        ]

    def test_returns_dict(self, evaluator):
        scores = self._make_scores([0.28, 0.31, 0.19])
        result = evaluator.aggregate(scores)
        assert isinstance(result, dict)

    def test_total_count(self, evaluator):
        scores = self._make_scores([0.28, 0.31, 0.19])
        result = evaluator.aggregate(scores)
        assert result["total"] == 3

    def test_passed_count(self, evaluator):
        scores = self._make_scores([0.28, 0.31, 0.19], passed=[True, True, False])
        result = evaluator.aggregate(scores)
        assert result["passed"] == 2
        assert result["failed"] == 1

    def test_pass_rate(self, evaluator):
        scores = self._make_scores([0.28, 0.31, 0.19, 0.15], passed=[True, True, False, False])
        result = evaluator.aggregate(scores)
        assert result["pass_rate"] == pytest.approx(0.5)

    def test_mean_clip_score(self, evaluator):
        scores = self._make_scores([0.20, 0.30])
        result = evaluator.aggregate(scores)
        assert result["mean_clip_score"] == pytest.approx(0.25)

    def test_min_clip_score(self, evaluator):
        scores = self._make_scores([0.20, 0.30, 0.25])
        result = evaluator.aggregate(scores)
        assert result["min_clip_score"] == pytest.approx(0.20)

    def test_max_clip_score(self, evaluator):
        scores = self._make_scores([0.20, 0.30, 0.25])
        result = evaluator.aggregate(scores)
        assert result["max_clip_score"] == pytest.approx(0.30)

    def test_std_clip_score_present(self, evaluator):
        scores = self._make_scores([0.20, 0.30])
        result = evaluator.aggregate(scores)
        assert "std_clip_score" in result

    def test_alignment_breakdown(self, evaluator):
        scores = self._make_scores([0.35, 0.27, 0.21, 0.10])
        result = evaluator.aggregate(scores)
        breakdown = result["alignment_breakdown"]
        assert "very high" in breakdown
        assert "high"      in breakdown
        assert "medium"    in breakdown
        assert "low"       in breakdown

    def test_empty_scores(self, evaluator):
        result = evaluator.aggregate([])
        assert result["total"] == 0
        assert result["passed"] == 0

    def test_model_device_in_result(self, evaluator):
        scores = self._make_scores([0.25])
        result = evaluator.aggregate(scores)
        assert "model"  in result
        assert "device" in result

    def test_result_is_json_serialisable(self, evaluator):
        scores = self._make_scores([0.28, 0.31, 0.19])
        result = evaluator.aggregate(scores)
        json.dumps(result)


# =============================================================================
# ── Static Methods
# =============================================================================

class TestStaticMethods:

    @pytest.mark.parametrize("score,expected", [
        (0.35, "very high"),
        (0.30, "very high"),
        (0.299, "high"),
        (0.25, "high"),
        (0.249, "medium"),
        (0.20, "medium"),
        (0.199, "low"),
        (0.00, "low"),
    ])
    def test_classify_alignment(self, score, expected):
        assert CLIPEvaluator._classify_alignment(score) == expected

    @pytest.mark.parametrize("alignment,expected", [
        ("very high", 1.00),
        ("high",      0.75),
        ("medium",    0.50),
        ("low",       0.25),
    ])
    def test_alignment_to_score(self, alignment, expected):
        assert CLIPEvaluator._alignment_to_score(alignment) == expected

    def test_alignment_to_score_unknown(self):
        # Unknown alignment should return 0.0
        result = CLIPEvaluator._alignment_to_score("unknown_label")
        assert result == 0.0

    @pytest.mark.parametrize("device,expected", [
        ("cpu",  "cpu"),
        ("cuda", "cuda"),
    ])
    def test_resolve_device_explicit(self, device, expected):
        result = CLIPEvaluator._resolve_device(device)
        assert result == expected

    def test_resolve_device_auto(self):
        result = CLIPEvaluator._resolve_device("auto")
        assert result in ("cpu", "cuda")


# =============================================================================
# ── Lifecycle: unload(), context manager
# =============================================================================

class TestLifecycle:

    def test_unload_clears_loaded_flag(self, evaluator):
        evaluator._loaded    = True
        evaluator._model     = MagicMock()
        evaluator._processor = MagicMock()
        evaluator.unload()
        assert evaluator.is_loaded is False

    def test_unload_clears_model(self, evaluator):
        evaluator._loaded    = True
        evaluator._model     = MagicMock()
        evaluator._processor = MagicMock()
        evaluator.unload()
        assert evaluator.model     is None
        assert evaluator.processor is None

    def test_unload_idempotent(self, evaluator):
        evaluator.unload()
        evaluator.unload()   # Should not raise

    def test_context_manager_enters(self):
        ev = CLIPEvaluator(device="cpu")
        with patch.object(ev, "load_clip", return_value=False):
            with ev as e:
                assert e is ev

    def test_context_manager_calls_unload_on_exit(self):
        ev = CLIPEvaluator(device="cpu")
        with patch.object(ev, "load_clip", return_value=False):
            with patch.object(ev, "unload") as mock_unload:
                with ev:
                    pass
                mock_unload.assert_called_once()


# =============================================================================
# ── _compute_similarity edge cases
# =============================================================================

class TestComputeSimilarityEdgeCases:

    def test_none_image_returns_none(self, evaluator):
        evaluator._loaded = True
        result = evaluator._compute_similarity(None, PROMPT)
        assert result is None

    def test_not_pil_image_returns_none(self, evaluator):
        evaluator._loaded = True
        result = evaluator._compute_similarity("not_an_image", PROMPT)
        assert result is None

    def test_not_loaded_returns_none(self, evaluator, img):
        evaluator._loaded = False
        with patch("src.evaluation.week2_clip_evaluator._HF_AVAILABLE", False):
            result = evaluator._compute_similarity(img, PROMPT)
        assert result is None


# =============================================================================
# ── Module-Level Convenience Functions
# =============================================================================

class TestModuleFunctions:

    def test_load_clip_returns_bool(self):
        result = load_clip()
        assert isinstance(result, bool)

    def test_evaluate_returns_dict(self, img):
        result = evaluate(img, PROMPT)
        assert isinstance(result, dict)

    def test_evaluate_required_keys(self, img):
        result = evaluate(img, PROMPT)
        assert "clip_score" in result
        assert "alignment"  in result

    def test_evaluate_clip_score_is_float(self, img):
        result = evaluate(img, PROMPT)
        assert isinstance(result["clip_score"], float)

    def test_rank_images_returns_list_of_dicts(self, images):
        result = rank_images(images, PROMPT)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)

    def test_rank_images_has_rank_key(self, images):
        result = rank_images(images, PROMPT)
        for item in result:
            assert "rank" in item

    def test_rank_images_length(self, images):
        result = rank_images(images, PROMPT)
        assert len(result) == len(images)

    def test_compare_generations_returns_dict(self, img, img_red):
        result = compare_generations([img], [img_red], PROMPT)
        assert isinstance(result, dict)

    def test_compare_generations_has_winner(self, img, img_red):
        result = compare_generations([img], [img_red], PROMPT)
        assert "winner" in result
        assert result["winner"] in ("A", "B", "tie")

    def test_compare_generations_has_delta(self, img, img_red):
        result = compare_generations([img], [img_red], PROMPT)
        assert "delta" in result

    def test_module_functions_json_serialisable(self, img):
        result = evaluate(img, PROMPT)
        json.dumps(result)
