"""
week2/pipelines/text2image_pipeline.py
=========================================
Complete text-to-image fashion generation pipeline.

Flow
----
  Input prompt (str or FashionRecord)
        ↓
  PromptValidator   — sanitise & check tokens
        ↓
  PromptBuilder     — add style/quality tags
        ↓
  SDXLGenerator     — SDXL inference
        ↓
  QualityScorer     — CLIP + quality checks
        ↓
  EvaluationReport  — per-run report
        ↓
  Output (images + metadata + report)

Usage
-----
    from src.generation.pipelines.text2image_pipeline import Text2ImagePipeline
    from src.utils.config_manager import get_config

    pipeline = Text2ImagePipeline(config=get_config())
    result = pipeline.run(
        prompt  = "A woman in an elegant red silk evening gown",
        style   = "luxury",
        gender  = "women",
    )
    print(result)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.generation.pipelines.base_pipeline import BasePipeline, PipelineRunResult
from src.generation.generator.sdxl_generator import SDXLGenerator
from src.generation.prompts.prompt_builder import PromptBuilder
from src.generation.prompts.prompt_validator import PromptValidator
from src.evaluation.week2_quality_scorer import QualityScorer
from src.evaluation.week2_evaluation_report import EvaluationReport
from src.utils.logging_setup import generation_logger


class Text2ImagePipeline(BasePipeline):
    """
    End-to-end text-to-image fashion generation pipeline.

    Parameters
    ----------
    config : Week2Config
    """

    def __init__(self, config) -> None:
        super().__init__(config, name="Text2ImagePipeline")
        self._generator: Optional[SDXLGenerator] = None
        self._builder   = PromptBuilder(config)
        self._validator = PromptValidator(
            max_tokens  = config.prompts.structure.max_tokens,
        )
        self._scorer    = QualityScorer(config)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Pre-load SDXL model."""
        super().setup()
        self._generator = SDXLGenerator(config=self._cfg)
        use_refiner = self._cfg.generation.use_refiner
        self._generator.warm_up(load_refiner=use_refiner)

    def teardown(self) -> None:
        """Release model memory."""
        if self._generator:
            self._generator.unload()
            self._generator = None
        super().teardown()

    # ── Main Run ──────────────────────────────────────────────────────────

    def run(
        self,
        prompt:             str,
        style:              Optional[str]  = None,
        gender:             Optional[str]  = None,
        season:             Optional[str]  = None,
        occasion:           Optional[str]  = None,
        negative_prompt:    str            = "",
        num_images:         int            = 1,
        seed:               Optional[int]  = None,
        width:              Optional[int]  = None,
        height:             Optional[int]  = None,
        steps:              Optional[int]  = None,
        guidance_scale:     Optional[float]= None,
        scheduler:          Optional[str]  = None,
        preset:             Optional[str]  = None,
        save:               bool           = True,
        evaluate:           bool           = True,
        save_report:        bool           = True,
        output_dir:         Optional[Path] = None,
        raw_prompt:         bool           = False,
    ) -> PipelineRunResult:
        """
        Run the full text-to-image pipeline.

        Parameters
        ----------
        prompt : str
            Base description (e.g. "a woman in a floral summer dress").
        style : str, optional
            Style preset name ("luxury", "streetwear", etc.).
        gender : str, optional  men | women | unisex
        season : str, optional  spring | summer | autumn | winter
        occasion : str, optional
        negative_prompt : str   Additional negative tags.
        num_images : int        How many images to generate.
        seed : int, optional    Fixed seed (-1 = random).
        width, height : int, optional   Override canvas size.
        steps : int, optional
        guidance_scale : float, optional
        scheduler : str, optional
        preset : str, optional  Use a generation preset ("draft"/"high_quality").
        save : bool             Save images to disk.
        evaluate : bool         Run quality evaluation.
        save_report : bool      Save evaluation report JSON.
        output_dir : Path, optional
        raw_prompt : bool       Skip PromptBuilder; use prompt verbatim.

        Returns
        -------
        PipelineRunResult
        """
        result   = PipelineRunResult()
        t_start  = self._log_start(prompt=prompt[:60], style=style, n=num_images)

        # ── Apply generation preset overrides ─────────────────────────────
        if preset and preset in self._cfg.generation.presets:
            p = self._cfg.generation.presets[preset]
            width  = width  or p.width
            height = height or p.height
            steps  = steps  or p.num_inference_steps
            guidance_scale = guidance_scale or p.guidance_scale
            logger.debug("Preset {!r} applied", preset)

        # ── Ensure generator is ready ─────────────────────────────────────
        if self._generator is None:
            self._generator = SDXLGenerator(config=self._cfg)

        # ── Step 1: Prompt validation ─────────────────────────────────────
        val_result = self._validator.validate(prompt)
        if not val_result.is_valid:
            result.success = False
            result.errors  = val_result.errors
            self._log_end(t_start, result)
            return result

        for w in val_result.warnings:
            result.warnings.append(w)

        # ── Step 2: Prompt building ───────────────────────────────────────
        if raw_prompt:
            final_positive = val_result.sanitized_prompt
            final_negative = negative_prompt
            style_name     = ""
        else:
            built = self._builder.build(
                subject         = val_result.sanitized_prompt,
                style           = style,
                gender          = gender,
                season          = season,
                occasion        = occasion,
                extra_negative  = [negative_prompt] if negative_prompt else None,
            )
            final_positive = built.positive
            final_negative = built.negative
            style_name     = built.style_name
            if built.was_truncated:
                result.warnings.append("Prompt was truncated to fit token limit.")

        # ── Step 3: Generation ────────────────────────────────────────────
        gen_result = None
        with generation_logger(prompt=final_positive[:80], style=style_name):
            gen_result = self._generator.generate(
                prompt                = final_positive,
                negative_prompt       = final_negative,
                width                 = width,
                height                = height,
                num_inference_steps   = steps,
                guidance_scale        = guidance_scale,
                seed                  = seed if seed is not None else -1,
                num_images_per_prompt = num_images,
                scheduler             = scheduler,
                save                  = save,
                output_dir            = output_dir or self._cfg.output_dir,
            )

        if not gen_result.success:
            result.success = False
            result.errors.append(f"Generation failed: {gen_result.error}")
            self._log_end(t_start, result)
            return result

        result.total_generated = len(gen_result.images)
        result.output_paths    = [p for p in gen_result.image_paths if p]

        # ── Step 4: Evaluation ────────────────────────────────────────────
        scores = []
        if evaluate and gen_result.images:
            scores = self._scorer.score_batch(
                images    = gen_result.images,
                image_ids = gen_result.image_ids,
                prompts   = [final_positive] * len(gen_result.images),
            )
            result.passed_evaluation = sum(1 for s in scores if s.passed)
            result.failed_evaluation = sum(1 for s in scores if not s.passed)

            # Step 5: Report
            if save_report:
                report = EvaluationReport(scores, config=self._cfg)
                result.report_path = report.save()
                logger.info(report.summary())

        result.success = True
        result.extra   = {
            "seed":           gen_result.seed,
            "image_ids":      gen_result.image_ids,
            "final_prompt":   final_positive,
            "style_applied":  style_name,
        }

        self._log_end(t_start, result)
        return result

    # ── Convenience: generate from dataset record ─────────────────────────

    def run_from_record(
        self,
        record: Dict[str, Any],
        **kwargs,
    ) -> PipelineRunResult:
        """
        Run the pipeline using a Week 1 fashion dataset record as input.

        Parameters
        ----------
        record : dict   A unified fashion record (from Week 1 data pipeline).
        **kwargs        Additional keyword arguments forwarded to ``run()``.

        Returns
        -------
        PipelineRunResult
        """
        built = self._builder.build_from_metadata(record)
        return self.run(
            prompt   = built.positive,
            raw_prompt = True,      # Already fully built
            **kwargs,
        )
