"""
week2 — Text-to-Image Fashion Generation Foundation
====================================================

Week 2 of the AI-Powered Fashion Design Assistant project.
Provides a complete SDXL-based text-to-image generation system
with prompt engineering, quality evaluation, and batch pipelines.

Quick Start
-----------
    from week2.config_manager import get_config
    from week2.logging_setup import setup_logging
    from week2.pipelines import Text2ImagePipeline

    setup_logging()
    cfg      = get_config()
    pipeline = Text2ImagePipeline(config=cfg)
    result   = pipeline.run(
        prompt = "An elegant red silk evening gown, fashion photography",
        style  = "luxury",
        gender = "women",
    )
    print(result)
"""

__version__ = "2.0.0"
__author__  = "AI Fashion Design Team"
__week__    = 2
