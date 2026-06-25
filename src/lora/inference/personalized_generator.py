"""
week4/inference/personalized_generator.py
=========================================
Personalized Fashion Generation System.
Integrates user brand, color, and style preferences with the LoRA SDXL inference pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from src.lora.inference.lora_inference import LoraInferenceSystem


class PersonalizedFashionGenerator:
    """
    Coordinates user profile style preference mappings (brand, color, preferred style)
    and executes personalized text-to-image SDXL inference runs with target brand LoRA adapters.
    """

    def __init__(
        self,
        inference_system: Optional[LoraInferenceSystem] = None,
        config: Any = None,
        output_dir: Optional[Union[str, Path]] = None,
        dry_run: bool = True
    ) -> None:
        """
        Initialize the personalized generator.

        Parameters
        ----------
        inference_system : LoraInferenceSystem, optional
            Pre-configured inference system wrapper.
        config : Any, optional
        output_dir : Path or str, optional
        dry_run : bool
        """
        self.inference_system = inference_system or LoraInferenceSystem(
            config=config,
            output_dir=output_dir,
            dry_run=dry_run
        )
        self.output_dir = Path(self.inference_system.output_dir)
        logger.info(f"Initialized PersonalizedFashionGenerator | output_dir={self.output_dir}")

    def build_personalized_prompt(
        self,
        base_item: str,
        color: str,
        style: str
    ) -> str:
        """
        Build a descriptively personalized prompt incorporating color and style keywords.

        Parameters
        ----------
        base_item : str
            E.g. "hoodie", "jacket", "shirt".
        color : str
            User's favorite color.
        style : str
            User's preferred style.

        Returns
        -------
        str
            Enriched prompt.
        """
        color_clean = color.lower().strip()
        style_clean = style.lower().strip()
        item_clean = base_item.lower().strip()
        
        return f"A high-fidelity fashion photo of a {color_clean} {style_clean} {item_clean}"

    def generate_personalized_design(
        self,
        base_item: str,
        preferences: Dict[str, str],
        scale: float = 1.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a personalized design matching user preferences.

        Parameters
        ----------
        base_item : str
            The target apparel (e.g. "hoodie").
        preferences : dict of str to str
            Dictionary containing keys:
            - "favorite_brand" (e.g. "nike", "gucci", "zara", "h&m")
            - "favorite_color" (e.g. "black", "red")
            - "preferred_style" (e.g. "streetwear", "minimalist")
        scale : float
            LoRA active adapter blending scale.
        seed : int, optional

        Returns
        -------
        dict
            Inference output details matching LoraInferenceSystem schema.
        """
        fav_brand = preferences.get("favorite_brand", "nike").lower().strip()
        fav_color = preferences.get("favorite_color", "black").lower().strip()
        pref_style = preferences.get("preferred_style", "streetwear").lower().strip()

        logger.info(f"Generating personalized design for profile: brand={fav_brand}, color={fav_color}, style={pref_style}")

        # 1. Build personalized base prompt
        personalized_prompt = self.build_personalized_prompt(
            base_item=base_item,
            color=fav_color,
            style=pref_style
        )

        # 2. Delegate to the LoRA Inference System
        # This will switch to the target brand's LoRA, enrich prompt, generate design, and serialize metadata
        result = self.inference_system.generate(
            prompt=personalized_prompt,
            brand=fav_brand,
            scale=scale,
            seed=seed
        )

        # 3. Add personalized preferences details to the result payload
        result["preferences"] = {
            "favorite_brand": fav_brand,
            "favorite_color": fav_color,
            "preferred_style": pref_style
        }

        # 4. Update the output sidecar JSON file to include the preferences key
        metadata_path = Path(result["metadata_path"])
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta_content = json.load(f)
                
                meta_content["preferences"] = result["preferences"]
                
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(meta_content, f, indent=2, sort_keys=True)
                logger.debug(f"Appended personalized profile to sidecar metadata: {metadata_path}")
            except Exception as err:
                logger.warning(f"Failed to append preferences to sidecar metadata: {err}")

        return result
