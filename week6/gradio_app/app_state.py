"""
Week 6 — App State Management.
Provides session state container for the Gradio Creative Studio.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class SessionState:
    """Session state attributes for tracking user session variables."""
    user_id: str = "demo_user"
    selected_brand: str = "nike"
    current_season: str = "spring_summer"
    last_prompt: str = ""
    last_generated_image: Optional[Any] = None
    generation_history: List[Dict[str, Any]] = field(default_factory=list)
    chat_history: List[List[str]] = field(default_factory=list)

    def log_generation(self, prompt: str, image: Any, meta: Dict[str, Any]) -> None:
        """Record generated fashion design output details."""
        self.generation_history.append({
            "timestamp": time.time(),
            "prompt": prompt,
            "image": image,
            "metadata": meta
        })
        self.last_prompt = prompt
        self.last_generated_image = image

    def reset(self) -> None:
        """Reset session history state variables."""
        self.last_prompt = ""
        self.last_generated_image = None
        self.generation_history.clear()
        self.chat_history.clear()
