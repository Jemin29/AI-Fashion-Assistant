"""
week2/prompts/prompt_validator.py
====================================
Validates and sanitises prompts before sending to SDXL.

Checks
------
- Empty / None prompt
- Token length within CLIP limit (77 tokens)
- Blocked words / NSFW content
- Minimum meaningful length
- Encoding safety (ASCII-fallback for problematic characters)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


# =============================================================================
# ── Blocked word lists
# =============================================================================

_NSFW_PATTERNS: List[str] = [
    r"\bnude\b", r"\bnaked\b", r"\bnsfw\b", r"\bexplicit\b",
    r"\bpornograph\w*\b", r"\berotic\b",
]

_INJECTION_PATTERNS: List[str] = [
    r"<[^>]{1,200}>",              # Any angle-bracket tag, including <lora:name:1.0>
    r"\{[^}]{0,100}\}",           # Brace syntax
    r"\([^)]{0,200}\:\d",         # Attention weight syntax (word:1.2)
]


# =============================================================================
# ── Validation Result
# =============================================================================

@dataclass
class ValidationResult:
    """
    Result of PromptValidator.validate().

    Attributes
    ----------
    is_valid : bool
    sanitized_prompt : str   The cleaned prompt (may differ from input).
    warnings : list of str   Non-fatal issues detected.
    errors : list of str     Fatal issues that block generation.
    token_estimate : int     Rough token count for the sanitised prompt.
    """
    is_valid:           bool
    sanitized_prompt:   str
    warnings:           List[str] = field(default_factory=list)
    errors:             List[str] = field(default_factory=list)
    token_estimate:     int       = 0

    def __bool__(self) -> bool:
        return self.is_valid


# =============================================================================
# ── Validator
# =============================================================================

class PromptValidator:
    """
    Validate and sanitise a text prompt for SDXL generation.

    Parameters
    ----------
    max_tokens : int     Maximum allowed CLIP tokens (default 77).
    min_chars : int      Minimum meaningful prompt length.
    block_nsfw : bool    Reject prompts containing NSFW patterns.
    strip_injection : bool  Remove LoRA/attention weight syntax.

    Example
    -------
        v = PromptValidator()
        result = v.validate("A slim-fit navy blazer, photorealistic")
        if result:
            generate(result.sanitized_prompt)
        else:
            print(result.errors)
    """

    _TOKENS_PER_WORD: float = 1.3

    def __init__(
        self,
        max_tokens:       int  = 77,
        min_chars:        int  = 5,
        block_nsfw:       bool = True,
        strip_injection:  bool = True,
    ) -> None:
        self._max_tokens      = max_tokens
        self._min_chars       = min_chars
        self._block_nsfw      = block_nsfw
        self._strip_injection = strip_injection

        # Compile patterns once
        self._nsfw_re = [re.compile(p, re.I) for p in _NSFW_PATTERNS]
        self._inj_re  = [re.compile(p, re.I) for p in _INJECTION_PATTERNS]

    def validate(self, prompt: Optional[str]) -> ValidationResult:
        """
        Validate and sanitise a single prompt string.

        Parameters
        ----------
        prompt : str or None

        Returns
        -------
        ValidationResult
        """
        errors:   List[str] = []
        warnings: List[str] = []

        # ── Guard: None / empty ───────────────────────────────────────────
        if not prompt:
            return ValidationResult(
                is_valid         = False,
                sanitized_prompt = "",
                errors           = ["Prompt is empty or None."],
            )

        text = str(prompt)

        # ── Injection stripping ───────────────────────────────────────────
        if self._strip_injection:
            for pat in self._inj_re:
                if pat.search(text):
                    warnings.append(f"Removed injection syntax: {pat.pattern!r}")
                    text = pat.sub("", text)

        # ── Whitespace normalisation ──────────────────────────────────────
        text = re.sub(r"\s+", " ", text).strip()

        # ── Minimum length ────────────────────────────────────────────────
        if len(text) < self._min_chars:
            errors.append(
                f"Prompt too short ({len(text)} chars, minimum {self._min_chars})."
            )

        # ── NSFW check ────────────────────────────────────────────────────
        if self._block_nsfw:
            for pat in self._nsfw_re:
                if pat.search(text):
                    errors.append(f"Prompt contains blocked content: {pat.pattern!r}")

        # ── Token length ──────────────────────────────────────────────────
        token_est = self._estimate_tokens(text)
        if token_est > self._max_tokens:
            warnings.append(
                f"Prompt may exceed CLIP token limit "
                f"(estimated {token_est} tokens, limit {self._max_tokens}). "
                f"Some text will be truncated by the tokenizer."
            )

        # ── Encoding safety ───────────────────────────────────────────────
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            warnings.append(f"Encoding issue detected: {exc}. Replacing bad chars.")
            text = text.encode("utf-8", errors="replace").decode("utf-8")

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning("Prompt validation FAILED | errors={}", errors)
        elif warnings:
            logger.debug("Prompt validation OK with warnings | {}", warnings)
        else:
            logger.debug("Prompt validation OK | tokens≈{}", token_est)

        return ValidationResult(
            is_valid         = is_valid,
            sanitized_prompt = text,
            warnings         = warnings,
            errors           = errors,
            token_estimate   = token_est,
        )

    def validate_pair(
        self,
        prompt:          Optional[str],
        negative_prompt: Optional[str],
    ) -> tuple[ValidationResult, ValidationResult]:
        """
        Validate both positive and negative prompts.

        Returns
        -------
        (positive_result, negative_result)
        """
        pos = self.validate(prompt)
        # Negative prompt is optional — treat empty as valid
        if not negative_prompt:
            neg = ValidationResult(
                is_valid         = True,
                sanitized_prompt = "",
                token_estimate   = 0,
            )
        else:
            neg = self.validate(negative_prompt)
        return pos, neg

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * self._TOKENS_PER_WORD)
