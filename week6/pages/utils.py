"""Week 6 — Shared UI Utilities & Centralized Error Handler Decorator."""
from __future__ import annotations
import traceback
from typing import Any, Callable, List, Optional, Tuple
import gradio as gr
from loguru import logger


def safe_callback(num_outputs: int, fallback_values: Optional[List[Any]] = None) -> Callable:
    """Decorator to wrap Gradio callback functions with centralized error handling.

    Catches all exceptions, logs the full traceback internally, displays a clean,
    traceback-free warning to the user via gr.Warning, and returns a tuple of
    gr.update() to prevent UI component state corruption.
    """
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                # Log traceback internally
                tb = traceback.format_exc()
                logger.error(f"Error in callback '{fn.__name__}': {e}\n{tb}")
                
                # Determine user-facing error message
                if hasattr(e, "message") and e.message:
                    error_msg = e.message
                else:
                    error_msg = str(e)
                
                # Show clean warning toast (never expose full traceback details to client)
                gr.Warning(f"Studio Notice: {error_msg}")
                
                # Return list of updates to match expected output count exactly
                if fallback_values is not None:
                    vals = list(fallback_values)
                    if len(vals) < num_outputs:
                        vals += [gr.update()] * (num_outputs - len(vals))
                    return tuple(vals[:num_outputs]) if num_outputs > 1 else vals[0]
                else:
                    return tuple([gr.update()] * num_outputs) if num_outputs > 1 else gr.update()
        return wrapper
    return decorator
