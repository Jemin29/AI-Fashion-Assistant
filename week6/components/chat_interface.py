"""Week 6 Chat Interface Component."""
from __future__ import annotations
from typing import Callable
import gradio as gr


def build_chat_interface(
    fn: Callable,
    title: str = "Conversational Assistant",
    placeholder: str = "Ask me anything...",
) -> gr.ChatInterface:
    """Build a styled gr.ChatInterface with standard configurations."""
    return gr.ChatInterface(
        fn=fn,
        title=None,
        description=None,
        textbox=gr.Textbox(placeholder=placeholder, container=False, scale=7),
    )
