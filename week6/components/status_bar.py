"""Week 6 Status Bar Component."""
from __future__ import annotations
import gradio as gr
import time


def build_status_bar(mock_mode: bool = True) -> None:
    """Build the system status bar footer component."""
    mode_text = "🟡 Mock Mode (CPU-friendly)" if mock_mode else "🟢 GPU Production Mode"
    gr.HTML(f"""
    <div class="studio-status-bar" style="background: #111118; border: 1px solid rgba(255, 255, 255, 0.05); padding: 0.8rem 1.5rem; display: flex; justify-content: space-between; align-items: center; border-radius: 6px; margin-top: 2rem; font-size: 0.85rem; color: #888;">
        <div>⚡ System Status: <span style="color: #ff9f43; font-weight: 500;">{mode_text}</span></div>
        <div>🌐 Environment: <span style="color: #3498db;">Development</span></div>
        <div>📅 Local Time: <span style="color: #9b59b6;">{time.strftime("%Y-%m-%d")}</span></div>
        <div>🏷️ Version: <span style="color: #2ecc71;">v1.0.0</span></div>
    </div>
    """)
