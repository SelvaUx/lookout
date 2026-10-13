"""Lookout — a screen-aware assistant that watches your screen and follows your commands.

Three modes:
  * ask  — one-shot Q&A about what's currently on screen
  * watch — continuous monitoring with alerts on change
  * act  — agentic: tell it what to do, it plans and clicks
"""

from .agent import Agent
from .capture import ScreenCapture, ScreenFrame
from .config import Settings, load_settings
from .controller import ScreenController
from .memory import ScreenMemory
from .vision import BaseVision, VisionError, build_vision

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Agent",
    "ScreenCapture",
    "ScreenFrame",
    "ScreenController",
    "ScreenMemory",
    "Settings",
    "load_settings",
    "build_vision",
    "BaseVision",
    "VisionError",
]
