"""Quick start — ask one question about the current screen.

Usage:
    python -m examples.ask "what's on this page?"
"""

from __future__ import annotations

import asyncio

from lookout.agent import Agent
from lookout.capture import ScreenCapture
from lookout.config import load_settings
from lookout.controller import ScreenController
from lookout.memory import ScreenMemory
from lookout.vision import VisionError, build_vision


async def main() -> None:
    settings = load_settings()
    vision = build_vision(settings)
    try:
        async with ScreenCapture(settings) as cap:
            controller = ScreenController(settings)
            memory = ScreenMemory(settings)
            agent = Agent(settings, vision, cap, controller, memory)

            question = "Describe what's on the screen in 2-3 sentences."
            try:
                answer = await agent.ask(question)
            except VisionError as exc:
                print(f"vision error: {exc}")
                return
            print("Q:", question)
            print("A:", answer)
    finally:
        await vision.aclose()


if __name__ == "__main__":
    asyncio.run(main())
