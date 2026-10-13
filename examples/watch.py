"""Quick start — watch the screen and alert when something changes.

Usage:
    python -m examples.watch --goal "alert me when the CI build is green"
"""

from __future__ import annotations

import argparse
import asyncio
import json

from lookout.agent import Agent
from lookout.capture import ScreenCapture
from lookout.config import load_settings
from lookout.controller import ScreenController
from lookout.memory import ScreenMemory
from lookout.vision import VisionError, build_vision


async def main(goal: str) -> None:
    settings = load_settings()
    vision = build_vision(settings)
    try:
        async with ScreenCapture(settings) as cap:
            controller = ScreenController(settings)
            memory = ScreenMemory(settings)
            agent = Agent(
                settings,
                vision,
                cap,
                controller,
                memory,
                sink=lambda e: print(f"[{e.type}] {e.payload}", flush=True),
            )
            try:
                await agent.watch(goal)
            except VisionError as exc:
                print(f"vision error: {exc}")
    finally:
        await vision.aclose()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--goal", "-g", required=True)
    args = p.parse_args()
    try:
        asyncio.run(main(args.goal))
    except KeyboardInterrupt:
        print("\nstopped")
