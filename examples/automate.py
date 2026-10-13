"""Quick start — let the agent do something on your screen.

Usage:
    python -m examples.automate "open Chrome and search for flights NYC to SF"
    python -m examples.automate --dry-run "click the search button"
"""

from __future__ import annotations

import argparse
import asyncio

from lookout.agent import Agent
from lookout.capture import ScreenCapture
from lookout.config import load_settings
from lookout.controller import ScreenController
from lookout.memory import ScreenMemory
from lookout.vision import VisionError, build_vision


async def main(goal: str, dry_run: bool) -> None:
    settings = load_settings()
    vision = build_vision(settings)
    try:
        async with ScreenCapture(settings) as cap:
            controller = ScreenController(settings, dry_run=dry_run)
            memory = ScreenMemory(settings)
            agent = Agent(
                settings,
                vision,
                cap,
                controller,
                memory,
                sink=lambda e: print(f"[{e.type}] {e.payload}", flush=True),
            )
            controller.start_abort_listener(agent.abort)
            try:
                result = await agent.act(goal)
            except VisionError as exc:
                print(f"vision error: {exc}")
                return
            print("result:", result)
    finally:
        await vision.aclose()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("goal")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        asyncio.run(main(args.goal, args.dry_run))
    except KeyboardInterrupt:
        print("\naborted")
