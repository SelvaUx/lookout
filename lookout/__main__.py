"""CLI entry point for Lookout.

Examples
--------
  # Ask a one-shot question about the screen
  lookout ask "what's the error in this terminal?"

  # Watch the screen for a condition
  lookout watch --goal "alert me when the build finishes"

  # Let the agent act on your screen
  lookout act "open Chrome and search for flights from NYC to SF next Friday"

  # Start the web UI on http://127.0.0.1:8765
  lookout serve
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .agent import Agent
from .capture import ScreenCapture
from .config import load_settings
from .controller import ScreenController
from .memory import ScreenMemory
from .vision import VisionError, build_vision


def _print_event(event) -> None:
    ts = event.timestamp
    if event.type == "frame":
        print(f"[{ts:.1f}] frame captured", file=sys.stderr)
    elif event.type == "text":
        print(event.payload)
    elif event.type == "caption":
        print(f"[watch] {json.dumps(event.payload)}", file=sys.stderr)
    elif event.type == "action":
        print(f"[act]   {json.dumps(event.payload)}", file=sys.stderr)
    elif event.type == "result":
        print(f"[act]   -> {event.payload}", file=sys.stderr)
    elif event.type == "error":
        print(f"[err]   {event.payload}", file=sys.stderr)
    elif event.type == "done":
        print(f"[done]  {json.dumps(event.payload)}", file=sys.stderr)


async def _run_ask(args: argparse.Namespace) -> int:
    settings = load_settings()
    vision = build_vision(settings)
    try:
        async with ScreenCapture(settings) as cap:
            controller = ScreenController(settings, dry_run=args.dry_run)
            memory = ScreenMemory(settings)
            agent = Agent(settings, vision, cap, controller, memory, sink=_print_event)
            try:
                await agent.ask(args.question)
            except VisionError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            return 0
    finally:
        await vision.aclose()


async def _run_watch(args: argparse.Namespace) -> int:
    settings = load_settings()
    vision = build_vision(settings)
    try:
        async with ScreenCapture(settings) as cap:
            controller = ScreenController(settings)
            memory = ScreenMemory(settings)
            agent = Agent(settings, vision, cap, controller, memory, sink=_print_event)
            try:
                await agent.watch(args.goal, max_iterations=args.max_iterations)
            except VisionError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            return 0
    finally:
        await vision.aclose()


async def _run_act(args: argparse.Namespace) -> int:
    settings = load_settings()
    vision = build_vision(settings)
    try:
        async with ScreenCapture(settings) as cap:
            controller = ScreenController(settings, dry_run=args.dry_run)
            memory = ScreenMemory(settings)
            agent = Agent(settings, vision, cap, controller, memory, sink=_print_event)
            controller.start_abort_listener(agent.abort)
            try:
                result = await agent.act(args.goal)
            except VisionError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print(result)
            return 0
    finally:
        await vision.aclose()


def _run_serve(args: argparse.Namespace) -> int:
    from .ui.app import run_server  # local import to avoid pulling fastapi unless needed

    run_server(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lookout",
        description="Watch your screen, ask questions, and act on what you see.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="enable debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("ask", help="ask a one-shot question about the current screen")
    pa.add_argument("question", help="the question to ask")
    pa.add_argument("--dry-run", action="store_true", help="do not act on the screen")
    pa.set_defaults(func=_run_ask)

    pw = sub.add_parser("watch", help="continuously watch the screen and alert on a condition")
    pw.add_argument("--goal", "-g", required=True, help="what to look for / when to alert")
    pw.add_argument("--max-iterations", type=int, default=None, help="stop after N ticks (testing)")
    pw.set_defaults(func=_run_watch)

    pac = sub.add_parser("act", help="agentically perform a task by clicking and typing")
    pac.add_argument("goal", help="the task to accomplish")
    pac.add_argument("--dry-run", action="store_true", help="plan but do not act")
    pac.set_defaults(func=_run_act)

    ps = sub.add_parser("serve", help="start the web UI")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8765)
    ps.set_defaults(func=lambda a: _run_serve(a))

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.cmd in {"ask", "watch", "act"}:
        return asyncio.run(args.func(args))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
