"""The agent loop — describes the screen, picks an action, executes, repeats.

Three modes:
  * ask   — single-turn Q&A about the current screen
  * watch — continuous monitoring with change detection
  * act   — multi-step plan + execute with verification
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from PIL import Image

from .capture import ScreenCapture, ScreenFrame
from .config import Settings
from .controller import ActionResult, ScreenController
from .memory import ScreenMemory
from .ocr import diff_against
from .vision import BaseVision, VisionError

log = logging.getLogger(__name__)


SYSTEM_PROMPT_GENERAL = (
    "You are Lookout, a screen-aware assistant. Be concise, observant, and concrete. "
    "When describing what is on screen, mention visible text, layout, errors, and any "
    "interactive elements (buttons, inputs, links)."
)


ASK_PROMPT = (
    "Look at the current screenshot. The user asks:\n\n"
    "{question}\n\n"
    "Answer concisely. If you can quote relevant on-screen text, do so. "
    "If the answer is unclear from the image, say so."
)


WATCH_PROMPT = (
    "You are watching the user's screen. The user has set up this monitor with the goal:\n\n"
    "{goal}\n\n"
    "Look at the latest screenshot. Compare it to the prior state described below and "
    "decide if the goal is achieved, if something changed, or if nothing relevant happened.\n\n"
    "Prior state:\n{prior}\n\n"
    "Respond strictly as JSON with this shape:\n"
    "{{\n"
    '  "status": "triggered" | "changed" | "idle",\n'
    '  "summary": "<one sentence about the current state>",\n'
    '  "reason": "<why you chose this status>",\n'
    '  "delta": "<short description of what changed vs prior state, or empty string>"\n'
    "}}"
)


ACT_PLAN_PROMPT = (
    "You control the user's computer. Look at the screenshot and decide the SINGLE next "
    "action to take toward the user's goal.\n\n"
    "Goal: {goal}\n\n"
    "Recent actions (most recent last):\n{history}\n\n"
    "Respond strictly as JSON with this shape and no extra text:\n"
    "{{\n"
    '  "thought": "<one short sentence on your reasoning>",\n'
    '  "action": "click" | "double_click" | "right_click" | "type" | "hotkey" | "scroll" | "wait" | "done",\n'
    '  "x": <int or null>,\n'
    '  "y": <int or null>,\n'
    '  "text": "<string or null>",\n'
    '  "keys": ["<list of key names or null>"],\n'
    '  "amount": <int scroll amount, or null>\n'
    "}}\n\n"
    "Pick coordinates that match the screenshot's pixel space ({w}x{h}). When finished, "
    "set action to 'done' with a thought explaining completion. If the goal is impossible "
    "from the current state, also return 'done' and explain why."
)


@dataclass
class AgentEvent:
    type: str  # "frame", "caption", "action", "result", "text", "error", "done"
    payload: Any = None
    timestamp: float = field(default_factory=lambda: __import__("time").time())


EventSink = Callable[[AgentEvent], Awaitable[None] | None]


class Agent:
    """The brain: combines capture, vision, memory, and control."""

    def __init__(
        self,
        settings: Settings,
        vision: BaseVision,
        capture: ScreenCapture,
        controller: ScreenController,
        memory: ScreenMemory,
        sink: EventSink | None = None,
    ) -> None:
        self.settings = settings
        self.vision = vision
        self.capture = capture
        self.controller = controller
        self.memory = memory
        self.sink = sink or (lambda e: None)
        self._abort = asyncio.Event()
        self.last_frame: ScreenFrame | None = None

    # ------------------------------------------------------------------
    # event helpers
    # ------------------------------------------------------------------

    async def emit(self, event: AgentEvent) -> None:
        result = self.sink(event)
        if asyncio.iscoroutine(result):
            await result

    def abort(self) -> None:
        self._abort.set()

    def reset_abort(self) -> None:
        self._abort.clear()

    # ------------------------------------------------------------------
    # ask
    # ------------------------------------------------------------------

    async def ask(self, question: str) -> str:
        frame = self.capture.grab()
        self.last_frame = frame
        await self.emit(AgentEvent("frame", frame.raw_bytes))
        prompt = ASK_PROMPT.format(question=question)
        try:
            resp = await self.vision.describe(
                frame.image,
                prompt,
                system=SYSTEM_PROMPT_GENERAL,
                max_tokens=600,
            )
        except VisionError as exc:
            await self.emit(AgentEvent("error", str(exc)))
            raise
        text = resp.text.strip()
        self.memory.add(frame.image, f"Q: {question}\nA: {text}")
        await self.emit(AgentEvent("text", text))
        return text

    # ------------------------------------------------------------------
    # watch
    # ------------------------------------------------------------------

    async def watch(self, goal: str, *, max_iterations: int | None = None) -> None:
        self.reset_abort()
        prev_image: Image.Image | None = None
        prev_caption = "(no prior frame)"
        triggered = False
        iteration = 0
        while not self._abort.is_set():
            if max_iterations is not None and iteration >= max_iterations:
                break
            iteration += 1
            frame = self.capture.grab()
            await self.emit(AgentEvent("frame", frame.raw_bytes))
            diff = diff_against(prev_image, frame.image) if prev_image else 1.0
            if prev_image is not None and diff < self.settings.watch_change_threshold:
                # nothing meaningful changed
                await asyncio.sleep(self.settings.capture_interval)
                continue
            prompt = WATCH_PROMPT.format(goal=goal, prior=prev_caption)
            try:
                resp = await self.vision.describe(
                    frame.image,
                    prompt,
                    system=SYSTEM_PROMPT_GENERAL,
                    want_json=True,
                    max_tokens=400,
                )
            except VisionError as exc:
                await self.emit(AgentEvent("error", str(exc)))
                await asyncio.sleep(self.settings.capture_interval)
                continue
            parsed = resp.parsed or {
                "status": "changed",
                "summary": resp.text,
                "reason": "",
                "delta": "",
            }
            await self.emit(AgentEvent("caption", parsed))
            self.memory.add(frame.image, json.dumps(parsed), diff=diff)
            prev_image = frame.image
            prev_caption = parsed.get("summary", "")
            if parsed.get("status") == "triggered":
                triggered = True
                await self.emit(AgentEvent("done", {"triggered": True, "summary": prev_caption}))
                return
            await asyncio.sleep(self.settings.capture_interval)
        await self.emit(AgentEvent("done", {"triggered": triggered}))

    # ------------------------------------------------------------------
    # act (agentic)
    # ------------------------------------------------------------------

    async def act(self, goal: str) -> str:
        self.reset_abort()
        history: list[dict[str, Any]] = []
        for step in range(self.settings.act_max_steps):
            if self._abort.is_set():
                await self.emit(AgentEvent("error", "aborted by user"))
                return "aborted"
            frame = self.capture.grab()
            await self.emit(AgentEvent("frame", frame.raw_bytes))
            history_text = "\n".join(
                f"- {h.get('action', '?')} {h.get('detail', '')}" for h in history[-5:]
            ) or "(no actions yet)"
            prompt = ACT_PLAN_PROMPT.format(
                goal=goal,
                history=history_text,
                w=frame.width,
                h=frame.height,
            )
            try:
                resp = await self.vision.describe(
                    frame.image,
                    prompt,
                    system=SYSTEM_PROMPT_GENERAL,
                    want_json=True,
                    max_tokens=400,
                )
            except VisionError as exc:
                await self.emit(AgentEvent("error", str(exc)))
                return f"vision error: {exc}"
            action = resp.parsed or self._fallback_action(resp.text)
            if not action:
                await self.emit(AgentEvent("error", "could not parse action"))
                return "could not parse action"
            await self.emit(AgentEvent("action", action))
            kind = action.get("action")
            if kind == "done":
                thought = action.get("thought", "")
                self.memory.add(frame.image, f"Goal: {goal}\nResult: {thought}")
                await self.emit(AgentEvent("done", {"thought": thought}))
                return thought or "done"
            result = self._execute_action(action)
            history.append({"action": kind, "detail": result.detail, "ok": result.ok})
            await self.emit(AgentEvent("result", {"action": kind, "result": result.detail, "ok": result.ok}))
            await asyncio.sleep(self.settings.act_step_pause)
        await self.emit(AgentEvent("error", "exceeded max steps"))
        return "exceeded max steps"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _fallback_action(self, text: str) -> dict[str, Any] | None:
        """Best-effort: pull a click coord from free-form text if JSON failed."""
        m = re.search(r"click[^0-9]*(\d{2,4})\D+(\d{2,4})", text, re.IGNORECASE)
        if m:
            return {
                "thought": "fallback parser",
                "action": "click",
                "x": int(m.group(1)),
                "y": int(m.group(2)),
                "text": None,
                "keys": None,
                "amount": None,
            }
        if "done" in text.lower():
            return {"thought": text, "action": "done", "x": None, "y": None, "text": None, "keys": None, "amount": None}
        return None

    def _execute_action(self, action: dict[str, Any]) -> ActionResult:
        kind = action.get("action")
        try:
            if kind == "click":
                return self.controller.click(int(action.get("x") or 0), int(action.get("y") or 0))
            if kind == "double_click":
                return self.controller.double_click(int(action.get("x") or 0), int(action.get("y") or 0))
            if kind == "right_click":
                return self.controller.right_click(int(action.get("x") or 0), int(action.get("y") or 0))
            if kind == "type":
                return self.controller.type_text(str(action.get("text") or ""))
            if kind == "hotkey":
                keys = action.get("keys") or []
                if isinstance(keys, str):
                    keys = [k.strip() for k in keys.split("+") if k.strip()]
                if not keys:
                    return ActionResult(False, "hotkey with no keys")
                return self.controller.hotkey(*keys)
            if kind == "scroll":
                return self.controller.scroll(int(action.get("amount") or 0))
            if kind == "wait":
                return ActionResult(True, "wait")
            return ActionResult(False, f"unknown action: {kind}")
        except Exception as exc:
            return ActionResult(False, f"{kind} failed: {exc}")
