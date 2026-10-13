"""Mouse + keyboard controller for the act mode.

Wraps pyautogui and pynput. Provides a small safety layer: dry-run mode,
optional pause-before-action, and a global hotkey to abort.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class ActionResult:
    ok: bool
    detail: str = ""


class ScreenController:
    """Drives the mouse and keyboard. Cross-platform via pyautogui."""

    def __init__(self, settings: Settings, *, dry_run: bool = False) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self._pyautogui: Any | None = None
        self._abort_key = "ctrl+shift+esc"

    def _pg(self) -> Any:
        if self._pyautogui is None:
            import pyautogui  # type: ignore[import-not-found]

            pyautogui.FAILSAFE = True
            self._pyautogui = pyautogui
        return self._pyautogui

    # ------------------------------------------------------------------
    # primitives
    # ------------------------------------------------------------------

    def size(self) -> tuple[int, int]:
        return tuple(self._pg().size())  # type: ignore[return-value]

    def move_to(self, x: int, y: int, duration: float = 0.25) -> ActionResult:
        if self.dry_run:
            return ActionResult(True, f"[dry-run] move_to({x},{y})")
        self._pg().moveTo(x, y, duration=duration)
        return ActionResult(True, f"moved to ({x},{y})")

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        *,
        button: str = "left",
        clicks: int = 1,
        duration: float = 0.1,
    ) -> ActionResult:
        if self.dry_run:
            return ActionResult(True, f"[dry-run] click({x},{y},{button})")
        if x is not None and y is not None:
            self._pg().click(x=x, y=y, button=button, clicks=clicks, duration=duration)
        else:
            self._pg().click(button=button, clicks=clicks, duration=duration)
        return ActionResult(True, f"clicked {button} at ({x},{y})")

    def double_click(self, x: int, y: int) -> ActionResult:
        if self.dry_run:
            return ActionResult(True, f"[dry-run] double_click({x},{y})")
        self._pg().doubleClick(x, y)
        return ActionResult(True, f"double-clicked at ({x},{y})")

    def right_click(self, x: int, y: int) -> ActionResult:
        return self.click(x, y, button="right")

    def type_text(self, text: str, interval: float = 0.02) -> ActionResult:
        if self.dry_run:
            return ActionResult(True, f"[dry-run] type_text({text!r})")
        # pyautogui.write doesn't handle unicode well; use keyboard for safety
        try:
            import pyperclip  # type: ignore[import-not-found]

            pyperclip.copy(text)
            self._pg().hotkey("command" if self._is_mac() else "ctrl", "v")
        except Exception:
            self._pg().write(text, interval=interval)
        return ActionResult(True, f"typed {len(text)} chars")

    def hotkey(self, *keys: str) -> ActionResult:
        if self.dry_run:
            return ActionResult(True, f"[dry-run] hotkey({','.join(keys)})")
        self._pg().hotkey(*keys)
        return ActionResult(True, f"pressed {'+'.join(keys)}")

    def scroll(self, amount: int, x: int | None = None, y: int | None = None) -> ActionResult:
        if self.dry_run:
            return ActionResult(True, f"[dry-run] scroll({amount})")
        if x is not None and y is not None:
            self._pg().scroll(amount, x=x, y=y)
        else:
            self._pg().scroll(amount)
        return ActionResult(True, f"scrolled {amount}")

    def screenshot(self) -> Any:
        return self._pg().screenshot()

    # ------------------------------------------------------------------
    # safety
    # ------------------------------------------------------------------

    def _is_mac(self) -> bool:
        import platform

        return platform.system() == "Darwin"

    def start_abort_listener(self, on_abort) -> None:
        """Listen for the global abort hotkey; calls on_abort() on the event loop.

        pynput invokes its callbacks from a background thread, so we marshal the
        callback back onto the running asyncio loop to keep it thread-safe.
        """
        try:
            from pynput import keyboard  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            log.warning("pynput unavailable, abort hotkey disabled: %s", exc)
            return

        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def _dispatch() -> None:
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(on_abort)
            else:
                on_abort()

        # simple: ESC always aborts
        def on_press(key):
            if key == keyboard.Key.esc:
                _dispatch()
                return False

        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()

    async def safe_execute(self, op, *args, **kwargs) -> ActionResult:
        """Optional pause before executing a risky action."""
        if self.settings.act_safety_pause and not self.dry_run:
            await asyncio.sleep(self.settings.act_step_pause)
        return op(*args, **kwargs)
