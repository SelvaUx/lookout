"""Screen capture module.

Wraps `mss` for fast cross-platform screen grabbing, with JPEG compression
and downscaling to keep payloads small for vision models.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mss
import mss.tools
from PIL import Image

from .config import Settings


@dataclass
class ScreenFrame:
    """A captured screen frame."""

    image: Image.Image
    raw_bytes: bytes
    timestamp: float
    width: int
    height: int
    monitor: int

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.raw_bytes)


class ScreenCapture:
    """Cross-platform screen capture using mss."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._sct: mss.mss | None = None

    def __enter__(self) -> "ScreenCapture":
        self._sct = mss.mss()
        return self

    def __exit__(self, *exc) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    async def __aenter__(self) -> "ScreenCapture":
        return self.__enter__()

    async def __aexit__(self, *exc) -> None:
        self.__exit__(*exc)

    def _ensure(self) -> mss.mss:
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def list_monitors(self) -> list[dict]:
        """List available monitors."""
        sct = self._ensure()
        return list(sct.monitors)

    def grab(self, monitor: int | None = None) -> ScreenFrame:
        """Grab a single frame from the given monitor (1-based, 0 = all)."""
        sct = self._ensure()
        mon_idx = monitor if monitor is not None else self.settings.capture_monitor
        if mon_idx < 0 or mon_idx >= len(sct.monitors):
            mon_idx = 1  # fall back to primary
        shot = sct.grab(sct.monitors[mon_idx])
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        img = self._downscale(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.settings.capture_quality, optimize=True)
        return ScreenFrame(
            image=img,
            raw_bytes=buf.getvalue(),
            timestamp=time.time(),
            width=img.width,
            height=img.height,
            monitor=mon_idx,
        )

    def _downscale(self, img: Image.Image) -> Image.Image:
        max_side = self.settings.max_image_side
        w, h = img.size
        longest = max(w, h)
        if longest <= max_side:
            return img
        scale = max_side / longest
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return img.resize(new_size, Image.LANCZOS)

    def stream(self, interval: float | None = None) -> Iterator[ScreenFrame]:
        """Yield ScreenFrames at a fixed interval (seconds)."""
        period = interval if interval is not None else self.settings.capture_interval
        while True:
            start = time.time()
            yield self.grab()
            elapsed = time.time() - start
            sleep_for = max(0.0, period - elapsed)
            if sleep_for:
                time.sleep(sleep_for)
