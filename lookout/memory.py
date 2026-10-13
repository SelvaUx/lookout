"""Rolling screen memory: keep the last N frames + their captions for follow-up Q&A."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque

from PIL import Image

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    timestamp: float
    caption: str
    path: str
    width: int
    height: int
    diff: float = 0.0


class ScreenMemory:
    """Stores the most recent N frames with their captions on disk."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.entries: Deque[MemoryEntry] = deque(maxlen=settings.memory_size)
        self._mem_dir = settings.storage_dir / "memory"
        self._mem_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = settings.storage_dir / "memory" / "log.jsonl"

    def add(self, image: Image.Image, caption: str, diff: float = 0.0) -> MemoryEntry:
        ts = time.time()
        fname = f"frame_{int(ts*1000)}.jpg"
        path = self._mem_dir / fname
        image.save(path, format="JPEG", quality=70, optimize=True)
        entry = MemoryEntry(
            timestamp=ts,
            caption=caption,
            path=str(path),
            width=image.width,
            height=image.height,
            diff=diff,
        )
        self.entries.append(entry)
        self._append_log(entry)
        return entry

    def _append_log(self, entry: MemoryEntry) -> None:
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except OSError as exc:  # pragma: no cover
            log.warning("could not write memory log: %s", exc)

    def recent_captions(self, n: int = 5) -> list[str]:
        out: list[str] = []
        for entry in list(self.entries)[-n:]:
            stamp = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            out.append(f"[{stamp}] {entry.caption}")
        return out

    def clear(self) -> None:
        for entry in list(self.entries):
            try:
                Path(entry.path).unlink(missing_ok=True)
            except OSError:
                pass
        self.entries.clear()
        if self._log_path.exists():
            self._log_path.unlink()
