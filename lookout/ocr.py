"""Lightweight OCR helper for fast text extraction.

Used as a cheap pre-filter before sending screenshots to a vision model.
When Tesseract is unavailable we degrade gracefully to a no-op.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageStat

log = logging.getLogger(__name__)


@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def extract_text(image: Image.Image, lang: str = "eng") -> list[TextBlock]:
    """Run Tesseract OCR and return text blocks with bounding boxes.

    Requires the `tesseract` binary on PATH. Returns [] if unavailable.
    """
    if not tesseract_available():
        log.warning("tesseract binary not found; OCR disabled")
        return []

    import pytesseract  # type: ignore[import-not-found]

    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:  # pragma: no cover - depends on env
        log.warning("tesseract failed: %s", exc)
        return []

    blocks: list[TextBlock] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf = float(data.get("conf", [0] * n)[i] or 0)
        if not text or conf < 30:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        blocks.append(TextBlock(text=text, confidence=conf / 100.0, bbox=(x, y, w, h)))
    return blocks


def extract_plain_text(image: Image.Image, lang: str = "eng") -> str:
    if not tesseract_available():
        return ""
    import pytesseract  # type: ignore[import-not-found]
    try:
        return pytesseract.image_to_string(image, lang=lang).strip()
    except Exception:  # pragma: no cover
        return ""


def diff_against(image_a: Image.Image, image_b: Image.Image) -> float:
    """Return a coarse 0..1 difference score between two images of equal size.

    Uses Pillow only (no numpy) so watch mode works with a minimal install.
    """
    if image_a.size != image_b.size:
        return 1.0
    # downsample to keep this fast
    a = image_a.convert("L").resize((128, 72))
    b = image_b.convert("L").resize((128, 72))
    diff = ImageChops.difference(a, b)
    stat = ImageStat.Stat(diff)
    mean = sum(stat.mean) / len(stat.mean)
    return float(mean / 255.0)


def _has_tesseract_subprocess() -> bool:  # pragma: no cover - debug helper
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=False, timeout=2)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
