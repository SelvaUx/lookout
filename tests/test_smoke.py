"""Smoke tests for Lookout.

Run with:  python -m tests.test_smoke
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

# allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows consoles default to cp1252 and cannot print the check marks below.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):  # pragma: no cover
    pass

from PIL import Image  # noqa: E402

from lookout.config import load_settings  # noqa: E402
from lookout.vision import (  # noqa: E402
    OllamaVision,
    _extract_json_block,
    _image_to_data_url,
)


def test_extract_json_block() -> None:
    # pure json
    assert _extract_json_block('{"a": 1}') == {"a": 1}
    # json inside prose
    assert _extract_json_block('hello {"a": 2} bye') == {"a": 2}
    # nothing
    assert _extract_json_block("just text") is None


def test_image_to_data_url() -> None:
    img = Image.new("RGB", (32, 32), (255, 0, 0))
    url = _image_to_data_url(img, quality=70)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > 100


def test_settings_default() -> None:
    s = load_settings()
    assert s.capture_interval > 0
    assert s.memory_size > 0
    assert s.vision_backend in {"ollama", "anthropic", "openai", "gemini"}


async def test_ollama_health_no_server() -> None:
    """Ollama health check should not crash when server is absent."""
    s = load_settings()
    v = OllamaVision(s)
    try:
        ok = await v.health()
    finally:
        await v.aclose()
    # We just want to confirm the call returns a bool, no exception.
    assert isinstance(ok, bool)


def main() -> None:
    test_extract_json_block()
    print("✓ extract_json_block")
    test_image_to_data_url()
    print("✓ image_to_data_url")
    test_settings_default()
    print("✓ settings default")
    asyncio.run(test_ollama_health_no_server())
    print("✓ ollama health (tolerates no server)")
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
