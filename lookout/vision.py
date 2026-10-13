"""Vision model integration with multiple backends.

Supports:
  * ollama    — local, free (default; needs `ollama serve` running)
  * anthropic — Claude (best for computer use)
  * openai    — GPT-4o
  * gemini    — Google Gemini

All backends expose the same async `describe` API returning a `VisionResponse`.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image

from .config import Settings, VisionBackend

log = logging.getLogger(__name__)


@dataclass
class VisionResponse:
    text: str
    raw: dict[str, Any] | None = None
    parsed: dict[str, Any] | None = None  # when JSON was requested and parseable

    def as_dict(self) -> dict[str, Any]:
        if self.parsed is not None:
            return self.parsed
        return {"text": self.text}


class VisionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _image_to_data_url(image: Image.Image, quality: int = 80) -> str:
    import io

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Try to pull a JSON object out of a model response."""
    # find first { ... last }
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class BaseVision:
    name: VisionBackend

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def describe(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system: str | None = None,
        want_json: bool = False,
        max_tokens: int = 1024,
    ) -> VisionResponse:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class OllamaVision(BaseVision):
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def describe(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system: str | None = None,
        want_json: bool = False,
        max_tokens: int = 1024,
    ) -> VisionResponse:
        data_url = _image_to_data_url(image, quality=80)
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": prompt,
                "images": [data_url],
            }
        )
        payload: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if want_json:
            payload["format"] = "json"
        try:
            r = await self._http().post(
                f"{self.settings.ollama_url.rstrip('/')}/api/chat",
                json=payload,
            )
        except httpx.RequestError as exc:
            raise VisionError(
                f"Could not reach Ollama at {self.settings.ollama_url}. "
                "Is `ollama serve` running? Error: " + str(exc)
            ) from exc
        if r.status_code != 200:
            raise VisionError(f"Ollama error {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = (data.get("message") or {}).get("content", "")
        parsed = _extract_json_block(text) if want_json else None
        return VisionResponse(text=text, raw=data, parsed=parsed)

    async def health(self) -> bool:
        try:
            r = await self._http().get(f"{self.settings.ollama_url.rstrip('/')}/api/tags", timeout=3.0)
            return r.status_code == 200
        except httpx.RequestError:
            return False


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicVision(BaseVision):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.anthropic_api_key:
            raise VisionError("ANTHROPIC_API_KEY is not set")
        from anthropic import AsyncAnthropic  # type: ignore[import-not-found]

        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def aclose(self) -> None:
        await self._client.close()

    async def describe(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system: str | None = None,
        want_json: bool = False,
        max_tokens: int = 1024,
    ) -> VisionResponse:
        data_url = _image_to_data_url(image, quality=85)
        # strip the data: prefix to get the raw base64
        b64 = data_url.split(",", 1)[1]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        kwargs: dict[str, Any] = {
            "model": self.settings.anthropic_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise VisionError(f"Anthropic error: {exc}") from exc
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        text = "\n".join(parts)
        parsed = _extract_json_block(text) if want_json else None
        return VisionResponse(text=text, raw=resp.model_dump() if hasattr(resp, "model_dump") else None, parsed=parsed)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIVision(BaseVision):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.openai_api_key:
            raise VisionError("OPENAI_API_KEY is not set")
        from openai import AsyncOpenAI  # type: ignore[import-not-found]

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def aclose(self) -> None:
        await self._client.close()

    async def describe(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system: str | None = None,
        want_json: bool = False,
        max_tokens: int = 1024,
    ) -> VisionResponse:
        data_url = _image_to_data_url(image, quality=85)
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self.settings.openai_model,
                messages=msgs,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise VisionError(f"OpenAI error: {exc}") from exc
        text = (resp.choices[0].message.content or "")
        parsed = _extract_json_block(text) if want_json else None
        return VisionResponse(text=text, raw=resp.model_dump() if hasattr(resp, "model_dump") else None, parsed=parsed)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class GeminiVision(BaseVision):
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.google_api_key:
            raise VisionError("GOOGLE_API_KEY is not set")
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def describe(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system: str | None = None,
        want_json: bool = False,
        max_tokens: int = 1024,
    ) -> VisionResponse:
        b64 = _image_to_data_url(image, quality=85).split(",", 1)[1]
        parts: list[dict[str, Any]] = [
            {"text": (system + "\n\n" if system else "") + prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]
        generation_config: dict[str, Any] = {"maxOutputTokens": max_tokens}
        if want_json:
            generation_config["responseMimeType"] = "application/json"
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        url = (
            f"{self.settings.gemini_url.rstrip('/')}/v1beta/models/"
            f"{self.settings.gemini_model}:generateContent"
        )
        try:
            r = await self._http().post(url, headers={"x-goog-api-key": self.settings.google_api_key}, json=payload)
        except httpx.RequestError as exc:
            raise VisionError(f"Could not reach Gemini: {exc}") from exc
        if r.status_code != 200:
            body = r.text[:400]
            if "has not been used" in body or "PERMISSION_DENIED" in body:
                raise VisionError(
                    "Gemini API is not enabled for this Google Cloud project. "
                    "Enable 'Generative Language API' at "
                    "https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com "
                    f"then retry.\n{body}"
                )
            raise VisionError(f"Gemini error {r.status_code}: {body}")
        data = r.json()
        text = ""
        candidates = data.get("candidates") or []
        if candidates:
            for part in (candidates[0].get("content") or {}).get("parts", []) or []:
                if "text" in part:
                    text += part["text"]
        text = text.strip()
        parsed = _extract_json_block(text) if want_json else None
        if want_json and parsed is None and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pass
        return VisionResponse(text=text, raw=data, parsed=parsed)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def build_vision(settings: Settings) -> BaseVision:
    backend = settings.vision_backend
    if backend == "ollama":
        return OllamaVision(settings)
    if backend == "anthropic":
        return AnthropicVision(settings)
    if backend == "openai":
        return OpenAIVision(settings)
    if backend == "gemini":
        return GeminiVision(settings)
    raise VisionError(f"Unknown backend: {backend}")
