"""FastAPI web UI for Lookout.

A minimal chat interface where each turn is a fresh screenshot + question.
Streams the latest frame as a base64 data URL so you can see what the agent sees.

Run: lookout serve --port 8765
Then open: http://127.0.0.1:8765
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..agent import Agent
from ..capture import ScreenCapture, ScreenFrame
from ..config import load_settings
from ..controller import ScreenController
from ..memory import ScreenMemory
from ..vision import VisionError, build_vision

log = logging.getLogger(__name__)

_TEMPLATES = Path(__file__).parent / "templates"


class AskBody(BaseModel):
    question: str


def _frame_to_dataurl(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def create_app() -> FastAPI:
    settings = load_settings()

    # mss instances are not thread-safe; pin all capture work to one worker thread.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="lookout-capture")
    state: dict[str, Any] = {
        "settings": settings,
        "vision": None,
        "capture": None,
        "controller": None,
        "memory": None,
        "agent": None,
        "history": [],  # list of {role, content}
        "lock": asyncio.Lock(),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if state["capture"] is not None:
            await asyncio.get_running_loop().run_in_executor(executor, state["capture"].__exit__, None, None, None)
        if state["vision"] is not None:
            await state["vision"].aclose()
        executor.shutdown(wait=False)

    app = FastAPI(title="Lookout", version="0.1.0", lifespan=lifespan)

    async def ensure_agent() -> Agent:
        async with state["lock"]:
            if state["agent"] is None:
                loop = asyncio.get_running_loop()
                vision = build_vision(settings)
                capture = ScreenCapture(settings)
                await loop.run_in_executor(executor, capture.__enter__)
                controller = ScreenController(settings)
                memory = ScreenMemory(settings)
                state.update(
                    vision=vision,
                    capture=capture,
                    controller=controller,
                    memory=memory,
                )
                state["agent"] = Agent(settings, vision, capture, controller, memory, sink=None)
            return state["agent"]

    async def grab() -> ScreenFrame:
        await ensure_agent()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, state["capture"].grab)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((_TEMPLATES / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return {
            "backend": settings.vision_backend,
            "ollama_url": settings.ollama_url,
            "ollama_model": settings.ollama_model,
            "anthropic_model": settings.anthropic_model,
            "openai_model": settings.openai_model,
            "gemini_model": settings.gemini_model,
            "has_cloud": settings.has_cloud_credentials(),
            "storage_dir": str(settings.storage_dir),
        }

    @app.get("/api/frame")
    async def frame() -> dict[str, Any]:
        f = await grab()
        return {
            "image": _frame_to_dataurl(f.raw_bytes),
            "width": f.width,
            "height": f.height,
            "timestamp": f.timestamp,
        }

    @app.post("/api/ask")
    async def ask(body: AskBody) -> dict[str, Any]:
        agent = await ensure_agent()
        try:
            text = await agent.ask(body.question)
        except VisionError as exc:
            return {"error": str(exc)}
        state["history"].append({"role": "user", "content": body.question})
        state["history"].append({"role": "assistant", "content": text})
        # Display the exact frame the model saw.
        f = agent.last_frame
        return {
            "answer": text,
            "image": _frame_to_dataurl(f.raw_bytes) if f else None,
            "timestamp": f.timestamp if f else time.time(),
        }

    @app.websocket("/ws")
    async def ws(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                msg = await ws.receive_text()
                data = json.loads(msg)
                kind = data.get("kind")
                if kind == "ask":
                    agent = await ensure_agent()
                    try:
                        text = await agent.ask(data["question"])
                    except VisionError as exc:
                        await ws.send_json({"type": "error", "error": str(exc)})
                        continue
                    f = agent.last_frame
                    if f is not None:
                        await ws.send_json(
                            {
                                "type": "frame",
                                "image": _frame_to_dataurl(f.raw_bytes),
                                "timestamp": f.timestamp,
                            }
                        )
                    await ws.send_json({"type": "answer", "text": text})
                elif kind == "ping":
                    await ws.send_json({"type": "pong", "ts": time.time()})
        except WebSocketDisconnect:
            return

    return app


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    run_server()
