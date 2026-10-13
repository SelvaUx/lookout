"""Configuration for Lookout.

All settings are overridable via environment variables (prefix LOOKOUT_) or a .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

VisionBackend = Literal["ollama", "anthropic", "openai", "gemini"]

_ENV_CANDIDATES = (
    Path.cwd() / ".env",
    Path(__file__).resolve().parents[1] / ".env",
)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    raw = os.environ.get(key)
    return default if raw is None else raw


def _fallback_dotenv(path: Path) -> None:
    """Minimal .env parser used when python-dotenv is not installed."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_dotenv_once() -> None:
    """Load the first .env found in the cwd or repo root (real env wins)."""
    env_path = next((p for p in _ENV_CANDIDATES if p.is_file()), None)
    if env_path is None:
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv(env_path, override=False)
    except Exception:
        _fallback_dotenv(env_path)


@dataclass
class Settings:
    # ---- vision backend ----
    vision_backend: VisionBackend = field(
        default_factory=lambda: _env_str("LOOKOUT_BACKEND", "ollama")  # type: ignore[arg-type]
    )

    # ---- ollama (local, free, default) ----
    ollama_url: str = field(default_factory=lambda: _env_str("LOOKOUT_OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env_str("LOOKOUT_OLLAMA_MODEL", "llama3.2-vision"))

    # ---- anthropic ----
    anthropic_api_key: str = field(default_factory=lambda: _env_str("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(
        default_factory=lambda: _env_str("LOOKOUT_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    )

    # ---- openai ----
    openai_api_key: str = field(default_factory=lambda: _env_str("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env_str("LOOKOUT_OPENAI_MODEL", "gpt-4o"))

    # ---- gemini ----
    google_api_key: str = field(default_factory=lambda: _env_str("GOOGLE_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _env_str("LOOKOUT_GEMINI_MODEL", "gemini-2.0-flash"))
    gemini_url: str = field(
        default_factory=lambda: _env_str("LOOKOUT_GEMINI_URL", "https://generativelanguage.googleapis.com")
    )

    # ---- capture ----
    capture_interval: float = field(default_factory=lambda: _env_float("LOOKOUT_CAPTURE_INTERVAL", 2.0))
    capture_monitor: int = field(default_factory=lambda: _env_int("LOOKOUT_CAPTURE_MONITOR", 1))  # 1 = primary
    capture_quality: int = field(default_factory=lambda: _env_int("LOOKOUT_CAPTURE_QUALITY", 70))  # JPEG quality
    max_image_side: int = field(default_factory=lambda: _env_int("LOOKOUT_MAX_IMAGE_SIDE", 1280))

    # ---- memory ----
    memory_size: int = field(default_factory=lambda: _env_int("LOOKOUT_MEMORY_SIZE", 8))

    # ---- watch mode ----
    watch_change_threshold: float = field(default_factory=lambda: _env_float("LOOKOUT_WATCH_THRESHOLD", 0.06))
    watch_idle_window: float = field(default_factory=lambda: _env_float("LOOKOUT_WATCH_IDLE_WINDOW", 30.0))

    # ---- act mode ----
    act_max_steps: int = field(default_factory=lambda: _env_int("LOOKOUT_ACT_MAX_STEPS", 20))
    act_step_pause: float = field(default_factory=lambda: _env_float("LOOKOUT_ACT_STEP_PAUSE", 0.6))
    act_safety_pause: bool = field(default_factory=lambda: _env_bool("LOOKOUT_ACT_SAFETY_PAUSE", True))

    # ---- storage ----
    storage_dir: Path = field(default_factory=lambda: Path.home() / ".lookout")

    def ensure_dirs(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "memory").mkdir(exist_ok=True)
        (self.storage_dir / "logs").mkdir(exist_ok=True)

    def has_cloud_credentials(self) -> bool:
        return bool(
            self.anthropic_api_key
            or self.openai_api_key
            or self.google_api_key
        )


def load_settings() -> Settings:
    """Construct a Settings instance, honouring env vars and a .env file if present."""
    load_dotenv_once()
    settings = Settings()
    settings.ensure_dirs()
    return settings
