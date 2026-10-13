# 👀 Lookout

> Your screen-aware assistant. Watches your screen, answers questions about
> what it sees, alerts you when things change, and (with permission) clicks
> and types to do things for you.

Lookout is a local-first "computer use" agent you can actually read and modify.
It pairs a vision model (Ollama by default — free & private — or Gemini /
OpenAI / Claude in the cloud) with mouse + keyboard control, so you can have a
conversation with your desktop.

```
+----------+        +---------+        +-------------------+
|  screen  |  --->  | capture |  --->  | vision model      |
+----------+        +---------+        | (Ollama/Gemini/   |
                                       |  GPT-4o/Claude)   |
                                       +---------+---------+
                                                 |
                              +------------------+------------------+
                              |                  |                  |
                         +----v-----+      +-----v-----+      +-----v-----+
                         |  ask     |      |  watch    |      |  act      |
                         | one Q    |      | monitor   |      | plan+do   |
                         +----+-----+      +-----+-----+      +-----+-----+
                              |                 |                  |
                              +--------+--------+--------+---------+
                                       |                 |
                                  +----v----+      +-----v-----+
                                  | answer  |      | pyautogui |
                                  +---------+      +-----------+
```

## What you can do with it

| Mode    | What it does                                  | Example                                                |
|---------|-----------------------------------------------|--------------------------------------------------------|
| `ask`   | One-shot Q&A about the current screen         | `lookout ask "what's the error in this terminal?"`     |
| `watch` | Continuously monitor; alert on a condition    | `lookout watch --goal "tell me when the build is green"` |
| `act`   | Agentically plan + click + type to do a task  | `lookout act "open Chrome, search for 'flights NYC to SF'"` |
| `serve` | Web chat UI on `http://127.0.0.1:8765`        | `lookout serve`                                        |

---

## Quick start

### Windows — one click

1. Double-click **`setup.bat`**
   - creates a virtual environment,
   - installs Lookout and its dependencies,
   - lets you pick a vision model (or a cloud backend),
   - writes a tuned `.env`.
2. Double-click **`start.bat`**
   - launches the web UI and opens `http://127.0.0.1:8765`.

That's it — ask questions in the browser. Close the window to stop.

### Any platform — manual

```bash
# 1. clone or unzip, then:
cd lookout
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # installs Lookout + the `lookout` command
# (or `pip install -r requirements.txt` to skip the console script)

# 2. pick a vision backend (see below), e.g. the free local default:
ollama pull moondream              # small & CPU-friendly

# 3. run
lookout ask "summarize what's on my screen"
lookout serve
```

---

## Choosing a vision model (important on low-end PCs)

Lookout sends each screenshot to a vision model. On a **16 GB RAM, no-GPU**
machine (e.g. an i5 6th gen), pick a small local model or use the cloud:

| Local model        | Size    | Notes                                          |
|--------------------|---------|------------------------------------------------|
| `moondream`        | ~1.7 GB | lightest; runs on CPU; best starting point     |
| `gemma3:4b`        | ~3.3 GB | better quality, noticeably slower on CPU       |
| `qwen2.5vl:3b`     | ~3.2 GB | strong at reading on-screen text               |
| `llama3.2-vision`  | ~7.9 GB | good quality but **very slow without a GPU**   |

Cloud backends (`gemini`, `openai`, `anthropic`) need **no local GPU or RAM** —
just an API key, and they're much faster for `act` mode (Gemini has a free tier).

> Rule of thumb: use a **local** model for private `ask` / `watch`; use a
> **cloud** model when you want speed or reliable `act` (multi-step) runs.

---

## Cloud backend setup

Set one backend and its key (copy `.env.example` → `.env` and edit, or use env vars):

```bash
# Gemini (free tier available)
LOOKOUT_BACKEND=gemini
GOOGLE_API_KEY=...
LOOKOUT_GEMINI_MODEL=gemini-2.0-flash

# OpenAI
LOOKOUT_BACKEND=openai
OPENAI_API_KEY=sk-...
LOOKOUT_OPENAI_MODEL=gpt-4o

# Anthropic
LOOKOUT_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
LOOKOUT_ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

Gemini and Ollama work over plain HTTP (via `httpx`) — **no extra SDK needed**.
OpenAI and Anthropic need their SDKs installed:

```bash
pip install "lookout[openai]"      # or [anthropic], or [all]
```

> **Gemini:** you must enable the *Generative Language API* for your Google
> Cloud project once:
> https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
> (Then wait ~1 minute before the first request.)

---

## Usage

```bash
# 1. one-shot question
lookout ask "summarize what's on my screen"

# 2. watch for a condition (Ctrl+C or ESC to stop)
lookout watch --goal "alert me when the CI build is green"

# 3. act on your screen (move mouse to a screen corner, or press ESC, to abort)
lookout act "open Slack and message @alex: 'running 5 min late'"
lookout act --dry-run "click the search button"   # plan without moving the mouse

# 4. the web UI
lookout serve                 # http://127.0.0.1:8765
lookout serve --host 0.0.0.0 --port 9000
```

Add `-v` / `--verbose` for debug logging.

### Without the CLI

```bash
python -m examples.ask
python -m examples.watch --goal "alert me when the deploy finishes"
python -m examples.automate "open Notion and create a page titled 'Today'"
python -m examples.automate --dry-run "click the search button"
```

### Programmatic use

```python
import asyncio
from lookout import Agent, ScreenCapture, ScreenController, ScreenMemory, build_vision, load_settings

async def main():
    settings = load_settings()
    vision = build_vision(settings)
    try:
        with ScreenCapture(settings) as cap:
            agent = Agent(settings, vision, cap, ScreenController(settings), ScreenMemory(settings))
            print(await agent.ask("what is on screen?"))
    finally:
        await vision.aclose()

asyncio.run(main())
```

---

## Configuration

All settings are environment variables (prefix `LOOKOUT_`) or entries in a
`.env` file next to the project. See `.env.example` for the full list.

| Variable                    | Default                          | What it does                                      |
|-----------------------------|----------------------------------|---------------------------------------------------|
| `LOOKOUT_BACKEND`           | `ollama`                         | `ollama`, `gemini`, `openai`, or `anthropic`      |
| `LOOKOUT_OLLAMA_URL`        | `http://localhost:11434`         | Where Ollama is serving                           |
| `LOOKOUT_OLLAMA_MODEL`      | `llama3.2-vision`                | Local vision model name                           |
| `GOOGLE_API_KEY`            | —                                | Gemini API key                                    |
| `LOOKOUT_GEMINI_MODEL`      | `gemini-2.0-flash`               | Gemini model name                                 |
| `LOOKOUT_GEMINI_URL`        | `https://generativelanguage.googleapis.com` | Gemini API base URL                    |
| `OPENAI_API_KEY`            | —                                | OpenAI API key                                    |
| `LOOKOUT_OPENAI_MODEL`      | `gpt-4o`                         | OpenAI model name                                 |
| `ANTHROPIC_API_KEY`         | —                                | Anthropic API key                                 |
| `LOOKOUT_ANTHROPIC_MODEL`   | `claude-3-5-sonnet-20241022`     | Anthropic model name                              |
| `LOOKOUT_CAPTURE_INTERVAL`  | `2.0`                            | Seconds between frames in `watch` mode            |
| `LOOKOUT_CAPTURE_MONITOR`   | `1`                              | Which monitor (`1` = primary, `0` = all)          |
| `LOOKOUT_CAPTURE_QUALITY`   | `70`                             | JPEG quality (1–95)                               |
| `LOOKOUT_MAX_IMAGE_SIDE`    | `1280`                           | Downscale longest side to this many px            |
| `LOOKOUT_MEMORY_SIZE`       | `8`                              | How many recent frames to keep                    |
| `LOOKOUT_WATCH_THRESHOLD`   | `0.06`                           | Min frame difference before re-asking the model   |
| `LOOKOUT_WATCH_IDLE_WINDOW` | `30.0`                           | Reserved for idle back-off (seconds)              |
| `LOOKOUT_ACT_MAX_STEPS`     | `20`                             | Cap on steps in `act` mode                        |
| `LOOKOUT_ACT_STEP_PAUSE`    | `0.6`                            | Pause between `act` steps (seconds)               |
| `LOOKOUT_ACT_SAFETY_PAUSE`  | `true`                           | Extra pause before each click/type                |

State is stored in `~/.lookout/` (recent frames + `memory/log.jsonl`).

> `.env` is **git-ignored** — keep your API keys out of version control.

---

## How it works

1. **Capture** — `mss` grabs the screen, downscaled to ≤ `LOOKOUT_MAX_IMAGE_SIDE`
   px and JPEG-encoded to keep payloads small.
2. **Vision** — the screenshot is sent to the chosen model with a prompt tuned for
   the current mode (`ask` / `watch` / `act`).
3. **Plan / act** — in `act` mode the model returns strict JSON
   `{action, x, y, text, keys, amount, thought}` which `pyautogui` executes. After
   every step a fresh screenshot is taken for the next decision.
4. **Safety** — `pyautogui.FAILSAFE` is on (slam the mouse into a screen corner to
   abort) and **ESC** aborts via `pynput`. Use `--dry-run` to plan without touching
   the mouse.
5. **Memory** — each frame + caption is kept in `~/.lookout/memory/` and a rolling
   `log.jsonl`, so follow-up questions can reference recent state.

---

## Project layout

```
lookout/
├── lookout/
│   ├── __init__.py
│   ├── __main__.py        # CLI entry: ask / watch / act / serve
│   ├── config.py          # Settings, .env + env loading
│   ├── capture.py         # mss-based screen capture (sync + async CM)
│   ├── vision.py          # multi-backend vision client (httpx / SDKs)
│   ├── controller.py      # pyautogui mouse + keyboard, abort hotkey
│   ├── memory.py          # rolling screen memory on disk
│   ├── agent.py           # ask / watch / act loops
│   ├── ocr.py             # optional Tesseract helper + frame diff
│   └── ui/
│       ├── app.py         # FastAPI server
│       └── templates/
│           └── index.html # chat UI
├── examples/
│   ├── ask.py             # programmatic ask
│   ├── watch.py           # programmatic watch
│   └── automate.py        # programmatic act
├── tests/
│   └── test_smoke.py      # syntax + import + light unit tests
├── setup.bat              # Windows one-click setup
├── start.bat              # Windows one-click launcher
├── pyproject.toml         # packaging + `lookout` console script
├── requirements.txt
├── .env.example
└── README.md
```

---

## Troubleshooting

**"Could not reach Ollama at http://localhost:11434"** — start it (`ollama serve`)
and pull a model (`ollama pull moondream`).

**Gemini returns 403 / "has not been used in project"** — enable the *Generative
Language API* for your project (link above), then wait a minute.

**Gemini returns 404** — the model name doesn't exist for your account; try
`LOOKOUT_GEMINI_MODEL=gemini-2.5-flash`.

**It's very slow and I have no GPU** — use `moondream` locally, or switch to a
cloud backend. `act` mode is the slowest (many screenshots); cloud is recommended.

**Click coordinates are off** — your screen may be a different scale than the
screenshot. Lower `LOOKOUT_MAX_IMAGE_SIDE` to match your effective resolution.

**The agent loops without finishing** — increase `LOOKOUT_ACT_MAX_STEPS`,
rephrase the goal more concretely, or use a stronger model.

**macOS** — Lookout needs Accessibility + Screen Recording permissions; approve it
in System Settings → Privacy & Security.

**Windows smart-screen warning** — when packaging a binary yourself, sign it.

---

## Security

- API keys live only in `.env` (git-ignored) or environment variables — never in
  source. Don't commit `.env`.
- `act` mode controls your real mouse and keyboard. Use `--dry-run` to preview,
  and keep the FAILSAFE corner (or ESC) in mind.

---

## Contributing

Issues and PRs welcome. Run the smoke tests before submitting:

```bash
python -m tests.test_smoke
```

## License

MIT — do whatever you want, no warranty. Be careful with `act` mode: the agent
*will* click and type on your real screen.
