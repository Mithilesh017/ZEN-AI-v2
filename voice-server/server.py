"""
ZEN's own text-to-speech server: Kokoro-82M on CPU, behind a small HTTP API.

It answers the subset of the OpenAI speech API that ZEN uses, so the app can
treat it like any hosted voice:

    POST /v1/audio/speech   {"input": "...", "voice": "af_heart", "speed": 1.0}
                            Authorization: Bearer <VOICE_TOKEN>
                            -> audio/wav (24 kHz mono PCM16)
    GET  /health            -> {"ok": true}

Synthesis is CPU-bound, so only MAX_CONCURRENT requests run at once. A request
that cannot start within QUEUE_WAIT seconds gets 503 straight away; ZEN then
uses its fallback voice for that sentence instead of making the user wait.
"""

from __future__ import annotations

import io
import logging
import os
import secrets
import threading
import time
import wave

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from kokoro_onnx import Kokoro
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zen-voice")

TOKEN = os.environ.get("VOICE_TOKEN", "")
MODEL_PATH = os.environ.get("KOKORO_MODEL", "models/kokoro-v1.0.onnx")
VOICES_PATH = os.environ.get("KOKORO_VOICES", "models/voices-v1.0.bin")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "af_heart")
MAX_INPUT = int(os.environ.get("MAX_INPUT_CHARS", "400"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))
QUEUE_WAIT = float(os.environ.get("QUEUE_WAIT", "0.4"))

if len(TOKEN) < 32:
    raise RuntimeError("VOICE_TOKEN must be set to a random string of at least 32 characters")


def _thread_budget() -> int:
    """
    Threads for synthesis; 0 lets ONNX Runtime choose (one per physical core),
    which is fastest on a normal machine. In a container the runtime sees the
    host's cores, and that many threads on a 2-CPU allowance makes synthesis
    slower, so the container's CPU limit is used instead.
    """
    if os.environ.get("THREADS"):
        return max(1, int(os.environ["THREADS"]))
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return 0


def _load() -> Kokoro:
    options = ort.SessionOptions()
    options.intra_op_num_threads = _thread_budget()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_PATH, options, providers=["CPUExecutionProvider"])
    kokoro = Kokoro.from_session(session, VOICES_PATH)
    kokoro.create("Warming up.", voice=DEFAULT_VOICE)  # first run compiles kernels
    return kokoro


started = time.monotonic()
kokoro = _load()
VOICES = set(kokoro.get_voices())
slots = threading.BoundedSemaphore(MAX_CONCURRENT)
logger.info("Kokoro ready in %.1f s with %d voices (%s threads)",
            time.monotonic() - started, len(VOICES), _thread_budget() or "auto")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1)
    voice: str = DEFAULT_VOICE
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    model: str | None = None            # accepted for OpenAI compatibility, ignored
    response_format: str = "wav"


def _wav(samples: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buffer.getvalue()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/audio/speech")
def speech(body: SpeechRequest, authorization: str = Header(default="")):
    if not secrets.compare_digest(authorization, f"Bearer {TOKEN}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if body.response_format != "wav":
        raise HTTPException(status_code=400, detail="Only wav is supported")
    text = body.input.strip()
    if not text or len(text) > MAX_INPUT:
        raise HTTPException(status_code=400, detail=f"input must be 1-{MAX_INPUT} characters")
    voice = body.voice if body.voice in VOICES else DEFAULT_VOICE

    if not slots.acquire(timeout=QUEUE_WAIT):
        return JSONResponse({"error": "busy"}, status_code=503, headers={"Retry-After": "1"})
    try:
        began = time.monotonic()
        samples, rate = kokoro.create(text, voice=voice, speed=body.speed)
        took = time.monotonic() - began
    finally:
        slots.release()

    audio_seconds = len(samples) / rate
    logger.info("%d chars -> %.2f s audio in %.2f s (RTF %.2f)",
                len(text), audio_seconds, took, took / max(audio_seconds, 1e-3))
    return Response(
        _wav(samples, rate),
        media_type="audio/wav",
        headers={"X-Synthesis-Ms": str(round(took * 1000))},
    )
