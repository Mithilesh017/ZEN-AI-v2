"""
The three model calls a call turn makes, behind one small interface so the
session logic can be tested without the network:

    transcribe(wav) -> str          speech to text
    reply(messages) -> Iterator     streamed answer text
    synthesize(text) -> bytes       text to speech (WAV)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from typing import Protocol

logger = logging.getLogger("zen-ai.vision")

VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.getenv("TTS_VOICE", "autumn")

MAX_REPLY_TOKENS = 400
# Whisper invents text ("Thank you.") for near-silence; segments it is
# unsure contain speech are dropped.
NO_SPEECH_THRESHOLD = 0.6
# A rate limit that clears within this many seconds (the per-minute token
# budget) is waited out; a longer one (the daily request quota) turns voice
# off until it resets.
MAX_RATE_LIMIT_WAIT = 4.0


class SpeechUnavailable(RuntimeError):
    """Text-to-speech cannot be used right now (terms, quota, outage)."""

    def __init__(self, message: str, *, reason: str = "error"):
        super().__init__(message)
        self.reason = reason  # "quota" | "error"


class Engines(Protocol):
    def transcribe(self, wav: bytes) -> str: ...
    def reply(self, messages: list[dict]) -> Iterator[str]: ...
    def synthesize(self, text: str) -> bytes: ...
    def voice_available(self) -> bool: ...


class VoiceBreaker:
    """
    Process-wide memory that the TTS quota is spent, so new calls start in
    captions-only mode instead of each one rediscovering the limit one failed
    sentence at a time.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._until = 0.0

    def trip(self, seconds: float) -> None:
        with self._lock:
            self._until = max(self._until, self._clock() + seconds)

    def is_open(self) -> bool:
        with self._lock:
            return self._clock() >= self._until


_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def parse_reset(value: str | None) -> float | None:
    """Seconds from a Retry-After value ("2") or a Groq reset header ("19h12m0s", "800ms")."""
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    parts = _DURATION.findall(value)
    if not parts:
        return None
    return sum(float(n) * _UNIT_SECONDS[unit] for n, unit in parts)


def rate_limit_wait(headers) -> float:
    """
    How long until a 429 clears. The daily request reset only matters when
    the request quota is what ran out; otherwise it is the token budget.
    """
    retry_after = parse_reset(headers.get("retry-after"))
    tokens_reset = parse_reset(headers.get("x-ratelimit-reset-tokens"))
    waits = [w for w in (retry_after, tokens_reset) if w is not None]
    if headers.get("x-ratelimit-remaining-requests") == "0":
        requests_reset = parse_reset(headers.get("x-ratelimit-reset-requests"))
        if requests_reset is not None:
            waits.append(requests_reset)
    return max(waits, default=60.0)


class GroqEngines:
    def __init__(self, client, breaker: VoiceBreaker | None = None, sleep=time.sleep):
        self._client = client
        self._breaker = breaker or VoiceBreaker()
        self._sleep = sleep

    def voice_available(self) -> bool:
        return self._breaker.is_open()

    def transcribe(self, wav: bytes) -> str:
        result = self._client.audio.transcriptions.create(
            file=("utterance.wav", wav),
            model=STT_MODEL,
            response_format="verbose_json",
            temperature=0,
        )
        segments = getattr(result, "segments", None)
        if not segments:
            return (getattr(result, "text", "") or "").strip()
        spoken = [
            _field(s, "text") for s in segments
            if (_field(s, "no_speech_prob") or 0) < NO_SPEECH_THRESHOLD
        ]
        return " ".join(t.strip() for t in spoken if t).strip()

    def reply(self, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            stream=True,
            max_tokens=MAX_REPLY_TOKENS,
            temperature=0.6,
            # Hidden reasoning costs ~0.5 s before the first word; a spoken
            # reply needs to start immediately.
            reasoning_effort="none",
        )
        try:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

    def synthesize(self, text: str) -> bytes:
        if not self._breaker.is_open():
            raise SpeechUnavailable("Voice quota exhausted", reason="quota")
        for attempt in range(2):
            try:
                response = self._client.audio.speech.create(
                    model=TTS_MODEL,
                    voice=TTS_VOICE,
                    input=text,
                    response_format="wav",
                )
                return response.read()
            except Exception as err:
                if getattr(err, "status_code", None) == 429:
                    headers = getattr(getattr(err, "response", None), "headers", None) or {}
                    wait = rate_limit_wait(headers)
                    if attempt == 0 and wait <= MAX_RATE_LIMIT_WAIT:
                        self._sleep(wait + 0.1)
                        continue
                    logger.warning("Voice quota reached; captions only for %.0f s", wait)
                    self._breaker.trip(wait)
                    raise SpeechUnavailable(str(err), reason="quota") from err
                if _is_permanent(err):
                    raise SpeechUnavailable(str(err)) from err
                raise
        raise SpeechUnavailable("Voice is rate limited", reason="quota")


def _field(segment, name):
    return segment.get(name) if isinstance(segment, dict) else getattr(segment, name, None)


def _is_permanent(err: Exception) -> bool:
    """Errors retrying cannot fix: model terms not accepted, model missing, bad key."""
    status = getattr(err, "status_code", None)
    return status in (400, 401, 403, 404)
