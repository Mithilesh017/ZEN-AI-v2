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


class SpeechUnavailable(RuntimeError):
    """Text-to-speech cannot be used for this call (terms, quota, outage)."""


class Engines(Protocol):
    def transcribe(self, wav: bytes) -> str: ...
    def reply(self, messages: list[dict]) -> Iterator[str]: ...
    def synthesize(self, text: str) -> bytes: ...


class GroqEngines:
    def __init__(self, client):
        self._client = client

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
        try:
            response = self._client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=text,
                response_format="wav",
            )
            return response.read()
        except Exception as err:
            if _is_permanent(err):
                raise SpeechUnavailable(str(err)) from err
            raise


def _field(segment, name):
    return segment.get(name) if isinstance(segment, dict) else getattr(segment, name, None)


def _is_permanent(err: Exception) -> bool:
    """Errors retrying cannot fix: model terms not accepted, model missing, bad key."""
    status = getattr(err, "status_code", None)
    return status in (400, 401, 403, 404)
