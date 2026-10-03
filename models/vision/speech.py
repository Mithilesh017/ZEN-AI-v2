"""
The three model calls a call turn makes, behind one small interface so the
session logic can be tested without the network:

    transcribe(wav) -> str          speech to text
    reply(messages) -> Iterator     streamed answer text
    synthesize(text) -> bytes       text to speech (WAV)

Voice comes from ZEN's own Kokoro server (voice-server/) when one is
configured, with Groq's Orpheus as the overflow and fallback voice. Replies
come from Groq's vision model, falling over to Gemini when Groq is rate
limited or down.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from typing import Protocol

import requests

logger = logging.getLogger("zen-ai.vision")

VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.getenv("TTS_VOICE", "autumn")
VOICE_SERVER_URL = os.getenv("VOICE_SERVER_URL", "").rstrip("/")
VOICE_SERVER_TOKEN = os.getenv("VOICE_SERVER_TOKEN", "")
VOICE_SERVER_VOICE = os.getenv("VOICE_SERVER_VOICE", "af_heart")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Fallback reply models, tried in order after Groq. Flash-Lite answers within
# about two seconds on the free tier; the larger Flash models are often
# refused there ("high demand") or take far too long for a call.
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",") if m.strip()]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# Spoken replies are one to three sentences; the reservation also counts
# against the provider's tokens-per-minute limit, so keep it tight.
MAX_REPLY_TOKENS = 200
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
    Process-wide memory that a provider is out of quota or down, so calls skip
    it until it recovers instead of each one rediscovering the problem one
    failed request at a time.
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

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self._until - self._clock())


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


class ModelBusy(RuntimeError):
    """The reply model is rate limited; it can take requests again in `wait` seconds."""

    def __init__(self, wait: float):
        super().__init__(f"model busy for {wait:.1f} s")
        self.wait = wait


class GeminiReply:
    """
    A Gemini model as a fallback reply source, through Google's
    OpenAI-compatible endpoint, so it takes the same messages (including the
    camera frame) as Groq.
    """

    def __init__(self, model: str, api_key: str, *, http=None, timeout=(3.0, 20.0)):
        self.name = model
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = http or requests.Session()
        self._timeout = timeout

    def __call__(self, messages: list[dict]) -> Iterator[str]:
        response = self._http.post(
            GEMINI_URL,
            headers=self._headers,
            json={"model": self._model, "messages": messages, "stream": True,
                  "max_tokens": MAX_REPLY_TOKENS, "temperature": 0.6},
            stream=True,
            timeout=self._timeout,
        )
        if response.status_code == 429:
            response.close()
            raise ModelBusy(parse_reset(response.headers.get("retry-after")) or 60.0)
        if response.status_code != 200:
            detail = response.text[:200]
            response.close()
            raise RuntimeError(f"Gemini {self._model} returned {response.status_code}: {detail}")
        try:
            for line in response.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                choices = json.loads(line[6:]).get("choices") or [{}]
                text = (choices[0].get("delta") or {}).get("content")
                if text:
                    yield text
        finally:
            response.close()


def gemini_replies_from_env() -> list[GeminiReply]:
    if not GEMINI_API_KEY:
        return []
    return [GeminiReply(model, GEMINI_API_KEY) for model in GEMINI_MODELS]


class VoiceSkipped(RuntimeError):
    """Our own voice server can't take this sentence; use the fallback voice."""


class OwnVoice:
    """
    ZEN's self-hosted Kokoro server (see voice-server/). Unlimited and free to
    run, but a single small machine: when it is busy it answers 503 at once,
    and when it is unreachable it is skipped for a while rather than making
    every sentence wait for a timeout.
    """

    OUTAGE_PAUSE = 30.0       # seconds to skip the server after a failure
    AUTH_PAUSE = 300.0        # a wrong token will not fix itself quickly

    def __init__(self, url: str, token: str, voice: str = "af_heart", *,
                 http=None, breaker: VoiceBreaker | None = None, timeout=(2.0, 8.0)):
        self._endpoint = f"{url}/v1/audio/speech"
        self._headers = {"Authorization": f"Bearer {token}"}
        self._voice = voice
        self._http = http or requests.Session()
        self._breaker = breaker or VoiceBreaker()
        self._timeout = timeout

    def available(self) -> bool:
        return self._breaker.is_open()

    def synthesize(self, text: str) -> bytes:
        if not self._breaker.is_open():
            raise VoiceSkipped("voice server paused after a failure")
        try:
            response = self._http.post(
                self._endpoint,
                json={"input": text, "voice": self._voice, "response_format": "wav"},
                headers=self._headers,
                timeout=self._timeout,
            )
        except requests.RequestException as err:
            logger.warning("Voice server unreachable; using fallback voice for %.0f s: %s",
                           self.OUTAGE_PAUSE, err)
            self._breaker.trip(self.OUTAGE_PAUSE)
            raise VoiceSkipped(str(err)) from err

        if response.status_code == 200 and response.content.startswith(b"RIFF"):
            return response.content
        if response.status_code == 503:
            raise VoiceSkipped("voice server busy")  # overflow this sentence only
        pause = self.AUTH_PAUSE if response.status_code == 401 else self.OUTAGE_PAUSE
        logger.error("Voice server returned %s; using fallback voice for %.0f s",
                     response.status_code, pause)
        self._breaker.trip(pause)
        raise VoiceSkipped(f"voice server status {response.status_code}")


def own_voice_from_env() -> OwnVoice | None:
    if not VOICE_SERVER_URL:
        return None
    if len(VOICE_SERVER_TOKEN) < 32:
        logger.error("VOICE_SERVER_URL is set but VOICE_SERVER_TOKEN is missing; not using it")
        return None
    return OwnVoice(VOICE_SERVER_URL, VOICE_SERVER_TOKEN, VOICE_SERVER_VOICE)


class GroqEngines:
    DOWN_PAUSE = 20.0    # seconds a reply source is skipped after an error

    def __init__(self, client, breaker: VoiceBreaker | None = None, sleep=time.sleep,
                 own_voice: OwnVoice | None = None, fallback_replies=(), clock=time.monotonic):
        self._client = client
        self._breaker = breaker or VoiceBreaker()
        self._sleep = sleep
        self._own_voice = own_voice
        # Reply sources in order of preference, each with its own breaker.
        self._replies = [(self._groq_reply, VoiceBreaker(clock)),
                         *((source, VoiceBreaker(clock)) for source in fallback_replies)]

    def voice_available(self) -> bool:
        own = self._own_voice is not None and self._own_voice.available()
        return own or self._breaker.is_open()

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
        """
        The first reply source that will answer now. A rate-limited or failing
        source is skipped until it recovers. ModelBusy is raised only when
        every source is rate limited, with the shortest wait among them.
        """
        failure: Exception | None = None
        for source, breaker in self._replies:
            if not breaker.is_open():
                continue
            name = getattr(source, "name", VISION_MODEL)
            stream = source(messages)
            try:
                first = next(stream)
            except StopIteration:
                return
            except ModelBusy as busy:
                logger.info("Reply model %s rate limited for %.0f s", name, busy.wait)
                breaker.trip(busy.wait)
                continue
            except Exception as err:
                logger.warning("Reply model %s failed; skipping it for %.0f s: %s",
                               name, self.DOWN_PAUSE, err)
                breaker.trip(self.DOWN_PAUSE)
                failure = err
                continue
            yield first
            yield from stream
            return

        waits = [breaker.remaining() for _, breaker in self._replies]
        if failure is not None and len(self._replies) == 1:
            raise failure
        raise ModelBusy(min(waits))

    def _groq_reply(self, messages: list[dict]) -> Iterator[str]:
        try:
            # No silent SDK retries: on a live call a rate limit has to be
            # handled visibly (see CallSession), not by sleeping for 20 s.
            stream = self._client.with_options(max_retries=0).chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                stream=True,
                max_tokens=MAX_REPLY_TOKENS,
                temperature=0.6,
                # Hidden reasoning costs ~0.5 s before the first word; a spoken
                # reply needs to start immediately.
                reasoning_effort="none",
            )
        except Exception as err:
            if getattr(err, "status_code", None) == 429:
                headers = getattr(getattr(err, "response", None), "headers", None) or {}
                raise ModelBusy(rate_limit_wait(headers)) from err
            raise
        try:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

    def synthesize(self, text: str) -> bytes:
        if self._own_voice is not None:
            try:
                return self._own_voice.synthesize(text)
            except VoiceSkipped:
                pass
        return self._groq_voice(text)

    def _groq_voice(self, text: str) -> bytes:
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
