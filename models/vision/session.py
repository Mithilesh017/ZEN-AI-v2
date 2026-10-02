"""
One live call: conversation state, turn-taking and the speech pipeline.

A turn runs on its own thread:

    transcribe ─▶ recall memories (bounded) ─▶ stream reply ─▶ chunk sentences
                                                                   │
                         send in order ◀─ synthesize (2 in flight) ◀┘

Each finished sentence is synthesized while the model keeps writing, and
results are sent strictly in order, so the first audio leaves after roughly
one sentence of latency rather than a whole reply.

Barge-in: the client stops playback the instant the user talks over ZEN and
reports how many sentences were actually heard. The running turn is
cancelled, and the assistant's history is cut to what the user heard, so the
model never believes it said something the user never got to hear.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from .chunker import SentenceChunker
from .prompt import build_call_prompt, describe_view
from .protocol import Utterance, encode_speech
from .speech import Engines, SpeechUnavailable

logger = logging.getLogger("zen-ai.vision")

SendJson = Callable[[dict], None]
SendBytes = Callable[[bytes], None]
Recall = Callable[[str, str], tuple[object, list[str]]]
Remember = Callable[[str, str, object], None]

MIN_WORDS_TO_REMEMBER = 4


@dataclass(frozen=True)
class CallLimits:
    max_seconds: int = 15 * 60
    max_history: int = 16          # turns of context (user + assistant entries)
    recall_budget: float = 0.4     # seconds the first turn waits for long-term memory
    tts_in_flight: int = 2


@dataclass
class _Entry:
    turn: int
    role: str
    parts: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.parts)


@dataclass
class _Turn:
    id: int
    cancelled: threading.Event = field(default_factory=threading.Event)


class CallSession:
    def __init__(
        self,
        *,
        user_name: str,
        email: str,
        engines: Engines,
        send_json: SendJson,
        send_bytes: SendBytes,
        recall: Recall | None = None,
        remember: Remember | None = None,
        limits: CallLimits = CallLimits(),
        clock: Callable[[], float] = time.monotonic,
    ):
        self._user_name = user_name
        self._email = email
        self._engines = engines
        self._send_json = send_json
        self._send_bytes = send_bytes
        self._recall = recall
        self._remember = remember
        self._limits = limits
        self._clock = clock

        self._lock = threading.Lock()
        self._history: list[_Entry] = []
        self._heard: dict[int, int] = {}          # turn -> sentences the user heard
        self._active: _Turn | None = None
        self._last_turn = 0
        self._voice = True              # false after a permanent TTS failure
        self._quota_notified = False    # told the user the voice quota ran out
        self._memories: list[str] = []
        self._recalled = False
        self._started = clock()
        self._pool = ThreadPoolExecutor(max_workers=limits.tts_in_flight + 2, thread_name_prefix="zen-call")
        self._closed = False

    # ── lifecycle ────────────────────────────────────────────

    def ready_message(self) -> dict:
        return {"type": "ready", "max_seconds": self._limits.max_seconds, "voice": self._voice_on()}

    def _voice_on(self) -> bool:
        """Voice is used unless it failed for good, or the shared quota is spent for now."""
        available = getattr(self._engines, "voice_available", None)
        return self._voice and (available() if available else True)

    def remaining_seconds(self) -> float:
        return self._limits.max_seconds - (self._clock() - self._started)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._active:
                self._active.cancelled.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ── inbound ──────────────────────────────────────────────

    def interrupt(self, turn: int, played: int) -> None:
        """The user talked over ZEN after hearing `played` sentences of `turn`."""
        with self._lock:
            if self._active and self._active.id == turn:
                self._active.cancelled.set()
            self._heard[turn] = played
            for entry in self._history:
                if entry.turn == turn and entry.role == "assistant":
                    del entry.parts[played:]
            self._history = [e for e in self._history if e.parts]

    def handle_utterance(self, utterance: Utterance) -> threading.Thread | None:
        with self._lock:
            if self._closed or utterance.turn <= self._last_turn:
                return None
            if self._active:
                self._active.cancelled.set()
            self._last_turn = utterance.turn
            turn = self._active = _Turn(utterance.turn)

        worker = threading.Thread(
            target=self._run_turn, args=(turn, utterance), name=f"zen-turn-{turn.id}", daemon=True
        )
        worker.start()
        return worker

    # ── the turn pipeline ────────────────────────────────────

    def _run_turn(self, turn: _Turn, utterance: Utterance) -> None:
        started = self._clock()
        timings: dict[str, int] = {}

        def mark(name: str) -> None:
            timings.setdefault(name, round((self._clock() - started) * 1000))

        try:
            text = self._engines.transcribe(utterance.audio)
        except Exception:
            logger.exception("Transcription failed")
            self._emit(turn, {"type": "turn_done", "turn": turn.id, "skipped": "error"})
            return
        mark("stt_ms")

        if turn.cancelled.is_set():
            return
        if not text:
            self._emit(turn, {"type": "turn_done", "turn": turn.id, "skipped": "no_speech"})
            return
        self._emit(turn, {"type": "transcript", "turn": turn.id, "text": text})

        messages = self._build_messages(text, utterance)
        self._record(turn.id, "user", [text])

        spoken: list[str] = []
        try:
            self._speak(turn, messages, spoken, mark)
        except Exception:
            logger.exception("Reply failed on turn %s", turn.id)
            self._emit(turn, {"type": "notice", "code": "reply_failed",
                              "message": "I lost my train of thought. Could you say that again?"})
        finally:
            with self._lock:
                heard = self._heard.get(turn.id)
            self._record(turn.id, "assistant", spoken if heard is None else spoken[:heard])
            with self._lock:
                if self._active is turn:
                    self._active = None

        timings["total_ms"] = round((self._clock() - started) * 1000)
        logger.info("Call turn %s for %s: %s", turn.id, self._email, timings)
        self._emit(turn, {"type": "turn_done", "turn": turn.id, "timings": timings})

    def _speak(self, turn: _Turn, messages: list[dict], spoken: list[str], mark) -> None:
        chunker = SentenceChunker()
        pending: deque[tuple[int, str, Future | None]] = deque()
        seq = 0

        def queue(chunk: str) -> None:
            nonlocal seq
            seq += 1
            future = self._pool.submit(self._engines.synthesize, chunk) if self._voice_on() else None
            pending.append((seq, chunk, future))

        def drain(block: bool) -> None:
            while pending and not turn.cancelled.is_set():
                n, chunk, future = pending[0]
                if future and not future.done() and not block:
                    return
                pending.popleft()
                audio = self._audio_of(future)
                if not self._emit(turn, {"type": "say", "turn": turn.id, "seq": n,
                                         "text": chunk, "audio": audio is not None}):
                    return
                if audio is not None:
                    self._send_bytes(encode_speech(turn.id, n, audio))
                    self._quota_notified = False
                    mark("first_audio_ms")
                spoken.append(chunk)

        for delta in self._engines.reply(messages):
            if turn.cancelled.is_set():
                break
            mark("ttft_ms")
            for chunk in chunker.feed(delta):
                queue(chunk)
            drain(block=False)
        else:
            for chunk in chunker.flush():
                queue(chunk)
        drain(block=True)

        for _, _, future in pending:
            if future:
                future.cancel()

    def _audio_of(self, future: Future | None) -> bytes | None:
        if future is None:
            return None
        try:
            return future.result()
        except SpeechUnavailable as err:
            if err.reason == "quota":
                # Temporary: voice returns by itself once the quota resets.
                if not self._quota_notified:
                    self._quota_notified = True
                    self._send_json({"type": "notice", "code": "voice_limited",
                                     "message": "My voice has hit its usage limit for now, so I'll reply in captions."})
            else:
                logger.warning("Text-to-speech unavailable; continuing with captions only", exc_info=True)
                if self._voice:
                    self._voice = False
                    self._send_json({"type": "notice", "code": "voice_unavailable",
                                     "message": "Voice is unavailable right now, so I'll reply in captions."})
        except Exception:
            logger.warning("Text-to-speech failed for one sentence", exc_info=True)
        return None

    # ── context ──────────────────────────────────────────────

    def _build_messages(self, text: str, utterance: Utterance) -> list[dict]:
        memories = self._recall_for(text)
        with self._lock:
            history = [
                {"role": e.role, "content": e.text}
                for e in self._history[-self._limits.max_history:]
            ]

        note = describe_view(utterance.camera and utterance.image is not None,
                             utterance.facing, utterance.detections)
        content: list[dict] = [{"type": "text", "text": f"{note}\n{text}"}]
        if utterance.camera and utterance.image:
            data_url = "data:image/jpeg;base64," + base64.b64encode(utterance.image).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        return [
            {"role": "system", "content": build_call_prompt(self._user_name, memories)},
            *history,
            {"role": "user", "content": content},
        ]

    def _recall_for(self, text: str) -> list[str]:
        """
        Long-term memory lookup that never holds up a reply for long. Only the
        first turn waits (briefly) for it; afterwards each lookup runs in the
        background and its result is used from the next turn on, since a
        call's topic rarely changes from one sentence to the next.
        """
        if not self._recall:
            return self._memories
        budget = 0.0 if self._recalled else self._limits.recall_budget
        self._recalled = True

        def lookup():
            vector, memories = self._recall(self._email, text)
            self._memories = memories or self._memories
            if self._remember and len(text.split()) >= MIN_WORDS_TO_REMEMBER:
                self._remember(self._email, text, vector)

        try:
            self._pool.submit(lookup).result(timeout=budget)
        except FutureTimeout:
            pass
        except Exception:
            logger.warning("Memory lookup failed during call", exc_info=True)
        return self._memories

    def _record(self, turn: int, role: str, parts: list[str]) -> None:
        if not parts:
            return
        with self._lock:
            self._history.append(_Entry(turn, role, list(parts)))
            # Keep the buffer bounded; the prompt only uses the tail anyway.
            del self._history[: max(0, len(self._history) - 4 * self._limits.max_history)]

    def _emit(self, turn: _Turn, message: dict) -> bool:
        """Send a message for `turn` unless it has been superseded."""
        if turn.cancelled.is_set():
            return False
        self._send_json(message)
        return True
