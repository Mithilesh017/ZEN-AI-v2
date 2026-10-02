"""Tests for a live call's turn pipeline, with the model calls faked."""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.vision.protocol import Detection, Utterance, decode_speech
from models.vision.session import CallLimits, CallSession
from models.vision.speech import SpeechUnavailable

WAV = b"RIFF" + b"\x00" * 40
JPEG = b"\xff\xd8" + b"\x00" * 20


class FakeEngines:
    def __init__(self, transcript="What is this?", reply="That's a mug. It looks empty. Want a refill?",
                 tts_error=None, reply_gate=None):
        self.transcript = transcript
        self.reply_text = reply
        self.tts_error = tts_error
        self.reply_gate = reply_gate
        self.messages = None
        self.synthesized = []

    def transcribe(self, wav):
        return self.transcript

    def reply(self, messages):
        self.messages = messages
        for word in self.reply_text.split(" "):
            if self.reply_gate:
                self.reply_gate.wait()
            yield word + " "

    def synthesize(self, text):
        if self.tts_error:
            raise self.tts_error
        self.synthesized.append(text)
        return b"RIFF" + text.encode()


class Wire:
    def __init__(self):
        self.events, self.audio = [], []

    def json(self, msg):
        self.events.append(msg)

    def bytes(self, data):
        self.audio.append(decode_speech(data))

    def of(self, kind):
        return [e for e in self.events if e["type"] == kind]


def make_call(engines, wire=None, **kwargs):
    wire = wire or Wire()
    call = CallSession(user_name="Asha", email="a@x.com", engines=engines,
                       send_json=wire.json, send_bytes=wire.bytes, **kwargs)
    return call, wire


def utter(turn=1, **overrides):
    fields = dict(turn=turn, audio=WAV, image=JPEG, camera=True, facing="environment",
                  detections=(Detection("cup", 0.91), Detection("noise", 0.2)))
    fields.update(overrides)
    return Utterance(**fields)


def run(call, utterance):
    worker = call.handle_utterance(utterance)
    worker.join(timeout=5)
    assert not worker.is_alive()


def test_a_turn_streams_transcript_captions_and_ordered_audio():
    call, wire = make_call(FakeEngines())
    run(call, utter())

    assert wire.of("transcript") == [{"type": "transcript", "turn": 1, "text": "What is this?"}]
    says = wire.of("say")
    assert [s["text"] for s in says] == ["That's a mug.", "It looks empty.", "Want a refill?"]
    assert [s["seq"] for s in says] == [1, 2, 3]
    assert all(s["audio"] for s in says)
    assert [(t, s) for t, s, _ in wire.audio] == [(1, 1), (1, 2), (1, 3)]

    done = wire.of("turn_done")[0]
    assert set(done["timings"]) >= {"stt_ms", "ttft_ms", "first_audio_ms", "total_ms"}


def test_the_model_sees_the_frame_and_detector_hints():
    engines = FakeEngines()
    call, _ = make_call(engines)
    run(call, utter())

    content = engines.messages[-1]["content"]
    assert "back camera" in content[0]["text"]
    assert "cup (91%)" in content[0]["text"] and "noise" not in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_camera_off_sends_no_image():
    engines = FakeEngines()
    call, _ = make_call(engines)
    run(call, utter(camera=False, image=None))
    content = engines.messages[-1]["content"]
    assert len(content) == 1 and "Camera is off" in content[0]["text"]


def test_history_carries_into_the_next_turn_as_text_only():
    engines = FakeEngines()
    call, _ = make_call(engines)
    run(call, utter(1))
    run(call, utter(2))
    roles = [m["role"] for m in engines.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert engines.messages[1]["content"] == "What is this?"
    assert engines.messages[2]["content"] == "That's a mug. It looks empty. Want a refill?"


def test_interrupt_cuts_history_to_what_was_heard():
    engines = FakeEngines()
    call, _ = make_call(engines)
    run(call, utter(1))
    call.interrupt(1, played=1)
    run(call, utter(2))
    assert engines.messages[2] == {"role": "assistant", "content": "That's a mug."}


def test_interrupt_before_anything_was_heard_drops_the_reply():
    engines = FakeEngines()
    call, _ = make_call(engines)
    run(call, utter(1))
    call.interrupt(1, played=0)
    run(call, utter(2))
    assert [m["role"] for m in engines.messages] == ["system", "user", "user"]


def test_interrupt_cancels_a_running_turn():
    gate = threading.Event()
    engines = FakeEngines(reply_gate=gate)
    call, wire = make_call(engines)
    worker = call.handle_utterance(utter(1))
    while not wire.of("transcript"):
        time.sleep(0.01)
    call.interrupt(1, played=0)
    gate.set()
    worker.join(timeout=5)
    assert wire.of("say") == [] and wire.of("turn_done") == []


def test_a_new_utterance_supersedes_the_running_turn():
    gate = threading.Event()
    engines = FakeEngines(reply_gate=gate)
    call, wire = make_call(engines)
    first = call.handle_utterance(utter(1))
    while not wire.of("transcript"):
        time.sleep(0.01)
    second = call.handle_utterance(utter(2))
    gate.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert {s["turn"] for s in wire.of("say")} == {2}


def test_stale_or_repeated_turn_numbers_are_ignored():
    call, _ = make_call(FakeEngines())
    run(call, utter(3))
    assert call.handle_utterance(utter(3)) is None
    assert call.handle_utterance(utter(2)) is None


def test_voice_failure_degrades_to_captions():
    call, wire = make_call(FakeEngines(tts_error=SpeechUnavailable("terms")))
    run(call, utter())
    assert [n["code"] for n in wire.of("notice")] == ["voice_unavailable"]
    assert [s["audio"] for s in wire.of("say")] == [False, False, False]
    assert wire.audio == []


def test_empty_transcript_skips_the_turn():
    engines = FakeEngines(transcript="")
    call, wire = make_call(engines)
    run(call, utter())
    assert wire.of("turn_done") == [{"type": "turn_done", "turn": 1, "skipped": "no_speech"}]
    assert engines.messages is None


def test_memories_are_used_and_slow_lookups_never_block_a_reply():
    remembered = []

    def recall(email, text):
        return "vec", ["Likes espresso"]

    engines = FakeEngines(transcript="Can you help me make some coffee?")
    call, _ = make_call(engines, recall=recall,
                        remember=lambda e, t, v: remembered.append(t))
    run(call, utter())
    assert "Likes espresso" in engines.messages[0]["content"]
    assert remembered == ["Can you help me make some coffee?"]

    def slow_recall(email, text):
        time.sleep(1)
        return "vec", []

    engines = FakeEngines()
    call, wire = make_call(engines, recall=slow_recall, limits=CallLimits(recall_budget=0.05))
    started = time.monotonic()
    run(call, utter())
    assert time.monotonic() - started < 0.9
    assert wire.of("say")


def test_only_the_first_turn_waits_for_memory():
    calls = []

    def recall(email, text):
        calls.append(text)
        time.sleep(0.3)
        return "vec", [f"fact from: {text}"]

    engines = FakeEngines()
    call, _ = make_call(engines, recall=recall, limits=CallLimits(recall_budget=1.0))
    run(call, utter(1))
    assert "fact from: What is this?" in engines.messages[0]["content"]

    started = time.monotonic()
    run(call, utter(2))
    assert time.monotonic() - started < 0.25
    assert len(calls) == 2


def test_call_time_limit():
    now = [0.0]
    call, _ = make_call(FakeEngines(), limits=CallLimits(max_seconds=60), clock=lambda: now[0])
    assert call.remaining_seconds() == 60
    now[0] = 61
    assert call.remaining_seconds() < 0
