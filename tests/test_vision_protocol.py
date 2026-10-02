"""Tests for the live-call wire protocol."""

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.vision.protocol import (
    MAX_IMAGE_BYTES, Detection, ProtocolError, Utterance, decode_speech,
    encode_speech, encode_utterance, parse_control, parse_utterance,
)
from models.vision.routes import same_origin

WAV = b"RIFF" + b"\x00" * 40
JPEG = b"\xff\xd8" + b"\x00" * 20


def utterance(**overrides):
    fields = dict(turn=1, audio=WAV, image=JPEG, camera=True, facing="environment",
                  detections=(Detection("cup", 0.9),))
    fields.update(overrides)
    return Utterance(**fields)


def test_utterance_round_trips():
    original = utterance()
    assert parse_utterance(encode_utterance(original)) == original


def test_utterance_without_image():
    parsed = parse_utterance(encode_utterance(utterance(image=None, camera=False)))
    assert parsed.image is None and parsed.camera is False


def raw(header, body=WAV):
    data = json.dumps(header).encode()
    return struct.pack(">BI", 1, len(data)) + data + body


@pytest.mark.parametrize("data, reason", [
    (b"\x01", "Truncated"),
    (raw({"turn": 0, "audio_bytes": len(WAV)}), "turn"),
    (raw({"turn": 1, "audio_bytes": 10_000}), "Truncated audio"),
    (raw({"turn": 1, "audio_bytes": len(WAV)}, b"OggS" + b"\x00" * 40), "WAV"),
    (raw({"turn": 1, "audio_bytes": len(WAV)}, WAV + b"GIF89a"), "JPEG"),
    (raw({"turn": 1, "audio_bytes": len(WAV)}, WAV + b"\xff\xd8" + b"\x00" * MAX_IMAGE_BYTES), "too large"),
    (struct.pack(">BI", 1, 5) + b"nope!" + WAV, "Malformed"),
    (struct.pack(">BI", 9, 2) + b"{}" + WAV, "Unknown"),
], ids=["truncated", "turn", "short-audio", "not-wav", "not-jpeg", "big-image", "bad-header", "kind"])
def test_rejects_malformed_utterances(data, reason):
    with pytest.raises(ProtocolError, match=reason):
        parse_utterance(data)


def test_detections_are_sanitised():
    parsed = parse_utterance(raw({
        "turn": 1, "audio_bytes": len(WAV),
        "detections": [{"label": "  dog ", "score": 7}, {"label": 3, "score": 1}, "junk",
                       {"label": "x" * 100, "score": -1}],
    }))
    assert parsed.detections == (Detection("dog", 1.0), Detection("x" * 40, 0.0))
    assert parsed.facing == "user"


def test_speech_frames_round_trip():
    assert decode_speech(encode_speech(7, 3, WAV)) == (7, 3, WAV)


def test_control_messages():
    assert parse_control('{"type":"hello","v":1}') == {"type": "hello"}
    assert parse_control('{"type":"interrupt","turn":2,"played":1}') == {
        "type": "interrupt", "turn": 2, "played": 1}
    for bad in ('{"type":"hello","v":0}', '{"type":"interrupt","turn":2,"played":-1}',
                '{"type":"dance"}', "[]", "not json"):
        with pytest.raises(ProtocolError):
            parse_control(bad)


def test_same_origin_check():
    assert same_origin("https://zen.neuzem.com", "zen.neuzem.com")
    assert same_origin("http://localhost:5173", "localhost:5173")
    assert not same_origin("https://evil.example", "zen.neuzem.com")
    assert not same_origin(None, "zen.neuzem.com")
