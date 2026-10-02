"""
Wire protocol for a live call over a single WebSocket.

Control messages travel as JSON text frames. Media travels as binary frames
with a small envelope, so an utterance (its audio plus the camera frame taken
when the user stopped speaking) arrives as one atomic message and can never be
split or reordered.

Client -> server
  text   {"type": "hello", "v": 1}
  text   {"type": "interrupt", "turn": int, "played": int}
  text   {"type": "bye"}
  binary UTTERANCE  u8 kind | u32 header_len | header JSON | WAV bytes | JPEG bytes
         header = {"turn": int, "audio_bytes": int, "camera": bool,
                   "facing": str, "detections": [{"label": str, "score": float}]}

Server -> client
  text   ready | transcript | say | turn_done | notice | error | ended
  binary SPEECH  u8 kind | u32 turn | u16 seq | WAV bytes

All integers are big-endian.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

VERSION = 1

KIND_UTTERANCE = 0x01
KIND_SPEECH = 0x10

MAX_HEADER_BYTES = 4 * 1024
MAX_AUDIO_BYTES = 1_000_000   # ~30 s of 16 kHz mono PCM16
MAX_IMAGE_BYTES = 400_000
MAX_DETECTIONS = 20
MAX_LABEL_LENGTH = 40

FACINGS = ("user", "environment")

_ENVELOPE = struct.Struct(">BI")
_SPEECH = struct.Struct(">BIH")


class ProtocolError(ValueError):
    """The client sent something malformed. The message is safe to show."""


@dataclass(frozen=True)
class Detection:
    label: str
    score: float


@dataclass(frozen=True)
class Utterance:
    turn: int
    audio: bytes
    image: bytes | None
    camera: bool
    facing: str
    detections: tuple[Detection, ...] = field(default_factory=tuple)


def _facing(value) -> str:
    return value if value in FACINGS else "user"


def _detections(raw) -> tuple[Detection, ...]:
    if not isinstance(raw, list):
        return ()
    found = []
    for item in raw[:MAX_DETECTIONS]:
        if not isinstance(item, dict):
            continue
        label, score = item.get("label"), item.get("score")
        if not isinstance(label, str) or not isinstance(score, (int, float)):
            continue
        label = label.strip()[:MAX_LABEL_LENGTH]
        if label:
            found.append(Detection(label, max(0.0, min(1.0, float(score)))))
    return tuple(found)


def parse_utterance(data: bytes) -> Utterance:
    if len(data) < _ENVELOPE.size:
        raise ProtocolError("Truncated media message.")
    kind, header_len = _ENVELOPE.unpack_from(data)
    if kind != KIND_UTTERANCE:
        raise ProtocolError("Unknown media message.")
    if header_len > MAX_HEADER_BYTES:
        raise ProtocolError("Media header too large.")

    start = _ENVELOPE.size
    try:
        header = json.loads(data[start:start + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProtocolError("Malformed media header.") from None
    if not isinstance(header, dict):
        raise ProtocolError("Malformed media header.")

    turn, audio_bytes = header.get("turn"), header.get("audio_bytes")
    if not isinstance(turn, int) or turn < 1:
        raise ProtocolError("Invalid turn number.")
    if not isinstance(audio_bytes, int) or not 0 < audio_bytes <= MAX_AUDIO_BYTES:
        raise ProtocolError("Invalid audio length.")

    body = data[start + header_len:]
    if len(body) < audio_bytes:
        raise ProtocolError("Truncated audio.")
    audio, image = body[:audio_bytes], body[audio_bytes:]
    if not audio.startswith(b"RIFF"):
        raise ProtocolError("Audio must be WAV.")
    if len(image) > MAX_IMAGE_BYTES:
        raise ProtocolError("Camera frame too large.")
    if image and not image.startswith(b"\xff\xd8"):
        raise ProtocolError("Camera frame must be JPEG.")

    return Utterance(
        turn=turn,
        audio=audio,
        image=image or None,
        camera=bool(header.get("camera")),
        facing=_facing(header.get("facing")),
        detections=_detections(header.get("detections")),
    )


def encode_utterance(utterance: Utterance) -> bytes:
    """Client-side encoding; used by tests and the end-to-end probe script."""
    header = json.dumps({
        "turn": utterance.turn,
        "audio_bytes": len(utterance.audio),
        "camera": utterance.camera,
        "facing": utterance.facing,
        "detections": [{"label": d.label, "score": d.score} for d in utterance.detections],
    }).encode("utf-8")
    return (
        _ENVELOPE.pack(KIND_UTTERANCE, len(header))
        + header
        + utterance.audio
        + (utterance.image or b"")
    )


def encode_speech(turn: int, seq: int, wav: bytes) -> bytes:
    return _SPEECH.pack(KIND_SPEECH, turn, seq) + wav


def decode_speech(data: bytes) -> tuple[int, int, bytes]:
    kind, turn, seq = _SPEECH.unpack_from(data)
    if kind != KIND_SPEECH:
        raise ProtocolError("Unknown media message.")
    return turn, seq, data[_SPEECH.size:]


def parse_control(text: str) -> dict:
    """Validate a JSON control message and normalise its fields."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        raise ProtocolError("Malformed message.") from None
    if not isinstance(msg, dict):
        raise ProtocolError("Malformed message.")

    kind = msg.get("type")
    if kind == "hello":
        if msg.get("v") != VERSION:
            raise ProtocolError("Please refresh the page to use the latest version.")
        return {"type": "hello"}
    if kind == "interrupt":
        turn, played = msg.get("turn"), msg.get("played")
        if not isinstance(turn, int) or not isinstance(played, int) or played < 0:
            raise ProtocolError("Invalid interrupt.")
        return {"type": "interrupt", "turn": turn, "played": played}
    if kind == "bye":
        return {"type": "bye"}
    raise ProtocolError("Unknown message type.")
