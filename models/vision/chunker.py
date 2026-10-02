"""
Split a streaming LLM reply into speakable chunks.

Text-to-speech is the slowest stage of a turn, so each sentence is handed to
it the moment it is complete instead of waiting for the whole reply. The first
chunk may end at a clause boundary to get audio playing sooner, and every
chunk respects the TTS engine's input limit.
"""

from __future__ import annotations

import re

MAX_CHUNK = 180          # Orpheus rejects longer inputs
FIRST_CHUNK_MIN = 24     # an early clause break is allowed once this long

_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*(?=\s)")
_CLAUSE_END = re.compile(r"[,;:—–](?=\s)")
# Characters that never belong in speech (markdown the model may still emit).
_UNSPOKEN = re.compile(r"[*_`#>|~]+")


def clean_for_speech(text: str) -> str:
    text = _UNSPOKEN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceChunker:
    def __init__(self, max_chunk: int = MAX_CHUNK, first_min: int = FIRST_CHUNK_MIN):
        self._buffer = ""
        self._max = max_chunk
        self._first_min = first_min
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        """Add streamed text; return any chunks that are now complete."""
        self._buffer += delta
        chunks = []
        while (chunk := self._next()) is not None:
            chunks.append(chunk)
        return chunks

    def flush(self) -> list[str]:
        """Return whatever is left once the stream has ended."""
        rest = clean_for_speech(self._buffer)
        self._buffer = ""
        return self._split_long(rest) if rest else []

    def _next(self) -> str | None:
        cut = self._find_cut()
        if cut is None:
            return None
        piece, self._buffer = self._buffer[:cut], self._buffer[cut:].lstrip()
        piece = clean_for_speech(piece)
        if not piece:
            return self._next()
        self._emitted += 1
        return piece

    def _find_cut(self) -> int | None:
        text = self._buffer
        match = _SENTENCE_END.search(text)
        if match and match.end() <= self._max:
            return match.end()

        if self._emitted == 0:
            clause = _CLAUSE_END.search(text, self._first_min)
            if clause and clause.end() <= self._max:
                return clause.end()

        if len(text) > self._max:
            return self._soft_break(text[: self._max])
        return None

    @staticmethod
    def _soft_break(window: str) -> int:
        for pattern in (_CLAUSE_END, re.compile(r"\s")):
            positions = [m.end() for m in pattern.finditer(window)]
            if positions:
                return positions[-1]
        return len(window)

    def _split_long(self, text: str) -> list[str]:
        pieces = []
        while len(text) > self._max:
            cut = self._soft_break(text[: self._max])
            pieces.append(text[:cut].strip())
            text = text[cut:].strip()
        if text:
            pieces.append(text)
        return pieces
