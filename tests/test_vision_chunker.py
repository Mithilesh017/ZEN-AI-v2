"""Tests for splitting a streamed reply into speakable chunks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.vision.chunker import SentenceChunker, clean_for_speech


def stream(text, step=3):
    chunker = SentenceChunker()
    out = []
    for i in range(0, len(text), step):
        out += chunker.feed(text[i:i + step])
    return out + chunker.flush()


def test_first_sentence_goes_alone_then_sentences_are_grouped():
    # One TTS request per chunk: the first is fast, the rest are batched.
    assert stream("That's a cartridge valve. You'll need a spanner! Ready?") == [
        "That's a cartridge valve.", "You'll need a spanner! Ready?",
    ]


def test_grouped_chunks_close_once_long_enough():
    text = ("Sure. " + "This sentence is here to fill out the second chunk nicely. "
            + "Another one makes it long enough to close. And a short tail.")
    chunks = stream(text)
    assert chunks[0] == "Sure."
    assert chunks[1].endswith("long enough to close.") and len(chunks[1]) >= 80
    assert chunks[2] == "And a short tail."


def test_does_not_split_decimals():
    assert stream("It costs 3.50 today. Cheap.") == ["It costs 3.50 today.", "Cheap."]


def test_first_chunk_may_end_at_a_clause_for_faster_audio():
    chunks = stream("Looking at the label on that bottle, it says olive oil. Good choice.")
    assert chunks[0] == "Looking at the label on that bottle,"
    assert chunks[1] == "it says olive oil. Good choice."


def test_later_chunks_wait_for_full_sentences():
    chunks = stream("Yes. Then, after that, you rinse it.")
    assert chunks == ["Yes.", "Then, after that, you rinse it."]


def test_chunks_never_exceed_the_limit():
    long = "word " * 120 + "end."
    chunks = stream(long)
    assert all(len(c) <= 180 for c in chunks)
    assert " ".join(chunks).split() == long.split()


def test_strips_markdown_the_model_should_not_have_written():
    assert clean_for_speech("**Bold** and `code`  here") == "Bold and code here"
    assert stream("## Step one. *Done*.") == ["Step one.", "Done."]


def test_flush_returns_unterminated_tail():
    chunker = SentenceChunker()
    assert chunker.feed("No punctuation at all") == []
    assert chunker.flush() == ["No punctuation at all"]
    assert chunker.flush() == []
