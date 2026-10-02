"""Tests for text-to-speech rate-limit handling."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.vision.speech import (
    GroqEngines, SpeechUnavailable, VoiceBreaker, parse_reset, rate_limit_wait,
)


class RateLimited(Exception):
    status_code = 429

    def __init__(self, headers):
        super().__init__("rate limited")
        self.response = SimpleNamespace(headers=headers)


class FakeSpeech:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **_):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(read=lambda: outcome)


def engines(outcomes, clock=None):
    speech = FakeSpeech(outcomes)
    client = SimpleNamespace(audio=SimpleNamespace(speech=speech))
    breaker = VoiceBreaker(clock) if clock else VoiceBreaker()
    slept = []
    return GroqEngines(client, breaker=breaker, sleep=slept.append), speech, slept


def test_parses_groq_durations():
    assert parse_reset("19h12m0s") == 19 * 3600 + 12 * 60
    assert parse_reset("800ms") == pytest.approx(0.8)
    assert parse_reset("2") == 2
    assert parse_reset("1m30.5s") == pytest.approx(90.5)
    assert parse_reset(None) is None and parse_reset("soon") is None


def test_daily_reset_only_counts_when_requests_ran_out():
    tokens_only = {"x-ratelimit-reset-tokens": "800ms", "x-ratelimit-reset-requests": "19h",
                   "x-ratelimit-remaining-requests": "12"}
    assert rate_limit_wait(tokens_only) == pytest.approx(0.8)
    requests_out = {**tokens_only, "x-ratelimit-remaining-requests": "0"}
    assert rate_limit_wait(requests_out) == 19 * 3600


def test_short_rate_limit_is_waited_out():
    tts, speech, slept = engines([RateLimited({"retry-after": "1"}), b"RIFFok"])
    assert tts.synthesize("hi") == b"RIFFok"
    assert speech.calls == 2 and slept == [pytest.approx(1.1)]
    assert tts.voice_available()


def test_spent_daily_quota_turns_voice_off_until_reset():
    now = [0.0]
    headers = {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2h"}
    tts, speech, _ = engines([RateLimited(headers)], clock=lambda: now[0])

    with pytest.raises(SpeechUnavailable) as err:
        tts.synthesize("hi")
    assert err.value.reason == "quota"
    assert not tts.voice_available()

    # No further requests are spent while the quota is known to be gone.
    with pytest.raises(SpeechUnavailable):
        tts.synthesize("again")
    assert speech.calls == 1

    now[0] = 2 * 3600 + 1
    assert tts.voice_available()


def test_terms_or_auth_errors_are_permanent():
    class Forbidden(Exception):
        status_code = 400

    tts, _, _ = engines([Forbidden("model_terms_required")])
    with pytest.raises(SpeechUnavailable) as err:
        tts.synthesize("hi")
    assert err.value.reason == "error"
