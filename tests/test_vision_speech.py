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


# ── ZEN's own voice server ─────────────────────────────────

import requests

from models.vision.speech import OwnVoice


class FakeHttp:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.posts = []

    def post(self, url, json, headers, timeout):
        self.posts.append((url, json, headers))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        status, body = outcome
        return SimpleNamespace(status_code=status, content=body)


def own_voice(outcomes, clock=None):
    http = FakeHttp(outcomes)
    breaker = VoiceBreaker(clock) if clock else VoiceBreaker()
    return OwnVoice("https://voice.example", "t" * 64, "af_heart", http=http, breaker=breaker), http


def test_own_voice_is_used_first_and_sends_the_token():
    voice, http = own_voice([(200, b"RIFFkokoro")])
    tts, speech, _ = engines([b"RIFForpheus"])
    tts._own_voice = voice
    assert tts.synthesize("hi") == b"RIFFkokoro"
    assert speech.calls == 0
    url, body, headers = http.posts[0]
    assert url == "https://voice.example/v1/audio/speech"
    assert body["voice"] == "af_heart" and headers["Authorization"] == "Bearer " + "t" * 64


def test_busy_own_voice_overflows_one_sentence_to_groq():
    voice, _ = own_voice([(503, b""), (200, b"RIFFkokoro")])
    tts, speech, _ = engines([b"RIFForpheus"])
    tts._own_voice = voice
    assert tts.synthesize("one") == b"RIFForpheus"
    assert tts.synthesize("two") == b"RIFFkokoro"  # not paused by a busy reply
    assert speech.calls == 1


def test_unreachable_own_voice_is_skipped_for_a_while():
    now = [0.0]
    voice, http = own_voice([requests.ConnectionError("down"), (200, b"RIFFkokoro")], clock=lambda: now[0])
    tts, speech, _ = engines([b"RIFForpheus", b"RIFForpheus"])
    tts._own_voice = voice
    assert tts.synthesize("one") == b"RIFForpheus"
    assert tts.synthesize("two") == b"RIFForpheus"
    assert len(http.posts) == 1  # no second wait on a dead server

    now[0] = OwnVoice.OUTAGE_PAUSE + 1
    assert tts.synthesize("three") == b"RIFFkokoro"


def test_voice_is_available_while_either_voice_is():
    now = [0.0]
    voice, _ = own_voice([], clock=lambda: now[0])
    tts, _, _ = engines([])
    tts._own_voice = voice
    tts._breaker.trip(3600)          # Groq quota spent
    assert tts.voice_available()     # own server still speaks
    voice._breaker.trip(30)
    assert not tts.voice_available()


# ── Reply fallback: Groq, then Gemini ──────────────────────

from models.vision.speech import GeminiReply, ModelBusy


def source(name, outcomes):
    """A reply source that raises or streams, one outcome per call."""
    outcomes = list(outcomes)
    calls = []

    def run(messages):
        calls.append(messages)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        yield from outcome

    run.name = name
    run.calls = calls
    return run


def reply_engines(primary, *fallbacks, clock=None):
    tts = GroqEngines(SimpleNamespace(), fallback_replies=fallbacks, **({"clock": clock} if clock else {}))
    tts._replies[0] = (primary, tts._replies[0][1])
    return tts


def test_primary_reply_source_is_used_when_it_answers():
    groq, gemini = source("groq", [["Hi ", "there."]]), source("gemini", [])
    assert list(reply_engines(groq, gemini).reply([])) == ["Hi ", "there."]
    assert gemini.calls == []


def test_rate_limited_primary_falls_over_and_is_skipped_until_it_resets():
    now = [0.0]
    groq = source("groq", [ModelBusy(300), ["back"]])
    gemini = source("gemini", [["one"], ["two"]])
    tts = reply_engines(groq, gemini, clock=lambda: now[0])

    assert list(tts.reply([])) == ["one"]
    assert list(tts.reply([])) == ["two"]
    assert len(groq.calls) == 1          # not asked again while limited

    now[0] = 301
    assert list(tts.reply([])) == ["back"]


def test_a_failing_source_is_skipped_too():
    groq = source("groq", [RuntimeError("503")])
    gemini = source("gemini", [["ok"]])
    assert list(reply_engines(groq, gemini).reply([])) == ["ok"]


def test_busy_everywhere_reports_the_shortest_wait():
    now = [0.0]
    tts = reply_engines(source("groq", [ModelBusy(300)]), source("gemini", [ModelBusy(40)]), clock=lambda: now[0])
    with pytest.raises(ModelBusy) as err:
        next(tts.reply([]))
    assert err.value.wait == 40


def test_gemini_reply_streams_text_and_maps_rate_limits():
    class Response:
        def __init__(self, status, lines=(), headers=None):
            self.status_code, self._lines, self.headers, self.text = status, lines, headers or {}, "error"

        def iter_lines(self):
            return iter(self._lines)

        def close(self):
            pass

    class Http:
        def __init__(self, response):
            self.response, self.sent = response, None

        def post(self, url, headers, json, stream, timeout):
            self.sent = json
            return self.response

    lines = [b'data: {"choices":[{"delta":{"content":"A red "}}]}', b"",
             b'data: {"choices":[{"delta":{"content":"triangle."}}]}', b"data: [DONE]"]
    http = Http(Response(200, lines))
    reply = GeminiReply("gemini-x", "key", http=http)
    assert "".join(reply([{"role": "user", "content": "hi"}])) == "A red triangle."
    assert http.sent["model"] == "gemini-x" and http.sent["stream"] is True

    with pytest.raises(ModelBusy) as err:
        next(GeminiReply("gemini-x", "key", http=Http(Response(429, headers={"retry-after": "17"})))([]))
    assert err.value.wait == 17

    with pytest.raises(RuntimeError, match="503"):
        next(GeminiReply("gemini-x", "key", http=Http(Response(503)))([]))
