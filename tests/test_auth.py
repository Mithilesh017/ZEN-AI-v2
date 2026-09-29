"""Tests for the Google OAuth flow and chat throttling."""

import sys
import urllib.parse
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as zen


@pytest.fixture
def client():
    zen.app.config["TESTING"] = True
    with zen.app.test_client() as test_client:
        yield test_client


def _state_from_redirect(location):
    query = urllib.parse.urlparse(location).query
    return urllib.parse.parse_qs(query)["state"][0]


def test_google_login_sends_a_state_and_remembers_it(client):
    res = client.get("/google-login")
    assert res.status_code == 302

    state = _state_from_redirect(res.headers["Location"])
    assert len(state) >= 32
    with client.session_transaction() as session:
        assert session["oauth_state"] == state


def test_each_sign_in_gets_a_fresh_state(client):
    first = _state_from_redirect(client.get("/google-login").headers["Location"])
    second = _state_from_redirect(client.get("/google-login").headers["Location"])
    assert first != second


def test_callback_rejects_a_forged_state(client):
    client.get("/google-login")
    res = client.get("/callback?code=attacker-code&state=wrong")

    assert res.headers["Location"].endswith("error=access_denied")
    with client.session_transaction() as session:
        assert "user" not in session


def test_callback_rejects_a_missing_state(client):
    client.get("/google-login")
    res = client.get("/callback?code=attacker-code")

    assert res.headers["Location"].endswith("error=access_denied")
    with client.session_transaction() as session:
        assert "user" not in session


def test_callback_rejects_a_code_with_no_sign_in_started(client):
    # No /google-login first: nothing in this session to match against.
    res = client.get("/callback?code=attacker-code&state=anything")

    assert res.headers["Location"].endswith("error=access_denied")
    with client.session_transaction() as session:
        assert "user" not in session


def test_state_is_single_use(client):
    state = _state_from_redirect(client.get("/google-login").headers["Location"])
    client.get(f"/callback?code=bad&state={state}")      # consumes the state

    res = client.get(f"/callback?code=bad&state={state}")
    assert res.headers["Location"].endswith("error=access_denied")


def test_secret_key_is_not_a_known_constant():
    assert zen.app.secret_key
    assert zen.app.secret_key != "dev-only-fallback-key"


def test_chat_is_rate_limited(client, monkeypatch):
    monkeypatch.setattr(zen, "chat_limiter", zen.RateLimiter([(2, 60)]))
    with client.session_transaction() as session:
        session["user"] = {"name": "Test", "email": "limit@example.test"}

    # A request that fails validation still counts: the limit guards the route.
    for _ in range(2):
        assert client.post("/api/chat", json={"messages": []}).status_code == 400

    res = client.post("/api/chat", json={"messages": []})
    assert res.status_code == 429
    assert int(res.headers["Retry-After"]) > 0
    assert "quickly" in res.get_json()["error"]


def test_rate_limit_is_per_user(client, monkeypatch):
    monkeypatch.setattr(zen, "chat_limiter", zen.RateLimiter([(1, 60)]))

    for email in ("one@example.test", "two@example.test"):
        with client.session_transaction() as session:
            session["user"] = {"name": "Test", "email": email}
        assert client.post("/api/chat", json={"messages": []}).status_code == 400


def test_chat_requires_a_session(client):
    assert client.post("/api/chat", json={"messages": []}).status_code == 401
