"""
WebSocket endpoint for live calls: GET /api/vision/call (upgrade).

Authentication reuses the Flask session cookie sent with the upgrade request,
so a call is tied to the signed-in account exactly like /api/chat. Every
connection is handled on its own thread (gunicorn gthread worker); the turn
pipeline runs on further threads, so all sends go through one lock.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.parse

from flask import request, session
from simple_websocket import ConnectionClosed

from rate_limit import RateLimiter

from .protocol import ProtocolError, parse_control, parse_utterance
from .session import CallLimits, CallSession
from .speech import GroqEngines, gemini_replies_from_env, own_voice_from_env

logger = logging.getLogger("zen-ai.vision")

HELLO_TIMEOUT = 10        # seconds to send "hello" after connecting
POLL_INTERVAL = 1.0       # how often the loop wakes to check limits

# Close codes in the application range (4000-4999), mirrored by the client.
CLOSE_UNAUTHORIZED = 4401
CLOSE_FORBIDDEN = 4403
CLOSE_RATE_LIMITED = 4429


def _limits() -> CallLimits:
    return CallLimits(max_seconds=int(os.getenv("CALL_MAX_MINUTES", "15")) * 60)


call_limiter = RateLimiter([(int(os.getenv("CALL_STARTS_PER_HOUR", "10")), 3600)])
# Generous for a person, but stops a script from looping utterances.
turn_limiter = RateLimiter([(int(os.getenv("CALL_TURNS_PER_MINUTE", "30")), 60)])


class _ActiveCalls:
    """One live call per account: a new call (another tab, a reload) ends the old one."""

    def __init__(self):
        self._lock = threading.Lock()
        self._calls: dict[str, threading.Event] = {}

    def claim(self, email: str) -> threading.Event:
        superseded = threading.Event()
        with self._lock:
            previous = self._calls.get(email)
            if previous:
                previous.set()
            self._calls[email] = superseded
        return superseded

    def release(self, email: str, token: threading.Event) -> None:
        with self._lock:
            if self._calls.get(email) is token:
                del self._calls[email]


active_calls = _ActiveCalls()


def same_origin(origin: str | None, host: str | None) -> bool:
    """
    Browsers attach cookies to cross-site WebSocket upgrades in some
    configurations, so the Origin header is checked explicitly
    (cross-site WebSocket hijacking).
    """
    if not origin or not host:
        return False
    return urllib.parse.urlsplit(origin).netloc.lower() == host.lower()


def register_vision_routes(app, sock, *, client, recall=None, remember=None) -> None:
    app.config.setdefault("SOCK_SERVER_OPTIONS", {"ping_interval": 20, "max_message_size": 1_500_000})
    engines = GroqEngines(client, own_voice=own_voice_from_env(),
                          fallback_replies=gemini_replies_from_env())

    @sock.route("/api/vision/call")
    def vision_call(ws):
        user = session.get("user")
        if not user or not user.get("email"):
            ws.close(CLOSE_UNAUTHORIZED, "Please sign in again.")
            return
        if not same_origin(request.headers.get("Origin"), request.host):
            logger.warning("Rejected call from origin %r", request.headers.get("Origin"))
            ws.close(CLOSE_FORBIDDEN, "Origin not allowed.")
            return

        email = user["email"]
        retry_after = call_limiter.check(email)
        if retry_after is not None:
            ws.close(CLOSE_RATE_LIMITED, f"Too many calls. Try again in {retry_after // 60 + 1} minutes.")
            return

        send_lock = threading.Lock()

        def send(data) -> None:
            with send_lock:
                try:
                    ws.send(data)
                except ConnectionClosed:
                    pass

        call = CallSession(
            user_name=session.get("display_name") or user.get("name") or "there",
            email=email,
            engines=engines,
            send_json=lambda msg: send(_json(msg)),
            send_bytes=send,
            recall=recall,
            remember=remember,
            limits=_limits(),
        )
        superseded = active_calls.claim(email)
        logger.info("Call started for %s", email)
        try:
            _serve(ws, call, send, email, superseded)
        finally:
            call.close()
            active_calls.release(email, superseded)
            logger.info("Call ended for %s", email)


def _serve(ws, call: CallSession, send, email: str, superseded: threading.Event) -> None:
    first = ws.receive(timeout=HELLO_TIMEOUT)
    try:
        if not isinstance(first, str) or parse_control(first)["type"] != "hello":
            raise ProtocolError("Expected hello.")
    except ProtocolError as err:
        send(_json({"type": "error", "code": "protocol", "message": str(err)}))
        return
    send(_json(call.ready_message()))

    while True:
        if superseded.is_set():
            send(_json({"type": "ended", "reason": "replaced"}))
            return
        if call.remaining_seconds() <= 0:
            send(_json({"type": "ended", "reason": "time_limit"}))
            return

        data = ws.receive(timeout=POLL_INTERVAL)
        if data is None:
            continue
        try:
            if isinstance(data, bytes):
                utterance = parse_utterance(data)
                if turn_limiter.check(email) is not None:
                    send(_json({"type": "notice", "code": "slow_down",
                                "message": "Give me a second to catch up."}))
                    continue
                call.handle_utterance(utterance)
                continue

            msg = parse_control(data)
            if msg["type"] == "interrupt":
                call.interrupt(msg["turn"], msg["played"])
            elif msg["type"] == "bye":
                send(_json({"type": "ended", "reason": "hangup"}))
                return
        except ProtocolError as err:
            send(_json({"type": "error", "code": "protocol", "message": str(err)}))


def _json(msg: dict) -> str:
    return json.dumps(msg, separators=(",", ":"))
