from flask import (
    Flask, Response, jsonify, redirect, render_template, request,
    send_from_directory, session, stream_with_context, url_for,
)
import os
import sys
import json
import logging
import secrets
import urllib.parse
import urllib.request
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix

# Load env vars BEFORE importing memory engine modules
# so that PINECONE_API_KEY, HF_TOKEN, etc. are available.
load_dotenv()

from groq import Groq

# --- Memory Engine Imports (lazy — no API calls happen at import time) ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "zen_memory_engine"))
from embedder import text_to_vector
from database import save_memory, search_memories

# --- New Feature Imports ---
from timezone_helper import get_current_datetime as get_current_datetime_tz
from web_search import search_web, WEB_SEARCH_TOOL_DEFINITION
from system_prompt import build_system_prompt
from user_context import user_ctx, register_user_context_routes
from rate_limit import RateLimiter

# --- Structured Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zen-ai")

app = Flask(__name__)
# Render terminates HTTPS at its proxy; trust its X-Forwarded-* headers so
# url_for(..., _external=True) builds https:// URLs.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# --- Session signing key ---
# Never fall back to a fixed key: anyone who knows it can forge a session
# cookie for any account and read that user's chats and memories. Production
# refuses to start without one; elsewhere we use a random per-process key, so
# local runs work and the only cost is being logged out on restart.
_secret = os.getenv("SECRET_KEY")
if not _secret:
    if os.getenv("FLASK_ENV") == "production":
        raise RuntimeError("SECRET_KEY environment variable must be set in production")
    _secret = secrets.token_hex(32)
    logger.warning(
        "SECRET_KEY is not set - using a random key for this process. "
        "Sessions will not survive a restart. Set SECRET_KEY in .env."
    )
app.secret_key = _secret

app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("FLASK_ENV") == "production"
)

register_user_context_routes(app)

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "701868092175-vu87aklo8km85cdqfd0v2fin9tsac63e.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
# Optional override. When unset, the callback URL is derived from the
# current host, so the same code works on localhost and in production.
REDIRECT_URI         = os.getenv("REDIRECT_URI", "").strip()

api_key = os.getenv("GROQ_API_KEY")
client  = Groq(api_key=api_key)



tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the user's current date, time, day of the week, timezone, and UTC offset.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]
tools.append(WEB_SEARCH_TOOL_DEFINITION)


# --- Constants ---
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_MESSAGES = 40      # conversation turns the client sends per request
MAX_TOOL_ROUNDS = 3            # tool-call round trips before forcing a plain answer
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "app")

# Per-user chat limits: a short burst rule plus a sustained hourly one. Both are
# far above normal human use and exist to cap runaway scripts and API cost.
chat_limiter = RateLimiter([
    (int(os.getenv("CHAT_RATE_PER_MINUTE", "20")), 60),
    (int(os.getenv("CHAT_RATE_PER_HOUR", "300")), 3600),
])


# ==================== ROUTES ====================

@app.route("/health")
def health():
    """Liveness probe for monitoring and uptime checks."""
    return jsonify({"status": "ok", "service": "zen-ai", "version": "2.1.0"})

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    if not os.path.exists(os.path.join(FRONTEND_DIST, "index.html")):
        return "Frontend not built. Run: cd frontend && npm install && npm run build", 503
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.route("/login")
def login():
    if "user" in session:
        return redirect(url_for("home"))
    return render_template("login.html")


def _redirect_uri():
    return REDIRECT_URI or url_for("callback", _external=True)


OAUTH_TIMEOUT = 10        # seconds per call to Google
OAUTH_STATE_KEY = "oauth_state"


@app.route("/google-login")
def google_login():
    """Redirects browser to Google's OAuth consent screen."""
    # One-time token tying this sign-in to this browser session. Without it an
    # attacker could feed their own ?code= to /callback and log the visitor
    # into the attacker's account (OAuth login CSRF).
    state = secrets.token_urlsafe(32)
    session[OAUTH_STATE_KEY] = state

    params = urllib.parse.urlencode({
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  _redirect_uri(),
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "prompt":        "select_account"
    })
    return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@app.route("/callback")
def callback():
    """Google redirects here with ?code=... after user approves."""
    code  = request.args.get("code")
    error = request.args.get("error")

    # The state is single-use: drop it whatever happens, so a code cannot be
    # replayed against a later sign-in attempt.
    expected_state = session.pop(OAUTH_STATE_KEY, None)

    if error or not code:
        return redirect(url_for("login") + "?error=access_denied")

    state = request.args.get("state", "")
    if not expected_state or not secrets.compare_digest(state, expected_state):
        logger.warning("OAuth callback with missing or mismatched state")
        return redirect(url_for("login") + "?error=access_denied")

    try:
        # Step 1: Exchange code for tokens
        token_data = urllib.parse.urlencode({
            "code":          code,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri":  _redirect_uri(),
            "grant_type":    "authorization_code"
        }).encode()

        token_req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=token_data,
            method="POST"
        )
        with urllib.request.urlopen(token_req, timeout=OAUTH_TIMEOUT) as resp:
            token_json = json.loads(resp.read())

        access_token = token_json.get("access_token")

        # Step 2: Use access token to get user info
        userinfo_req = urllib.request.Request(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        with urllib.request.urlopen(userinfo_req, timeout=OAUTH_TIMEOUT) as resp:
            user_info = json.loads(resp.read())

        if not user_info.get("email"):
            logger.warning("OAuth callback returned no email")
            return redirect(url_for("login") + "?error=server_error")

        # Step 3: Save to Flask session
        session["user"] = {
            "name":    user_info.get("name"),
            "email":   user_info.get("email"),
            "picture": user_info.get("picture")
        }

        return redirect(url_for("home"))

    except Exception:
        logger.exception("OAuth callback error")
        return redirect(url_for("login") + "?error=server_error")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/me")
def api_me():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user = session["user"]
    return jsonify({
        "name":         user.get("name") or "",
        "email":        user.get("email") or "",
        "picture":      user.get("picture"),
        "display_name": session.get("display_name", ""),
    })


@app.route("/api/display_name", methods=["POST"])
def api_display_name():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    name = data.get("display_name")
    if not isinstance(name, str):
        return jsonify({"error": "display_name must be a string"}), 400
    session["display_name"] = name.strip()[:32]
    return jsonify({"display_name": session["display_name"]})


# ==================== CHAT ====================

def _parse_conversation(raw):
    """
    Validate the conversation sent by the client. The browser owns the
    thread (so edits, regenerations and multiple chats stay consistent),
    and sends it as [{role, content}, ...] ending with the new user turn.
    Returns (history, user_message) or raises ValueError with a user-facing reason.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError("Please provide a valid message.")

    conversation = []
    for item in raw[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            raise ValueError("Please provide a valid message.")
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError("Please provide a valid message.")
        content = content.strip()
        if content:
            conversation.append({"role": role, "content": content})

    if not conversation or conversation[-1]["role"] != "user":
        raise ValueError("Message cannot be empty.")
    user_message = conversation[-1]["content"]
    if len(user_message) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Message too long. Please keep it under {MAX_MESSAGE_LENGTH} characters.")
    return conversation[:-1], user_message


def _recall_memories(email, text):
    """Embed the message and fetch related long-term memories. Never fatal."""
    try:
        vector = text_to_vector(text)
        return vector, search_memories(email, vector, limit=5)
    except Exception:
        logger.warning("Memory recall unavailable; continuing without it", exc_info=True)
        return None, []


def _remember(email, text, vector):
    if vector is None:
        return
    try:
        save_memory(email, text, vector)
    except Exception:
        logger.warning("Could not save memory", exc_info=True)


def _run_tool(name, arguments, user_timezone):
    if name == "get_current_datetime":
        return get_current_datetime_tz(user_timezone)
    if name == "search_web":
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        return search_web(args.get("query", "") if isinstance(args, dict) else "")
    return json.dumps({"error": "Unknown tool requested."})


def _parse_tool_args(arguments):
    try:
        args = json.loads(arguments or "{}")
        return args if isinstance(args, dict) else {}
    except json.JSONDecodeError:
        return {}


def _without_tool_calls(messages):
    """
    Rewrite a tool-using conversation for a final, tools-free answer.
    If the history still contains tool calls, gpt-oss keeps trying to call
    tools and Groq rejects the turn ("Tool choice is none, but model called
    a tool"), so tool outputs are handed over as plain context instead.
    """
    results = [m["content"] for m in messages if m["role"] == "tool"]
    kept = [m for m in messages if m["role"] != "tool" and not m.get("tool_calls")]
    if not results:
        return kept
    return kept + [{
        "role": "system",
        "content": (
            "Information gathered with your tools for the latest message:\n\n"
            + "\n\n---\n\n".join(results)
            + "\n\nTools are no longer available. Answer the user now using this information."
        ),
    }]


def _stream_reply(messages, user_timezone):
    """
    Stream the model's answer as events:
      {"type": "text", "delta": str}
      {"type": "tool-call", "id", "name", "args"}   — a tool started
      {"type": "tool-result", "id"}                  — that tool finished
    Tool calls are executed server-side, then the model is called again.
    """
    tools_enabled = True
    for round_no in range(MAX_TOOL_ROUNDS + 1):
        offer_tools = tools_enabled and round_no < MAX_TOOL_ROUNDS
        extra = {"tools": tools, "tool_choice": "auto"} if offer_tools else {}
        text, calls = "", {}
        request_messages = messages if offer_tools else _without_tool_calls(messages)

        try:
            stream = client.chat.completions.create(
                model=MODEL, messages=request_messages, stream=True, **extra
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text += delta.content
                    yield {"type": "text", "delta": delta.content}
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        except Exception as err:
            # Groq sometimes returns 400 "tool_use_failed" when the model
            # generates a malformed tool call. Retry once without tools.
            if text or not offer_tools:
                raise
            logger.warning("Tool call failed, retrying without tools: %s", err)
            tools_enabled = False
            continue

        if not calls:
            return

        ordered = [calls[i] for i in sorted(calls)]
        messages.append({
            "role": "assistant",
            "content": text or None,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in ordered
            ],
        })
        for c in ordered:
            yield {"type": "tool-call", "id": c["id"], "name": c["name"],
                   "args": _parse_tool_args(c["arguments"])}
            messages.append({
                "role": "tool",
                "tool_call_id": c["id"],
                "content": _run_tool(c["name"], c["arguments"], user_timezone),
            })
            yield {"type": "tool-result", "id": c["id"]}


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Stream a reply as newline-delimited JSON events (see _stream_reply)."""
    if "user" not in session:
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    # Throttle before any paid work (Groq, HuggingFace, Pinecone) happens.
    retry_after = chat_limiter.check(session["user"].get("email") or "")
    if retry_after is not None:
        logger.info("Rate limited /api/chat for %s", session["user"].get("email"))
        return (
            jsonify({"error": "You're sending messages too quickly. Please wait a moment and try again."}),
            429,
            {"Retry-After": str(retry_after)},
        )

    data = request.get_json(silent=True) or {}
    try:
        history, user_message = _parse_conversation(data.get("messages"))
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    user_name  = session["user"].get("name", "User")
    user_email = session["user"].get("email")
    timezone   = data.get("timezone")
    if isinstance(timezone, str) and timezone.strip():
        user_ctx.set_timezone(user_email, timezone.strip()[:64])
    user_timezone = user_ctx.get_timezone(user_email)

    # --- Long-term memory: recall related facts, then store this message ---
    vector, memories = _recall_memories(user_email, user_message)
    _remember(user_email, user_message, vector)

    system_prompt = build_system_prompt(user_name)
    if memories:
        memories_text = "\n".join(f"- {m}" for m in memories)
        system_prompt += f"\n\nHere are some relevant past memories about this user:\n{memories_text}"

    display_name = session.get("display_name")
    if display_name:
        system_prompt += f"\n\nCRITICAL INSTRUCTION: The user prefers to be called '{display_name}'. Address them by this name naturally in conversation."

    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_message},
    ]

    def generate():
        try:
            for event in _stream_reply(messages, user_timezone):
                yield json.dumps(event) + "\n"
        except Exception:
            logger.exception("Chat error for user %s", user_email)
            yield json.dumps({"type": "error", "message": "Something went wrong. Please try again."}) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    # Grab the port from the cloud host, but fall back to 10000 for local testing
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
