from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import sys
import json
import urllib.parse
import urllib.request
from datetime import datetime
from dotenv import load_dotenv

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

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "zen-ai-super-secret-key-change-this")
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("FLASK_ENV") == "production"
)

register_user_context_routes(app)

GOOGLE_CLIENT_ID     = "701868092175-vu87aklo8km85cdqfd0v2fin9tsac63e.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI         = os.getenv("REDIRECT_URI", "http://localhost:10000/callback")

api_key = os.getenv("GROQ_API_KEY")
client  = Groq(api_key=api_key)


def get_current_datetime():
    now = datetime.now().astimezone()
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timezone": now.strftime("%Z"),
        "utc_offset": now.strftime("%z")
    })


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


# ==================== ROUTES ====================

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/login")
def login():
    if "user" in session:
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/google-login")
def google_login():
    """Redirects browser to Google's OAuth consent screen."""
    params = urllib.parse.urlencode({
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "prompt":        "select_account"
    })
    return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@app.route("/callback")
def callback():
    """Google redirects here with ?code=... after user approves."""
    code  = request.args.get("code")
    error = request.args.get("error")

    if error or not code:
        return redirect(url_for("login") + "?error=access_denied")

    try:
        # Step 1: Exchange code for tokens
        token_data = urllib.parse.urlencode({
            "code":          code,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri":  REDIRECT_URI,
            "grant_type":    "authorization_code"
        }).encode()

        token_req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=token_data,
            method="POST"
        )
        with urllib.request.urlopen(token_req) as resp:
            token_json = json.loads(resp.read())

        access_token = token_json.get("access_token")

        # Step 2: Use access token to get user info
        userinfo_req = urllib.request.Request(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        with urllib.request.urlopen(userinfo_req) as resp:
            user_info = json.loads(resp.read())

        # Step 3: Save to Flask session
        session["user"] = {
            "name":    user_info.get("name"),
            "email":   user_info.get("email"),
            "picture": user_info.get("picture")
        }

        return redirect(url_for("home"))

    except Exception as e:
        print("OAuth callback error:", e)
        return redirect(url_for("login") + "?error=server_error")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/update_name", methods=["POST"])
def update_name():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    session["display_name"] = data.get("display_name", "").strip()[:32]
    return jsonify({"status": "ok"})


@app.route("/get_display_name")
def get_display_name():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"display_name": session.get("display_name", "")})


@app.route("/chat", methods=["POST"])
def chat():
    if "user" not in session:
        return jsonify({"response": "Unauthorized. Please log in."}), 401

    try:
        user_message = request.json["message"]
        user_name    = session["user"].get("name", "User")
        user_email   = session["user"].get("email")
        user_timezone = request.json.get("timezone") or user_ctx.get_timezone(user_email)

        # --- Memory Engine: Embed the incoming message ---
        query_vector = text_to_vector(user_message)

        # --- Memory Engine: Search for relevant past memories ---
        memories = search_memories(user_email, query_vector, limit=5)

        system_prompt = build_system_prompt(user_name)

        # --- Memory Engine: Append relevant memories to the prompt ---
        if memories:
            memories_text = "\n".join(f"- {m}" for m in memories)
            system_prompt += f"\n\nHere are some relevant past memories about this user:\n{memories_text}"

        # --- Display Name: override how ZEN addresses the user ---
        display_name = session.get("display_name")
        if display_name:
            system_prompt += f"\n\nCRITICAL INSTRUCTION: The user prefers to be called '{display_name}'. Address them by this name naturally in conversation."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # --- First Groq call (with tools enabled) ---
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
        except Exception as tool_err:
            # Groq sometimes returns 400 "tool_use_failed" when the model
            # generates a malformed tool call.  Retry without tools.
            print(f"[ZEN] Tool call failed, retrying without tools: {tool_err}")
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages
            )

        response_message = response.choices[0].message

        # --- Handle tool calls (if any) ---
        if response_message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in response_message.tool_calls
                ]
            })

            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "get_current_datetime":
                    tool_result = get_current_datetime_tz(user_timezone)
                elif tool_call.function.name == "search_web":
                    args = json.loads(tool_call.function.arguments)
                    tool_result = search_web(args.get("query", ""))
                else:
                    tool_result = json.dumps({"error": "Unknown tool requested."})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages
                )
            except Exception as followup_err:
                # If the follow-up also fails, strip tool messages and retry
                print(f"[ZEN] Follow-up failed, retrying clean: {followup_err}")
                clean_messages = [m for m in messages if m["role"] in ("system", "user")]
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=clean_messages
                )

        reply = response.choices[0].message.content

        # --- Memory Engine: Save the user's message for future recall ---
        save_memory(user_email, user_message, query_vector)

        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"response": "Server error: " + str(e)})


if __name__ == "__main__":
    # Grab the port from the cloud host, but fall back to 10000 for local testing
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
