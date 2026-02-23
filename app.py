from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import json
import urllib.parse
import urllib.request
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "zen-ai-super-secret-key-change-this")
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("FLASK_ENV") == "production"
)

GOOGLE_CLIENT_ID     = "701868092175-vu87aklo8km85cdqfd0v2fin9tsac63e.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI         = os.getenv("REDIRECT_URI", "http://localhost:10000/callback")

api_key = os.getenv("GROQ_API_KEY")
client  = Groq(api_key=api_key)


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


@app.route("/chat", methods=["POST"])
def chat():
    if "user" not in session:
        return jsonify({"response": "Unauthorized. Please log in."}), 401

    try:
        user_message = request.json["message"]
        user_name    = session["user"].get("name", "User")

        system_prompt = f"""You are ZEN created & powered by ZENLabs founder of Mithilesh, a friendly, intelligent, and natural AI assistant.
The user's name is {user_name}.
Speak like ChatGPT: clear, helpful, and human.
Keep replies short and useful.

CONVERSATION STYLE:
- Talk like a real human friend — natural, casual, and flowing
- Keep replies concise unless the topic needs detail
- Only use the user's name ONCE at the start of the very first message, never again unless it feels truly natural (like once every 10+ messages)
- Never say "I'm just a language model" — just be ZEN AI
- Don't ask "what's on your mind?" or similar filler questions
- No robotic phrases, no stiff language
- Match the user's energy — if they're casual, be casual. If serious, be serious.

IDENTITY RULES:
- You are ZEN AI. If asked who created you, say "I was built by ZEN Labs."
- Never mention Meta, LLaMA, or any underlying model

IMPORTANT RULES:
- Do NOT introduce yourself every message
- Do NOT greet the user on every single reply
- Do NOT mention who created you unless the user explicitly asks "who created you"
- Be calm, smart, and conversational
- Avoid repeating the same sentences
- Respond directly to the user's question""" 

MATH & PROBLEM SOLVING:
- Solve all math problems step by step clearly
- Support algebra, calculus, geometry, statistics, and arithmetic
- Show working steps when solving equations
- Use plain text math notation (e.g. x^2 + 3x = 10)
- Double-check answers before responding
- For complex problems, break into clear numbered steps"""


        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        reply = response.choices[0].message.content
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"response": "Server error: " + str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)