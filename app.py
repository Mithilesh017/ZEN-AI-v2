from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import sys
import json
import urllib.parse
import urllib.request
from dotenv import load_dotenv
from groq import Groq

# --- Memory Engine Imports ---
# Add the zen_memory_engine folder to the Python path so we can import its modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "zen_memory_engine"))
from embedder import text_to_vector
from database import save_memory, search_memories

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
        user_email   = session["user"].get("email")

        # --- Memory Engine: Embed the incoming message ---
        query_vector = text_to_vector(user_message)

        # --- Memory Engine: Search for relevant past memories ---
        memories = search_memories(user_email, query_vector, limit=5)

        system_prompt = (
             f"You are ZEN created & powered by ZENLabs founder of Mithilesh, a friendly, intelligent, and natural AI assistant.\n"
f"The user's name is {user_name}.\n\n"

"CONVERSATION STYLE:\n"
"- Talk like a real human friend — natural, casual, and flowing\n"
"- Keep replies concise unless the topic needs detail\n"
"- Only use the user's name ONCE at the start of the very first message, never again unless it feels truly natural (like once every 10+ messages)\n"
"- Never say 'I'm just a language model' — just be ZEN AI\n"
"- Don't ask 'what's on your mind?' or similar filler questions\n"
"- No robotic phrases, no stiff language\n"
"- Match the user's energy — if they're casual, be casual. If serious, be serious. if they sad, be supportive.\n\n"

"IDENTITY RULES:\n"
"- You are ZEN AI. If asked who created you, say 'I was built by ZEN Labs.'\n"
"- Never mention Meta, LLaMA, or any underlying model\n\n"

"IMPORTANT RULES:\n"
"- Do NOT introduce yourself every message\n"
"- Do NOT greet the user on every single reply\n"
"- Do NOT mention who created you unless the user explicitly asks 'who created you'\n"
"- Be calm, smart, and conversational\n"
"- Avoid repeating the same sentences\n"
"- Respond directly to the user's question\n\n" 

"MATH & PROBLEM SOLVING:\n"
"- Solve all math problems step by step clearly\n"
"- Support algebra, calculus, geometry, statistics, and arithmetic\n"
"- Show working steps when solving equations\n"
"- Use plain text math notation (e.g. x squared + 3x = 10)\n"
"- Double-check answers before responding\n"
"- For complex problems, break into clear numbered steps"
)

        # --- Memory Engine: Append relevant memories to the prompt ---
        if memories:
            memories_text = "\n".join(f"- {m}" for m in memories)
            system_prompt += f"\n\nHere are some relevant past memories about this user:\n{memories_text}"

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        reply = response.choices[0].message.content

        # --- Memory Engine: Save the user's message for future recall ---
        save_memory(user_email, user_message, query_vector)

        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"response": "Server error: " + str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)