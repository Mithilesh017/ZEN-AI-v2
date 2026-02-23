from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
from dotenv import load_dotenv
from groq import Groq
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "zen-ai-super-secret-key-change-this")

GOOGLE_CLIENT_ID = "701868092175-vu87aklo8km85cdqfd0v2fin9tsac63e.apps.googleusercontent.com"

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)


# ==================== ROUTES ====================

@app.route("/")
def home():
    # If not logged in, redirect to login page
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/login")
def login():
    # If already logged in, go straight to chat
    if "user" in session:
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/verify-token", methods=["POST"])
def verify_token():
    """Receives Google credential token from frontend, verifies it, creates session."""
    try:
        token = request.json.get("token")

        if not token:
            return jsonify({"success": False, "error": "No token provided"}), 400

        # Verify the token with Google
        id_info = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        # Token is valid — save user info in Flask session
        session["user"] = {
            "name": id_info.get("name"),
            "email": id_info.get("email"),
            "picture": id_info.get("picture")
        }

        return jsonify({"success": True, "user": session["user"]})

    except ValueError as e:
        # Invalid token
        return jsonify({"success": False, "error": "Invalid token: " + str(e)}), 401

    except Exception as e:
        return jsonify({"success": False, "error": "Server error: " + str(e)}), 500


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/chat", methods=["POST"])
def chat():
    # Protect chat — must be logged in
    if "user" not in session:
        return jsonify({"response": "Unauthorized. Please log in."}), 401

    try:
        user_message = request.json["message"]
        user_name = session["user"].get("name", "User")

        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are ZEN AI, a smart digital companion. The user's name is {user_name}."
                },
                {"role": "user", "content": user_message}
            ]
        )

        reply = response.choices[0].message.content
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"response": "Server error: " + str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)