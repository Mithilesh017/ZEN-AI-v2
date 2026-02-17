from flask import Flask, render_template, request, jsonify
import os
import re
import requests
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Optional: set NEWS_API_KEY in .env for richer news (free at newsapi.org)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ─────────────────────────────────────────────────────────
#  SYSTEM PROMPT  —  ZEN AI identity + full intelligence
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are ZEN AI — an elite, highly intelligent AI assistant created and developed by ZEN Labs, founded by Mithilesh.

════════════════════════════════════
IDENTITY  (NEVER violate these rules)
════════════════════════════════════
• You were created by ZEN Labs, founded by Mithilesh.
• You are NOT made by Meta, OpenAI, Google, Anthropic, or any other company. Never mention them.
• If asked "who made you?", "who created you?", "who are you?", "what model are you?" or anything similar, always reply:
  "I'm ZEN AI, a smart AI assistant created by ZEN Labs, founded by Mithilesh! I'm here to help you with anything. 😊"
• Never mention LLaMA, Meta, or any underlying model/technology.
• Your name is always ZEN AI. Refer to yourself only as ZEN AI.

════════════════════════════════════
INTELLIGENCE & CAPABILITIES
════════════════════════════════════
You are as capable as the world's top AI models. You excel at:

🔢 MATH (ALL levels):
  - Arithmetic, Algebra, Geometry, Trigonometry
  - Calculus (derivatives, integrals, limits)
  - Statistics, Probability, Linear Algebra
  - Word problems, Equations, Inequalities
  - Always solve step-by-step with clear explanation.
  - Format: Step 1 → Step 2 → ... → ✅ Final Answer

📰 NEWS & CURRENT EVENTS:
  - When [NEWS DATA] is provided in the message, use it to give accurate, up-to-date answers.
  - Summarize news clearly with key points and context.
  - Always mention it's based on the latest available data.

💻 CODING: Python, JavaScript, HTML/CSS, SQL, and more. Write clean, commented code.

🌍 GENERAL KNOWLEDGE: Science, History, Geography, Culture, Technology — answer with depth.

🗣️ CONVERSATION: Be warm, friendly, and helpful. Match the user's tone.

════════════════════════════════════
MATH SOLVING FORMAT
════════════════════════════════════
When solving math problems, ALWAYS use this format:
  📌 Problem: [restate the problem]
  🔍 Solution:
    Step 1: [explain and compute]
    Step 2: [explain and compute]
    ...
  ✅ Final Answer: [clear answer]

════════════════════════════════════
RESPONSE STYLE
════════════════════════════════════
• Be helpful, confident, and concise.
• Use emojis occasionally to be friendly (but not excessive).
• For complex topics, structure with headers or bullet points.
• Never say you "can't" do something — always give your best answer.
• Today's date: {datetime.now().strftime("%B %d, %Y")}
"""


# ─────────────────────────────────────────────
#  QUERY DETECTORS
# ─────────────────────────────────────────────

def is_news_query(message: str) -> bool:
    """Detect if the user is asking about news or current events."""
    patterns = [
        r'\b(news|latest|today|breaking|headlines|current events|what happened|what.s happening)\b',
        r'\b(update|recent|trending|top stories)\b',
        r'\b(tell me about today|whats new|anything new)\b',
    ]
    for p in patterns:
        if re.search(p, message, re.IGNORECASE):
            return True
    return False


# ─────────────────────────────────────────────
#  NEWS FETCHER
# ─────────────────────────────────────────────

def fetch_news(query: str = "top headlines") -> str:
    """Fetch latest news. Uses NewsAPI if key is set, else falls back to Google News RSS."""

    # ── NewsAPI (richer, requires free key from newsapi.org) ──
    if NEWS_API_KEY:
        try:
            is_general = any(w in query.lower() for w in ["top", "latest", "today", "breaking", "headline"])
            if is_general:
                url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=7&apiKey={NEWS_API_KEY}"
            else:
                url = (
                    f"https://newsapi.org/v2/everything"
                    f"?q={requests.utils.quote(query)}&language=en"
                    f"&pageSize=7&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
                )
            resp = requests.get(url, timeout=7)
            articles = resp.json().get("articles", [])
            if articles:
                lines = []
                for a in articles[:7]:
                    title = a.get("title", "").split(" - ")[0].strip()
                    source = a.get("source", {}).get("name", "")
                    desc = a.get("description", "") or ""
                    desc = desc[:120] + "..." if len(desc) > 120 else desc
                    lines.append(f"• **{title}** ({source})\n  {desc}")
                return "\n\n".join(lines)
        except Exception:
            pass  # fall through to RSS

    # ── Google News RSS fallback (no key needed) ──
    try:
        safe_query = requests.utils.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={safe_query}&hl=en&gl=US&ceid=US:en"
        resp = requests.get(rss_url, timeout=7)
        if resp.status_code == 200:
            # Extract titles from RSS
            titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', resp.text)
            if not titles:
                titles = re.findall(r'<title>(.*?)</title>', resp.text)
            titles = [re.sub(r'<[^>]+>', '', t).strip() for t in titles]
            # Skip feed-level title (first item)
            titles = [t for t in titles if t and "Google News" not in t][:7]
            if titles:
                return "\n".join(f"• {t}" for t in titles)
    except Exception:
        pass

    return ""  # couldn't fetch news


# ─────────────────────────────────────────────
#  CONVERSATION MEMORY (in-memory per session)
# ─────────────────────────────────────────────
conversation_histories: dict[str, list] = {}


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        body = request.json or {}
        user_message = body.get("message", "").strip()
        session_id   = body.get("session_id", "default")

        if not user_message:
            return jsonify({"response": "Please send a message!"})

        # ── Retrieve or init conversation history ──
        if session_id not in conversation_histories:
            conversation_histories[session_id] = []
        history = conversation_histories[session_id]

        # ── News augmentation ──
        augmented_message = user_message
        if is_news_query(user_message):
            # Try to extract specific topic
            topic_match = re.search(
                r'(?:news about|news on|latest on|update on|headlines about|tell me about)\s+(.+)',
                user_message, re.IGNORECASE
            )
            topic = topic_match.group(1).strip() if topic_match else "top headlines"
            news_data = fetch_news(topic)
            if news_data:
                augmented_message = (
                    f"{user_message}\n\n"
                    f"[NEWS DATA — use this to answer accurately]\n{news_data}"
                )

        # ── Build messages for Groq ──
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-12:])   # last 6 turns of context
        messages.append({"role": "user", "content": augmented_message})

        # ── Call Groq API ──
        groq_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.65,
            max_tokens=2048,
        )

        reply = groq_response.choices[0].message.content

        # ── Save to history (store clean message, not augmented) ──
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": reply})

        # Trim history to prevent unbounded growth (keep last 40 entries = 20 turns)
        if len(history) > 40:
            conversation_histories[session_id] = history[-40:]

        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"response": f"⚠️ ZEN AI encountered an error: {str(e)}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)