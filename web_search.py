# ============================================================
# web_search.py — Real-Time Web Search (Feature 2)
# ============================================================
#
# Uses DuckDuckGo HTML search (no API key required).
# Auto-detects "news/latest/recent" queries and applies date
# filters so results are actually current.
# Only uses the `requests` library (already in requirements.txt).
# ============================================================

import re
import requests
from datetime import datetime, timezone

# ── Groq tool definition (ready to append to the tools list) ──────

WEB_SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for real-time information. "
            "Use this when the user asks about current events, live scores, "
            "recent news, weather, stock prices, or anything that requires "
            "up-to-date information beyond your training data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                }
            },
            "required": ["query"],
        },
    },
}

# ── User-Agent to avoid being blocked ─────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Keywords that signal the user wants RECENT results ────────────

_RECENCY_KEYWORDS = [
    "latest", "recent", "today", "news", "current", "now",
    "this week", "this month", "update", "updates", "breaking",
    "new", "just", "live", "trending", "2026", "2025",
    "stock", "price", "score", "weather", "launch", "release",
    "announced", "launched", "worth", "how much is",
]


def _needs_recency_filter(query: str) -> str:
    """
    Detect if the query is asking for recent/current information.
    Returns DuckDuckGo date filter value:
      - 'w'  = past week  (for news/latest/today queries)
      - 'm'  = past month (for broader recent queries)
      - ''   = no filter  (for factual/evergreen queries)
    """
    q = query.lower()

    # Strong recency signals → past week
    strong = ["today", "latest", "breaking", "live", "this week",
              "news", "score", "weather", "stock", "price", "worth",
              "how much is", "trending"]
    if any(kw in q for kw in strong):
        return "w"

    # Moderate recency signals → past month
    moderate = ["recent", "new", "update", "updates", "current",
                "launch", "launched", "release", "announced",
                "2026", "2025", "this month"]
    if any(kw in q for kw in moderate):
        return "m"

    return ""


# ── Core search function ──────────────────────────────────────────

def search_web(query: str) -> str:
    """
    Search the web for `query` and return a plain-text summary
    suitable for feeding back to the LLM as tool output.

    Strategy (in order):
      1. DuckDuckGo HTML lite with auto date-filter — parse titles + snippets.
      2. DuckDuckGo Instant Answer API — for direct facts.
      3. Graceful fallback message.
    """
    if not query or not query.strip():
        return "No search query provided."

    # Determine if we need date filtering for freshness
    date_filter = _needs_recency_filter(query)

    # ---------- Attempt 1: DuckDuckGo HTML (with date filter) ----------
    results = _search_ddg_html(query, date_filter=date_filter)

    # If date-filtered search returned too few results, retry without filter
    if len(results) < 2 and date_filter:
        results = _search_ddg_html(query, date_filter="")

    if results:
        now = datetime.now(timezone.utc).strftime("%B %d, %Y")
        header = (
            f"Web search results for: {query}\n"
            f"Search date: {now}\n"
            + "=" * 50 + "\n\n"
        )
        body = "\n\n".join(results[:8])
        return header + body

    # ---------- Attempt 2: DuckDuckGo Instant Answer API ----------
    instant = _search_ddg_instant(query)
    if instant:
        return instant

    return (
        f"I searched the web for \"{query}\" but couldn't retrieve results right now. "
        "The search service may be temporarily unavailable. "
        "Try rephrasing the query or asking again in a moment."
    )


def _search_ddg_html(query: str, date_filter: str = "") -> list:
    """
    Fetch search results from DuckDuckGo HTML lite and parse
    out titles, URLs, and snippets.

    Args:
        query: Search query string.
        date_filter: DuckDuckGo date filter ('d'=day, 'w'=week, 'm'=month, ''=none).
    """
    try:
        post_data = {"q": query, "b": ""}
        if date_filter:
            post_data["df"] = date_filter

        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data=post_data,
            headers=_HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            return []

        html = resp.text
        results = []

        # Parse result blocks
        title_pattern = re.compile(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL
        )
        snippet_pattern = re.compile(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|td|span)>',
            re.DOTALL
        )

        titles = title_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i in range(min(len(titles), len(snippets), 10)):
            url = titles[i][0]
            title = _strip_tags(titles[i][1]).strip()
            snippet = _strip_tags(snippets[i]).strip()

            if title and snippet:
                # Clean up DuckDuckGo redirect URLs
                if "uddg=" in url:
                    try:
                        from urllib.parse import unquote, parse_qs, urlparse
                        parsed = parse_qs(urlparse(url).query)
                        url = unquote(parsed.get("uddg", [url])[0])
                    except Exception:
                        pass

                results.append(f"[{i+1}] {title}\n{snippet}\nSource: {url}")

        return results

    except Exception:
        return []


def _search_ddg_instant(query: str) -> str:
    """
    Try DuckDuckGo Instant Answer API for direct factual answers.
    """
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            },
            headers={"User-Agent": "ZenAI/1.0"},
            timeout=8,
        )
        if resp.status_code != 200:
            return ""

        data = resp.json()
        parts = []

        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
            if data.get("AbstractSource"):
                parts.append(f"(Source: {data['AbstractSource']})")

        if data.get("Answer"):
            parts.append(f"Answer: {data['Answer']}")

        for topic in (data.get("RelatedTopics") or [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(f"• {topic['Text']}")

        return "\n".join(parts) if parts else ""

    except Exception:
        return ""


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#x27;", "'")
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
