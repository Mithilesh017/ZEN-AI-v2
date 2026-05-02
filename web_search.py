# NEW FILE — do not modify existing files
# ============================================================
# web_search.py — Real-Time Web Search (Feature 2)
# ============================================================
#
# Uses the DuckDuckGo Instant Answer API (no API key required)
# with a fallback to DuckDuckGo HTML search for richer results.
# Only uses the `requests` library (already in requirements.txt).
#
# INTEGRATION INSTRUCTIONS (additive changes to app.py):
#
# 1. Add this import at the top of app.py:
#
#        from web_search import search_web, WEB_SEARCH_TOOL_DEFINITION
#
# 2. Right after the existing `tools = [...]` list, EXTEND it:
#
#        tools.append(WEB_SEARCH_TOOL_DEFINITION)
#
# 3. Inside the /chat route's tool-call loop, add an elif branch
#    right after the `if tool_call.function.name == "get_current_datetime":`
#    block:
#
#        elif tool_call.function.name == "search_web":
#            import json as _json
#            args = _json.loads(tool_call.function.arguments)
#            tool_result = search_web(args.get("query", ""))
#
# That's it — three additions, zero modifications to existing lines.
# ============================================================

import requests


# ── Groq tool definition (ready to append to the tools list) ──────

WEB_SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for real-time information on any topic. "
            "Use this when the user asks about current events, live data, "
            "recent news, or anything you don't have up-to-date knowledge about."
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


# ── Core search function ──────────────────────────────────────────

def search_web(query: str) -> str:
    """
    Search the web for `query` and return a plain-text summary
    suitable for feeding back to the LLM as tool output.

    Strategy:
      1. Try DuckDuckGo Instant Answer API (fast, structured).
      2. If that yields nothing useful, fall back to DuckDuckGo
         HTML lite search and scrape the top results.
    """
    if not query or not query.strip():
        return "No search query provided."

    # ---------- Attempt 1: DuckDuckGo Instant Answer API ----------
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            headers={"User-Agent": "ZenAI/1.0"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            parts = []

            # Abstract (Wikipedia-style summary)
            if data.get("AbstractText"):
                parts.append(data["AbstractText"])
                if data.get("AbstractSource"):
                    parts.append(f"(Source: {data['AbstractSource']})")

            # Answer (direct computation / fact)
            if data.get("Answer"):
                parts.append(f"Answer: {data['Answer']}")

            # Related topics (first 5)
            for topic in (data.get("RelatedTopics") or [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    parts.append(f"• {topic['Text']}")

            if parts:
                return "\n".join(parts)
    except Exception:
        pass  # Fall through to method 2

    # ---------- Attempt 2: DuckDuckGo HTML Lite ----------
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
            timeout=10,
        )
        if resp.status_code == 200:
            results = _parse_ddg_html(resp.text)
            if results:
                return "\n\n".join(results[:5])
    except Exception:
        pass

    return f"I couldn't find any web results for \"{query}\". Try rephrasing the query."


def _parse_ddg_html(html: str) -> list:
    """
    Minimal parser to extract search result snippets from
    DuckDuckGo HTML lite (no BeautifulSoup dependency).
    """
    results = []
    # Each result snippet lives inside <a class="result__snippet">
    snippet_marker = 'class="result__snippet"'
    idx = 0
    while True:
        pos = html.find(snippet_marker, idx)
        if pos == -1:
            break
        # Find the closing tag after the marker
        tag_start = html.find(">", pos)
        if tag_start == -1:
            break
        tag_end = html.find("</a>", tag_start)
        if tag_end == -1:
            tag_end = html.find("</span>", tag_start)
        if tag_end == -1:
            break
        snippet = html[tag_start + 1: tag_end]
        # Strip remaining HTML tags
        clean = _strip_tags(snippet).strip()
        if clean:
            results.append(clean)
        idx = tag_end + 1
    return results


def _strip_tags(text: str) -> str:
    """Remove HTML tags from a string (lightweight, no deps)."""
    import re
    return re.sub(r"<[^>]+>", "", text)
