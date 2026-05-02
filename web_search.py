# ============================================================
# web_search.py — Real-Time Web Search (Feature 2)
# ============================================================
#
# Uses DuckDuckGo HTML search (no API key required).
# Only uses the `requests` library (already in requirements.txt).
# ============================================================

import re
import requests

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


# ── Core search function ──────────────────────────────────────────

def search_web(query: str) -> str:
    """
    Search the web for `query` and return a plain-text summary
    suitable for feeding back to the LLM as tool output.

    Strategy (in order):
      1. DuckDuckGo HTML lite — parse titles + snippets.
      2. DuckDuckGo Instant Answer API — for direct facts.
      3. Graceful fallback message.
    """
    if not query or not query.strip():
        return "No search query provided."

    # ---------- Attempt 1: DuckDuckGo HTML Lite ----------
    results = _search_ddg_html(query)
    if results:
        header = f"Web search results for: {query}\n" + "=" * 50 + "\n\n"
        body = "\n\n".join(results[:6])
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


def _search_ddg_html(query: str) -> list:
    """
    Fetch search results from DuckDuckGo HTML lite and parse
    out titles, URLs, and snippets.
    """
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "b": ""},
            headers=_HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            return []

        html = resp.text
        results = []

        # Parse result blocks — each result has a title link and a snippet
        # Title links: <a class="result__a" href="...">Title</a>
        # Snippets: <a class="result__snippet" ...>Snippet text</a>
        #        or <td class="result__snippet">...</td>

        # Find all result blocks
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

        for i in range(min(len(titles), len(snippets), 8)):
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
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
