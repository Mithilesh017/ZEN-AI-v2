# ============================================================
# chat_memory.py — In-Session Conversation History
# ============================================================
#
# Keeps recent chat messages per user in memory so the LLM has
# context of the current conversation (e.g. "explain THAT news").
#
# This is NOT the long-term Pinecone memory — it's a short-term,
# in-memory store that resets when the server restarts.
#
# INTEGRATION (3 additions to app.py):
#
#   1. Import:
#        from chat_memory import chat_memory
#
#   2. Before building `messages`, insert history:
#        history = chat_memory.get_history(user_email)
#        messages = [
#            {"role": "system", "content": system_prompt},
#            *history,
#            {"role": "user", "content": user_message}
#        ]
#
#   3. After getting `reply`, save both sides:
#        chat_memory.add_message(user_email, "user", user_message)
#        chat_memory.add_message(user_email, "assistant", reply)
# ============================================================

import threading
from collections import defaultdict


class ChatMemory:
    """
    Thread-safe, in-memory conversation history.
    Stores the last `max_messages` per user (keyed by email).
    Automatically trims older messages to stay within limits.
    """

    def __init__(self, max_messages: int = 20):
        self._store: dict[str, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()
        self._max = max_messages

    def add_message(self, email: str, role: str, content: str) -> None:
        """Add a user or assistant message to the history."""
        if not email or not content:
            return
        with self._lock:
            self._store[email].append({"role": role, "content": content})
            # Keep only the most recent messages
            if len(self._store[email]) > self._max:
                self._store[email] = self._store[email][-self._max:]

    def get_history(self, email: str) -> list[dict]:
        """Return a copy of the conversation history for this user."""
        with self._lock:
            return list(self._store.get(email, []))

    def clear(self, email: str) -> None:
        """Clear conversation history for a user."""
        with self._lock:
            self._store.pop(email, None)


# Singleton instance — import this in app.py
chat_memory = ChatMemory(max_messages=20)
