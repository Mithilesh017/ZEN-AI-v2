# NEW FILE — do not modify existing files
# ============================================================
# user_context.py — User Context Store (Feature 4)
# ============================================================
#
# An in-memory dictionary mapping user email → timezone string.
# This lets the backend look up a user's timezone automatically
# without requiring it in every request body.
#
# INTEGRATION INSTRUCTIONS (additive changes to app.py):
#
# 1. Add this import at the top of app.py:
#
#        from user_context import user_ctx, register_user_context_routes
#
# 2. Right after `app = Flask(__name__)` and the app.config block,
#    register the route:
#
#        register_user_context_routes(app)
#
#    This adds a POST /update_timezone endpoint that the browser
#    calls on page load (via timezone_patch.js) to persist the
#    user's timezone server-side.
#
# 3. (Optional) In the /chat route, you can auto-resolve timezone
#    from the store when the request body doesn't include one:
#
#        from user_context import user_ctx
#        user_timezone = request.json.get("timezone") or user_ctx.get_timezone(user_email)
#
#    This means even if the JS patch isn't loaded, the backend
#    can still find the user's last-known timezone.
#
# That's it — zero modifications to existing code.
# ============================================================

from __future__ import annotations

import threading
from typing import Optional


class UserContextStore:
    """
    Thread-safe in-memory store for per-user context.
    Currently stores: timezone (IANA string).
    Easily extendable to store locale, preferences, etc.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        #  _data schema:  { "user@example.com": { "timezone": "Asia/Kolkata" } }

    # ── Timezone ──────────────────────────────────────────────

    def set_timezone(self, email: str, timezone: str) -> None:
        """Store or update the timezone for a given user email."""
        with self._lock:
            if email not in self._data:
                self._data[email] = {}
            self._data[email]["timezone"] = timezone

    def get_timezone(self, email: str, default: str = "UTC") -> str:
        """Retrieve the stored timezone for a user, or `default`."""
        with self._lock:
            return self._data.get(email, {}).get("timezone", default)

    # ── Generic getters/setters (for future expansion) ────────

    def set(self, email: str, key: str, value) -> None:
        with self._lock:
            if email not in self._data:
                self._data[email] = {}
            self._data[email][key] = value

    def get(self, email: str, key: str, default=None):
        with self._lock:
            return self._data.get(email, {}).get(key, default)

    def get_all(self, email: str) -> dict:
        """Return a copy of all stored context for a user."""
        with self._lock:
            return dict(self._data.get(email, {}))


# ── Singleton instance ────────────────────────────────────────
user_ctx = UserContextStore()


# ── Flask route factory ───────────────────────────────────────

def register_user_context_routes(app):
    """
    Register the /update_timezone POST route on the given Flask app.
    Call this once after creating the Flask app instance.
    """
    from flask import request, jsonify, session

    @app.route("/update_timezone", methods=["POST"])
    def update_timezone():
        if "user" not in session:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(silent=True) or {}
        timezone = data.get("timezone", "").strip()

        if not timezone:
            return jsonify({"error": "No timezone provided"}), 400

        email = session["user"].get("email")
        if email:
            user_ctx.set_timezone(email, timezone)

        return jsonify({"status": "ok", "timezone": timezone})
