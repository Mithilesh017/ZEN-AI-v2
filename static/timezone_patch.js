// NEW FILE — do not modify existing files
// ============================================================
// timezone_patch.js — Auto Timezone Detection (Feature 1)
// ============================================================
//
// HOW TO INJECT THIS FILE:
//   Add the following <script> tag in templates/index.html, BEFORE the
//   closing </body> tag and BEFORE the existing inline <script> block:
//
//       <script src="/static/timezone_patch.js"></script>
//
//   Flask automatically serves files from the /static/ folder, so no
//   route changes are needed.
//
// WHAT IT DOES:
//   1. Detects the user's IANA timezone (e.g. "Asia/Kolkata") using the
//      browser's Intl API — no user input required.
//   2. Monkey-patches the global fetch() so that every POST to /chat
//      automatically includes a "timezone" field in the JSON body.
//   3. On page load, sends the detected timezone to /update_timezone
//      so it can be persisted in the user_context store (Feature 4).
//
// ZERO MODIFICATIONS to index.html's sendMessage() are needed.
// ============================================================

(function () {
    "use strict";

    // ---------- Detect timezone ----------
    const USER_TIMEZONE = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";

    // ---------- Monkey-patch fetch to inject timezone into /chat ----------
    const _originalFetch = window.fetch;

    window.fetch = function (url, options) {
        // Only intercept POST requests to /chat
        if (
            typeof url === "string" &&
            url.endsWith("/chat") &&
            options &&
            options.method &&
            options.method.toUpperCase() === "POST"
        ) {
            try {
                const body = JSON.parse(options.body);
                body.timezone = USER_TIMEZONE;
                options = Object.assign({}, options, {
                    body: JSON.stringify(body),
                });
            } catch (_) {
                // If the body isn't JSON, leave it alone
            }
        }
        return _originalFetch.call(this, url, options);
    };

    // ---------- Persist timezone server-side on page load ----------
    // This calls the /update_timezone endpoint (see user_context.py)
    // so the backend can map email → timezone for tool calls.
    window.addEventListener("load", function () {
        _originalFetch("/update_timezone", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ timezone: USER_TIMEZONE }),
        }).catch(function () {
            // Silently ignore — timezone will still be sent per-message
        });
    });

    // Expose for debugging if needed
    window.__ZEN_TIMEZONE = USER_TIMEZONE;
})();
