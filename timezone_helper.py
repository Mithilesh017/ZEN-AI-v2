# NEW FILE — do not modify existing files
# ============================================================
# timezone_helper.py — Timezone-Aware Datetime (Feature 1)
# ============================================================
#
# INTEGRATION INSTRUCTIONS (single-line changes in app.py):
#
# 1. Add this import at the top of app.py (after the existing imports):
#
#        from timezone_helper import get_current_datetime as get_current_datetime_tz
#
# 2. In the /chat route, read the timezone from the request JSON:
#
#        user_timezone = request.json.get("timezone", "UTC")
#
#    Place this line right after:  user_message = request.json["message"]
#
# 3. In the tool-call handler block, change:
#
#        tool_result = get_current_datetime()
#
#    to:
#
#        tool_result = get_current_datetime_tz(user_timezone)
#
# That's it — three single-line changes, zero rewrites.
# ============================================================

import json
from datetime import datetime
from zoneinfo import ZoneInfo


def get_current_datetime(timezone: str = "UTC") -> str:
    """
    Return the current date, time, day-of-week, timezone name,
    and UTC offset as a JSON string — localized to the given
    IANA timezone (e.g. "Asia/Kolkata", "America/New_York").

    Falls back to UTC if the timezone string is invalid.
    """
    try:
        tz = ZoneInfo(timezone)
    except (KeyError, Exception):
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)

    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timezone": timezone,
        "timezone_abbr": now.strftime("%Z"),
        "utc_offset": now.strftime("%z"),
    })
