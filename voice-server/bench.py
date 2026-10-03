"""
Measure a ZEN voice server the way a call uses it.

    python bench.py https://voice.example.com <token>

Reports, per sentence: wall time from request to complete WAV (what the
caller waits), the server's own synthesis time, and the real-time factor
(time to make the audio / length of the audio; below 1.0 keeps ahead of
playback). Then repeats with two sentences in flight, which is what one call
does mid-reply.
"""

from __future__ import annotations

import io
import statistics
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import requests

OPENER = "That's a red triangle."
SENTENCES = [
    OPENER,
    "Looks like a ceramic mug, probably holding coffee.",
    "It has three straight sides and three corners, and the top point is centred over the base.",
    "You could try turning it slightly so I can read the label on the side, then I can tell you more.",
]


def synth(session: requests.Session, url: str, token: str, text: str) -> dict:
    began = time.perf_counter()
    response = session.post(
        f"{url}/v1/audio/speech",
        json={"input": text, "voice": "af_heart"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    wall = time.perf_counter() - began
    if response.status_code != 200:
        return {"error": response.status_code, "wall": wall}
    with wave.open(io.BytesIO(response.content)) as audio:
        seconds = audio.getnframes() / audio.getframerate()
    server = int(response.headers.get("x-synthesis-ms", 0)) / 1000
    return {"wall": wall, "server": server, "audio": seconds, "rtf": server / seconds, "chars": len(text)}


def main() -> None:
    url, token = sys.argv[1].rstrip("/"), sys.argv[2]
    session = requests.Session()

    began = time.perf_counter()
    session.get(f"{url}/health", timeout=120).raise_for_status()
    print(f"health check: {time.perf_counter() - began:.2f} s (includes any wake-up)")
    synth(session, url, token, "Warm up.")

    print("\none sentence at a time")
    print(f"{'chars':>5} {'audio':>6} {'wait':>6} {'server':>7} {'RTF':>5}")
    rtfs, openers = [], []
    for round_no in range(3):
        for text in SENTENCES:
            r = synth(session, url, token, text)
            if "error" in r:
                print(f"  error {r['error']}")
                continue
            rtfs.append(r["rtf"])
            if text == OPENER:
                openers.append(r["wall"])
            if round_no == 0:
                print(f"{r['chars']:>5} {r['audio']:>5.2f}s {r['wall']:>5.2f}s {r['server']:>6.2f}s {r['rtf']:>5.2f}")
    print(f"median RTF {statistics.median(rtfs):.2f}; "
          f"opening sentence ready in {statistics.median(openers):.2f} s (median of {len(openers)})")

    print("\ntwo sentences in flight (mid-reply)")
    with ThreadPoolExecutor(2) as pool:
        pairs = []
        for _ in range(3):
            began = time.perf_counter()
            results = list(pool.map(lambda t: synth(requests.Session(), url, token, t), SENTENCES[1:3]))
            pairs.append((time.perf_counter() - began, results))
    ok = [r for _, results in pairs for r in results if "error" not in r]
    busy = sum(1 for _, results in pairs for r in results if "error" in r)
    if ok:
        print(f"median RTF {statistics.median(r['rtf'] for r in ok):.2f}; "
              f"both ready in {statistics.median(t for t, _ in pairs):.2f} s; "
              f"{busy} of {len(ok) + busy} refused as busy")


if __name__ == "__main__":
    main()
