# ZEN Vision: live video calls

Talk to ZEN face to face. ZEN sees through the phone's camera (front or back, switchable mid-call), hears you, answers out loud, can be interrupted like a person, and labels objects on screen as they appear.

```
PHONE (React, lazy-loaded)                               SERVER (Flask, same login)
 camera ─▶ EfficientDet-Lite0 (MediaPipe, ~12 fps) ─▶ tracked boxes on screen
        └▶ JPEG ≤768px, taken when you stop talking ─┐
 mic ─▶ Silero VAD (WASM) ─▶ utterance WAV ───────────┼── WS /api/vision/call ──▶ CallSession
        └▶ speech while ZEN talks = barge-in          │                           ├ Whisper (STT)
 speaker ◀─ gapless Web Audio queue ◀─────────────────┘◀── sentence WAVs ─────────├ Qwen VL (streamed)
 captions ◀─ revealed as each sentence starts playing                             ├ sentence chunker → Orpheus TTS
                                                                                  └ long-term memory
```

## Why it is built this way

| Decision | Reason |
|---|---|
| Voice detection on the phone | Turn-taking and interruption need no server round trip, so the call feels live. |
| One frame per utterance, not a video stream | The model only needs what is in view when you ask. That is roughly 50× cheaper than streaming, and there is no video to store or leak. |
| Sentence-level pipelining | Each sentence goes to TTS as soon as the model finishes it (two in flight), and results are sent in order. The first audio leaves after one sentence, not a whole reply. |
| Captions revealed by playback, not by generation | Text never runs ahead of the voice. |
| Barge-in reports what was heard | The assistant's history is cut to the sentences you actually heard, so ZEN never thinks it said something you missed. |
| Graceful degradation | No TTS: captions only. No WebGL2: Canvas2D orb. No detector: call without boxes. Camera blocked: voice-only call. |
| EfficientDet (Apache-2.0), not YOLO (AGPL) | Licensing for a commercial product. |
| Everything self-hosted | VAD, ONNX Runtime and MediaPipe WASM are copied from `node_modules` at build time; the detector model lives in `frontend/public/models/`. No CDN is involved at call time. |

## Files

| File | Responsibility |
|---|---|
| `protocol.py` | Wire format: JSON control frames plus an atomic binary utterance envelope. Validation and size limits. |
| `chunker.py` | Splits streamed text into speakable chunks (early clause break for the first chunk; 180-character TTS cap). |
| `speech.py` | `Engines` interface and its Groq implementation (Whisper, Qwen VL, Orpheus). Whisper's "no speech" segments are dropped. |
| `prompt.py` | Call-mode system prompt (spoken, brief, camera-grounded, never identifies people by face). |
| `session.py` | One call's state machine: turns, cancellation, ordered TTS pipeline, history, memory, timings. |
| `routes.py` | WebSocket endpoint: session-cookie auth, Origin check, rate limits, one live call per account. |

Frontend: `frontend/src/features/call/`. The UI talks only to the `CallTransport` interface (`transport/types.ts`).

## Protocol

```
client → server  text   {"type":"hello","v":1} | {"type":"interrupt","turn","played"} | {"type":"bye"}
client → server  binary u8 0x01 | u32 header_len | header JSON | WAV | JPEG
server → client  text   ready | transcript | say | turn_done | notice | error | ended
server → client  binary u8 0x10 | u32 turn | u16 seq | WAV
```

Close codes: `4401` signed out, `4403` bad origin, `4429` too many calls.

## Latency

Measured locally (Groq, an image in every turn):

| Stage | Typical |
|---|---|
| End of speech detected (VAD redemption) | 650 ms |
| Speech to text | 0.6–1.3 s |
| First token (reasoning disabled) | 0.2–0.9 s after the transcript |
| First audio | first token + one sentence of TTS |

Per-turn timings are logged by the server. Open the call with `?debug` to see them on screen.

## Configuration

| Env var | Default |
|---|---|
| `VISION_MODEL` | `qwen/qwen3.8-27b` |
| `STT_MODEL` | `whisper-large-v3-turbo` |
| `TTS_MODEL` / `TTS_VOICE` | `canopylabs/orpheus-v1-english` / `autumn` |
| `GEMINI_API_KEY` | unset. When set, replies fall over to Gemini while Groq is rate limited or down |
| `GEMINI_MODELS` | `gemini-3.5-flash-lite,gemini-3.1-flash-lite` (tried in order) |
| `VOICE_SERVER_URL` / `VOICE_SERVER_TOKEN` | unset. ZEN's own Kokoro server (see `voice-server/`); used first when set |
| `VOICE_SERVER_VOICE` | `af_heart` |
| `CALL_MAX_MINUTES` | `15` |
| `CALL_STARTS_PER_HOUR` | `10` |
| `CALL_TURNS_PER_MINUTE` | `30` |

**Orpheus needs a one-time terms acceptance** by the Groq org admin at console.groq.com (open the model in the playground). Until then calls run captions-only.

**Voice quota.** Every speech chunk is one TTS request. Groq's free tier allows Orpheus about 100 requests per day and 1,200 tokens per minute, which is roughly 10 minutes of conversation a day. To make that go further:
- After the first quick sentence, sentences are grouped into chunks of 80–180 characters, so a typical reply is two requests.
- A per-minute limit is waited out (up to 4 s) and the request is retried.
- When the daily quota runs out, voice turns off app-wide until Groq's reset time. Calls continue with captions and say why ("voice_limited" notice), new calls start captions-only, and voice comes back by itself after the reset.

For real use, move the Groq org to a paid tier.

## Fallbacks

Each part of a turn has an order of preference. A source that is rate limited or failing is skipped until it recovers, tracked per process.

| Part | Order |
|---|---|
| Reply | Groq vision model → Gemini Flash-Lite models. If all are rate limited, a short wait is ridden out and a long one is announced to the caller. |
| Voice | Own Kokoro server → Groq Orpheus → the phone's built-in voice → captions |

Free-tier limits measured on this account:
- **Groq vision:** 8,000 tokens/min and 200,000 tokens/day. A turn with a frame is about 1,100–2,400 tokens.
- **Groq Orpheus:** 100 requests/day.
- **Gemini free tier:** slower to start (about 2–4 s) and sometimes refuses with "high demand". Google may use free-tier inputs to improve its products, so use the paid tier before sending real users' camera frames to it.

## Deploying

A call holds a WebSocket open, so gunicorn needs threaded workers:

```
gunicorn -k gthread --threads 32 app:app
```

Each live call uses one thread for its socket plus short-lived threads per turn. All model calls are I/O-bound.

Browsers only allow camera and microphone on HTTPS (or `localhost`). To test on a phone over the LAN, serve the Vite dev server over HTTPS or use the deployed site.

## Adding a WebRTC transport (LiveKit)

The client implements `CallTransport` once (`transport/gateway.ts`). A LiveKit transport would publish the mic and camera tracks to a room and receive ZEN's audio track. A LiveKit Agents worker would wrap `CallSession`, feeding it utterances from the room's VAD and the latest video frame. Nothing in the UI, playback or detector code changes.
