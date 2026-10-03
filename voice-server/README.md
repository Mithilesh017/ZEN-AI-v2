# ZEN voice server

ZEN's own text-to-speech: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) on CPU. It removes the daily voice quota of hosted TTS. Built for an Oracle Cloud **Always Free** Ampere A1 VM (4 OCPU, 24 GB), and runs on any Ubuntu 24.04 machine.

```
ZEN (Render) ──HTTPS + bearer token──▶ Caddy :443 ──▶ uvicorn 127.0.0.1:8880 ──▶ Kokoro (ONNX Runtime, CPU)
```

## API

It is the subset of the OpenAI speech API that ZEN needs.

| | |
|---|---|
| `POST /v1/audio/speech` | `{"input": "...", "voice": "af_heart", "speed": 1.0}` with `Authorization: Bearer <VOICE_TOKEN>`. Returns `audio/wav` (24 kHz mono). |
| `GET /health` | `{"ok": true}` |

Responses: `401` wrong token, `400` empty or over 400 characters, `503` busy.

Only `MAX_CONCURRENT` syntheses run at once (default 2). A request that can't start within 0.4 s gets `503` immediately, and ZEN speaks that sentence with its fallback voice (Groq Orpheus) rather than making the caller wait.

## Setup

On a fresh Ubuntu 24.04 VM, with ports 80 and 443 open in the cloud firewall:

```bash
git clone <this repo> && cd <repo>/voice-server
sudo bash setup.sh            # or: sudo bash setup.sh voice.example.com
```

The script:
- installs packages
- creates a locked-down `zenvoice` system user
- downloads the model (about 350 MB)
- generates a random token in `/etc/zen-voice.env`
- runs the server under systemd
- opens the host firewall (Oracle images block everything except SSH)
- puts Caddy in front for automatic HTTPS

Without a domain it uses `<ip-with-dashes>.sslip.io`, which points at the server with no DNS setup. Re-running the script is safe.

Then set these in ZEN's environment (Render → Environment):

```
VOICE_SERVER_URL=https://<the domain the script printed>
VOICE_SERVER_TOKEN=<VOICE_TOKEN from /etc/zen-voice.env>
VOICE_SERVER_VOICE=af_heart          # optional; any Kokoro voice
```

## Performance

Real-time factor is the time to synthesise divided by the audio length; lower is faster.

| Machine | Real-time factor (one sentence) |
|---|---|
| 12-thread desktop x86 | 0.40 |
| Two sentences at once on that machine | 0.57 each |
| Oracle A1, 4 OCPU | measure after setup: `journalctl -u zen-voice` logs it per sentence |

Anything below 1.0 keeps ahead of playback, because ZEN streams sentence by sentence. The int8 model file is slower than full precision on CPUs (about 1.5), so use `kokoro-v1.0.onnx`.

## Operations

```bash
systemctl status zen-voice caddy
journalctl -u zen-voice -f
sudo systemctl restart zen-voice      # after changing /etc/zen-voice.env
```

Oracle reclaims Always Free VMs that stay idle (very low CPU) for about a week. Upgrading the account to Pay-As-You-Go prevents that, and usage within the Always Free limits still costs nothing.
