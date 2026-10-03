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

## Hugging Face Space

The same server runs as a free Docker Space (2 vCPU, 16 GB), with no card and no VM:

```bash
pip install huggingface_hub
HF_SPACE_TOKEN=hf_...  python deploy_space.py          # needs a token with write access
```

It uploads `Dockerfile`, `server.py`, `requirements.txt` and `space/README.md`, sets a random `VOICE_TOKEN` secret, and prints the URL for `VOICE_SERVER_URL`. Free Spaces sleep after about 48 hours without requests and take a minute or so to wake.

## Performance

Measure any host the same way with `python bench.py <url> <token>`. Real-time factor (RTF) is the time to synthesise divided by the audio length; below 1.0 keeps ahead of playback.

| Host | RTF, one sentence | Opening sentence ready | RTF, two in flight |
|---|---|---|---|
| Desktop PC, 6 cores / 12 threads | 0.44 | 0.78 s | 0.62 |
| Same PC limited to 2 vCPUs (one core, two threads) | 0.75 | 1.20 s | 0.96 |
| Hugging Face free Space, Oracle A1 | not measured yet; expect slower than the 2-vCPU row, since cloud vCPUs are slower than desktop cores | | |

Notes:
- Leave `THREADS` unset on a normal machine: ONNX Runtime's default (one thread per physical core) beat using every logical core (0.44 against 0.70). Inside a container the CPU limit is detected automatically.
- The int8 model file is slower than full precision on CPUs (about 1.5), so use `kokoro-v1.0.onnx`.

## Operations

```bash
systemctl status zen-voice caddy
journalctl -u zen-voice -f
sudo systemctl restart zen-voice      # after changing /etc/zen-voice.env
```

Oracle reclaims Always Free VMs that stay idle (very low CPU) for about a week. Upgrading the account to Pay-As-You-Go prevents that, and usage within the Always Free limits still costs nothing.
