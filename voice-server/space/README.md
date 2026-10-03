---
title: ZEN Voice
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Text-to-speech server for ZEN (Kokoro-82M on CPU)
---

Text-to-speech for ZEN: Kokoro-82M (Apache-2.0) behind a small API.

`POST /v1/audio/speech` needs `Authorization: Bearer <VOICE_TOKEN>`; the token
is a Space secret. `GET /health` is open.
