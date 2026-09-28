# ZEN AI — web UI

React + Vite + [assistant-ui](https://www.assistant-ui.com) chat interface, served by the Flask app.

```bash
npm install
npm run build   # outputs to ../static/app, which Flask serves at /
npm run dev     # hot-reload UI on :5173, proxying /api and auth routes to Flask on :10000
```

- `src/App.tsx` — app shell, runtime, and suggestions
- `src/lib/chat-adapter.ts` — streams replies from `POST /api/chat` (newline-delimited JSON)
- `src/components/assistant-ui/elements/` — assistant-ui elements (copied in, edit freely)

Conversations are stored in the browser's localStorage, per Google account.
