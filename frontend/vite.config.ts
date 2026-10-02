import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { viteStaticCopy } from 'vite-plugin-static-copy'

// Flask backend (python app.py) — the dev server proxies API + auth routes to it.
const BACKEND = 'http://localhost:10000'
const proxied = ['/api', '/login', '/google-login', '/callback', '/logout', '/static/zen']

// Runtime files for on-device voice and object detection, served from our own
// origin (no third-party CDN at call time). Flattened into /call-assets/.
const flat = { stripBase: true } as const
const callAssets = [
  { src: 'node_modules/@ricky0123/vad-web/dist/vad.worklet.bundle.min.js', dest: 'call-assets', rename: flat },
  { src: 'node_modules/@ricky0123/vad-web/dist/silero_vad_v5.onnx', dest: 'call-assets', rename: flat },
  { src: 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.{mjs,wasm}', dest: 'call-assets', rename: flat },
  { src: 'node_modules/@mediapipe/tasks-vision/wasm/*', dest: 'call-assets/mediapipe', rename: flat },
]

export default defineConfig({
  plugins: [react(), tailwindcss(), viteStaticCopy({ targets: callAssets })],
  resolve: {
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  // Production build is served by Flask from /static/app/
  base: '/static/app/',
  build: {
    outDir: '../static/app',
    emptyOutDir: true,
  },
  server: {
    proxy: Object.fromEntries(
      proxied.map((p) => [p, p === '/api' ? { target: BACKEND, ws: true } : BACKEND]),
    ),
  },
})
