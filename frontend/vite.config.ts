import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Flask backend (python app.py) — the dev server proxies API + auth routes to it.
const BACKEND = 'http://localhost:10000'
const proxied = ['/api', '/login', '/google-login', '/callback', '/logout', '/static/zen']

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  // Production build is served by Flask from /static/app/
  base: '/static/app/',
  build: {
    outDir: '../static/app',
    emptyOutDir: true,
  },
  server: {
    proxy: Object.fromEntries(proxied.map((p) => [p, BACKEND])),
  },
})
