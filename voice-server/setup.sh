#!/usr/bin/env bash
# Set up ZEN's voice server on a fresh Ubuntu 24.04 machine (built for an
# Oracle Cloud Always Free Ampere A1 VM, works on any arm64/x86_64 Ubuntu).
#
#   sudo bash setup.sh [domain]
#
# The domain defaults to <public-ip>.sslip.io, which resolves to this machine
# with no DNS setup, so Caddy can get a real HTTPS certificate for it.
# Safe to re-run: every step checks what is already in place.
set -euo pipefail

APP_DIR=/opt/zen-voice
ENV_FILE=/etc/zen-voice.env
SERVICE_USER=zenvoice
MODEL_BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ $EUID -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }

PUBLIC_IP="$(curl -fsS --max-time 5 https://api.ipify.org || true)"
DOMAIN="${1:-${PUBLIC_IP//./-}.sslip.io}"
[[ -n "$PUBLIC_IP" || -n "${1:-}" ]] || { echo "Could not detect the public IP; pass a domain." >&2; exit 1; }
echo "==> Setting up the voice server at https://$DOMAIN"

echo "==> Packages"
export DEBIAN_FRONTEND=noninteractive
echo iptables-persistent iptables-persistent/autosave_v4 boolean true | debconf-set-selections
echo iptables-persistent iptables-persistent/autosave_v6 boolean true | debconf-set-selections
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip espeak-ng caddy iptables-persistent curl >/dev/null

echo "==> App user and files"
id -u "$SERVICE_USER" >/dev/null 2>&1 || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
mkdir -p "$APP_DIR/models"
install -m 0644 "$SRC_DIR/server.py" "$SRC_DIR/requirements.txt" "$APP_DIR/"

echo "==> Python environment"
[[ -d "$APP_DIR/venv" ]] || python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install -q --upgrade pip
"$APP_DIR/venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "==> Kokoro model files"
for file in kokoro-v1.0.onnx voices-v1.0.bin; do
  [[ -s "$APP_DIR/models/$file" ]] || curl -fsSL -o "$APP_DIR/models/$file" "$MODEL_BASE/$file"
done
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"

echo "==> Secret token"
if [[ ! -f "$ENV_FILE" ]]; then
  umask 077
  cat > "$ENV_FILE" <<EOF
VOICE_TOKEN=$(openssl rand -hex 32)
KOKORO_MODEL=$APP_DIR/models/kokoro-v1.0.onnx
KOKORO_VOICES=$APP_DIR/models/voices-v1.0.bin
DEFAULT_VOICE=af_heart
MAX_CONCURRENT=2
EOF
fi
chown root:"$SERVICE_USER" "$ENV_FILE"
chmod 640 "$ENV_FILE"

echo "==> Service"
cat > /etc/systemd/system/zen-voice.service <<EOF
[Unit]
Description=ZEN voice server (Kokoro TTS)
After=network-online.target
Wants=network-online.target

[Service]
User=$SERVICE_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$APP_DIR/venv/bin/uvicorn server:app --host 127.0.0.1 --port 8880 --workers 1 --no-access-log
Restart=always
RestartSec=3
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadOnlyPaths=$APP_DIR

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now zen-voice
systemctl restart zen-voice

echo "==> Firewall"
# Oracle's Ubuntu images reject everything except SSH in iptables, on top of
# the cloud security list. Open 80 (certificate challenge) and 443.
for port in 80 443; do
  iptables -C INPUT -p tcp --dport "$port" -m conntrack --ctstate NEW -j ACCEPT 2>/dev/null \
    || iptables -I INPUT -p tcp --dport "$port" -m conntrack --ctstate NEW -j ACCEPT
done
netfilter-persistent save >/dev/null

echo "==> HTTPS (Caddy)"
cat > /etc/caddy/Caddyfile <<EOF
$DOMAIN {
    encode gzip
    request_body {
        max_size 16KB
    }
    reverse_proxy 127.0.0.1:8880
}
EOF
systemctl enable --now caddy
systemctl reload caddy || systemctl restart caddy

echo "==> Waiting for the voice server"
for _ in $(seq 1 60); do
  curl -fsS http://127.0.0.1:8880/health >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS http://127.0.0.1:8880/health && echo

echo
echo "Done. Put these in ZEN's environment (Render -> Environment):"
echo "  VOICE_SERVER_URL=https://$DOMAIN"
echo "  VOICE_SERVER_TOKEN=(the VOICE_TOKEN value in $ENV_FILE)"
