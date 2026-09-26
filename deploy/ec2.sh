#!/usr/bin/env bash
# Redeploy the app to the EC2 instance (Amazon Linux 2023 with Docker installed).
#
#   deploy/ec2.sh                      # stage, copy, build, restart
#   deploy/ec2.sh --key-only           # just (re)install the Gemini key from .env
#
# The Gemini key is copied into /etc/floodmap.env on the instance (root-only, 600) and passed to the container
# at run time; it is never part of the image. Request log + answer cache live in the floodmap-cache volume.
set -euo pipefail
HOST="${EC2_HOST:-ec2-user@18.119.105.145}"
KEY="${EC2_KEY:-$HOME/Downloads/hackathon.pem}"
cd "$(dirname "$0")/.."
SSH=(ssh -i "$KEY" -o ConnectTimeout=10 -o ServerAliveInterval=30 "$HOST")

install_key() {
  grep '^GEMINI_API_KEY=' .env | "${SSH[@]}" 'sudo install -m 600 -o root -g root /dev/stdin /etc/floodmap.env'
}

if [[ "${1:-}" == "--key-only" ]]; then install_key; exit 0; fi

.venv/bin/python -m deploy.pack
.venv/bin/python -m deploy.stage
COPYFILE_DISABLE=1 tar --no-xattrs -C .deploy -czf - . | "${SSH[@]}" 'rm -rf ~/floodmap && mkdir -p ~/floodmap && tar -C ~/floodmap -xzf -'
"${SSH[@]}" 'sudo test -f /etc/floodmap.env' || install_key
"${SSH[@]}" 'cd ~/floodmap && sudo docker build -q -t floodmap . && sudo docker rm -f floodmap >/dev/null 2>&1;
  sudo docker run -d --name floodmap --restart unless-stopped -p 80:7860 --env-file /etc/floodmap.env \
    -v floodmap-cache:/app/data/cache --memory 1200m floodmap >/dev/null && sleep 10 && curl -s localhost/api/health'
echo
echo "live: http://${HOST#*@}"
