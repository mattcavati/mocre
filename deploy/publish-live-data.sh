#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"
REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST to the SSH host or alias}"
REMOTE_BASE="${REMOTE_BASE:-/srv/mocre}"
SOURCE="$ROOT/globe/globe_data.json"
TOKEN="$(date -u +%Y%m%dT%H%M%S)-$$"
REMOTE_TMP="/tmp/mocre-globe-$TOKEN.json"

# No se publica un JSON parcial o una salida sin los campos de procedencia.
python3 -m json.tool "$SOURCE" >/dev/null
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("generated") and d.get("input_cutoff") and d.get("fields") and d.get("cells")' "$SOURCE"

scp -F "$SSH_CONFIG" "$SOURCE" "$REMOTE_HOST:$REMOTE_TMP"
ssh -F "$SSH_CONFIG" "$REMOTE_HOST" /bin/bash -s -- "$REMOTE_TMP" "$REMOTE_BASE" <<'REMOTE'
set -euo pipefail
src="$1"
base="$2"
dst="$base/current/public/globe/globe_data.json"
python3 -m json.tool "$src" >/dev/null
sudo install -o root -g www-data -m 0644 "$src" "$dst.new"
sudo mv -f "$dst.new" "$dst"
rm -f "$src"
REMOTE

echo "publicado: $REMOTE_HOST:$REMOTE_BASE/current/public/globe/globe_data.json"
