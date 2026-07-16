#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"
REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST to the SSH host or alias}"
REMOTE_BASE="${REMOTE_BASE:-/srv/mocre}"
INSTALL_NGINX="${MOCRE_INSTALL_NGINX:-auto}"
STAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

RELEASE_DIR="$("$ROOT/deploy/build-public.sh" "$STAMP")"
REMOTE_TMP="/tmp/mocre-release-$STAMP"

if [ ! -d "$RELEASE_DIR/public" ]; then
  echo "Refusing deploy: build did not produce a public/ directory" >&2
  exit 1
fi

rsync -az --delete "$RELEASE_DIR"/ -e "ssh -F $SSH_CONFIG" "$REMOTE_HOST:$REMOTE_TMP/"
scp -F "$SSH_CONFIG" "$ROOT/deploy/mocre.es.nginx.conf" "$REMOTE_HOST:/tmp/mocre.es.nginx.conf"

ssh -F "$SSH_CONFIG" "$REMOTE_HOST" /bin/bash -s -- "$STAMP" "$REMOTE_BASE" "$REMOTE_TMP" "$INSTALL_NGINX" <<'REMOTE'
set -euo pipefail

STAMP="$1"
REMOTE_BASE="$2"
REMOTE_TMP="$3"
INSTALL_NGINX="$4"
RELEASE="$REMOTE_BASE/releases/$STAMP"

sudo install -d -o root -g www-data -m 0710 "$REMOTE_BASE" "$REMOTE_BASE/releases"
sudo rm -rf "$RELEASE"
sudo install -d -o root -g www-data -m 0710 "$RELEASE"
sudo cp -a "$REMOTE_TMP"/. "$RELEASE"/

if ! sudo test -d "$RELEASE/public"; then
  echo "Refusing deploy: release has no public/ directory" >&2
  exit 1
fi

if sudo find "$RELEASE" -mindepth 1 -maxdepth 1 ! -name public -print -quit | grep -q .; then
  echo "Refusing deploy: release contains entries outside public/" >&2
  exit 1
fi

sudo chown -R root:root "$RELEASE"
sudo chown root:www-data "$REMOTE_BASE" "$REMOTE_BASE/releases" "$RELEASE"
sudo chmod 0710 "$REMOTE_BASE" "$REMOTE_BASE/releases" "$RELEASE"
sudo chown -R root:www-data "$RELEASE/public"
sudo find "$RELEASE/public" -type d -exec chmod 0755 {} +
sudo find "$RELEASE/public" -type f -exec chmod 0644 {} +

if sudo test -e "$REMOTE_BASE/current" && ! sudo test -L "$REMOTE_BASE/current"; then
  echo "Refusing to replace non-symlink $REMOTE_BASE/current" >&2
  exit 1
fi

sudo ln -sfn "releases/$STAMP" "$REMOTE_BASE/current"
sudo find "$REMOTE_BASE/releases" -mindepth 1 -maxdepth 1 -type d ! -name "$STAMP" -exec chown -R root:root {} +
sudo find "$REMOTE_BASE/releases" -mindepth 1 -maxdepth 1 -type d ! -name "$STAMP" -exec chmod -R go-rwx {} +
if [ "$INSTALL_NGINX" = "1" ] || { [ "$INSTALL_NGINX" = "auto" ] && ! sudo test -e /etc/nginx/sites-available/mocre.es; }; then
  sudo install -o root -g root -m 0644 /tmp/mocre.es.nginx.conf /etc/nginx/sites-available/mocre.es
  sudo ln -sfn /etc/nginx/sites-available/mocre.es /etc/nginx/sites-enabled/mocre.es
else
  echo "Keeping existing /etc/nginx/sites-available/mocre.es (set MOCRE_INSTALL_NGINX=1 to replace it)."
  sudo sed -i "s#root $REMOTE_BASE/current;#root $REMOTE_BASE/current/public;#g" /etc/nginx/sites-available/mocre.es
fi
sudo nginx -t
sudo systemctl reload nginx

rm -rf "$REMOTE_TMP" /tmp/mocre.es.nginx.conf
echo "$RELEASE"
REMOTE
