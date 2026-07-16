#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
DEPLOY_DIR="$ROOT/.deploy"
RELEASE="$DEPLOY_DIR/release-$STAMP"
OUT="$RELEASE/public"
CURRENT="$DEPLOY_DIR/public-current"
LOCAL_PUBLIC="$ROOT/public"
LOCAL_PUBLIC_TMP="$DEPLOY_DIR/public-local-$STAMP"

umask 022
mkdir -p "$OUT/map" "$OUT/globe" "$OUT/.well-known"

install -m 0644 "$ROOT/welcome/index.html" "$OUT/index.html"
install -m 0644 "$ROOT/welcome/access.html" "$OUT/access.html"
install -m 0644 "$ROOT/welcome/control.html" "$OUT/control.html"
install -m 0644 "$ROOT/welcome/icon.png" "$OUT/icon.png"
install -m 0644 "$ROOT/welcome/favicon.ico" "$OUT/favicon.ico"
install -m 0644 "$ROOT/welcome/icon.png" "$OUT/apple-touch-icon.png"
install -m 0644 "$ROOT/welcome/avaltix-mark.png" "$OUT/avaltix-mark.png"
install -m 0644 "$ROOT/welcome/avaltix-name.png" "$OUT/avaltix-name.png"

install -m 0644 "$ROOT/map/index.html" "$OUT/map/index.html"
install -m 0644 "$ROOT/map/world_land.json" "$OUT/map/world_land.json"
install -m 0644 "$ROOT/map/icon.png" "$OUT/map/icon.png"
install -m 0644 "$ROOT/map/favicon.ico" "$OUT/map/favicon.ico"

install -m 0644 "$ROOT/globe/index.html" "$OUT/globe/index.html"
install -m 0644 "$ROOT/globe/globe_data.json" "$OUT/globe/globe_data.json"
install -m 0644 "$ROOT/globe/icon.png" "$OUT/globe/icon.png"
install -m 0644 "$ROOT/globe/favicon.ico" "$OUT/globe/favicon.ico"
if [ -f "$ROOT/out/critical_shadow.json" ]; then
  install -m 0644 "$ROOT/out/critical_shadow.json" "$OUT/critical-shadow.json"
fi

perl -0pi -e 's#http://localhost:7899/?#globe/#g; s#http://localhost:7898/?#map/#g; s#http://localhost:7900/auth/google#/auth/google#g; s#http://localhost:7900#/auth#g' \
  "$OUT/index.html" "$OUT/access.html" "$OUT/control.html"

perl -0pi -e '
  s#<b>7899</b><span>globo vivo#<b>Globo</b><span>globo vivo#g;
  s#<b>7898</b><span>mapa MOCRE-1#<b>Mapa</b><span>mapa MOCRE-1#g;
  s#<b>7897</b><span>entrada [^<]*#<b>Web</b><span>entrada publica - producto - acceso#g;
  s#<b>7900</b><span>gateway propuesto#<b>Acceso</b><span>gateway propuesto#g;
  s#<span>globo</span><b>7899</b>#<span>globo</span><b>publicado</b>#g;
  s#<span>mapa</span><b>7898</b>#<span>mapa</span><b>publicado</b>#g;
  s#<span>gateway acceso</span><b>7900</b>#<span>gateway acceso</span><b>pendiente</b>#g;
  s#Gateway esperado: <code>/auth/google</code>#Gateway preparado: <code>/auth/google</code>#g;
' "$OUT/index.html" "$OUT/access.html" "$OUT/control.html"

cat > "$OUT/robots.txt" <<'ROBOTS'
User-agent: *
Allow: /
Disallow: /auth/

Sitemap: https://mocre.es/sitemap.xml
ROBOTS

cat > "$OUT/sitemap.xml" <<'SITEMAP'
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://mocre.es/</loc></url>
  <url><loc>https://mocre.es/access.html</loc></url>
  <url><loc>https://mocre.es/control.html</loc></url>
  <url><loc>https://mocre.es/map/</loc></url>
  <url><loc>https://mocre.es/globe/</loc></url>
</urlset>
SITEMAP

cat > "$OUT/.well-known/security.txt" <<'SECURITY'
Contact: mailto:contacto@avaltix.io
Preferred-Languages: es, en
SECURITY

if find "$OUT" -type l -print -quit | grep -q .; then
  echo "Refusing build: public artifact contains symlinks" >&2
  exit 1
fi

if find "$OUT" -type f \( \
    -name '*.py' -o -name '*.pyc' -o -name '*.sh' -o -name '*.csv' -o \
    -name '*.env' -o -name '*.md' -o -name '*.sql' -o -name '*.sqlite' -o \
    -name '*.db' -o -name '*.tar' -o -name '*.gz' -o -name '*.zip' \
  \) -print -quit | grep -q .; then
  echo "Refusing build: public artifact contains a forbidden file type" >&2
  exit 1
fi

find "$OUT" -type f -printf '%P\t%s\n' | sort > "$DEPLOY_DIR/manifest-$STAMP.txt"
ln -sfn "release-$STAMP/public" "$CURRENT"

rm -rf "$LOCAL_PUBLIC_TMP"
mkdir -p "$LOCAL_PUBLIC_TMP"
rsync -a --delete "$OUT"/ "$LOCAL_PUBLIC_TMP"/
rm -rf "$LOCAL_PUBLIC"
mv "$LOCAL_PUBLIC_TMP" "$LOCAL_PUBLIC"

printf '%s\n' "$RELEASE"
