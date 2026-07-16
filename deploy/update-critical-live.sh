#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 src/update_segment_catalogs.py
for f in config/segment_*.json; do
  python3 src/pipeline.py "${f##*/}" >/dev/null
done
python3 src/emit_critical_shadow.py
python3 src/forecast.py >/dev/null
python3 src/build_map.py >/dev/null
"$ROOT/deploy/publish-critical-shadow.sh"
