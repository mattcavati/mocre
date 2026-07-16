#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$root"
failed=0

fail() {
  printf 'FAIL\t%s\n' "$1" >&2
  failed=1
}

while IFS= read -r -d '' path; do
  case "$path" in
    */.env.example|*/.env.mail.example) ;;
    *) fail "forbidden credential/runtime filename: ${path#./}" ;;
  esac
done < <(
  find . -path './.git' -prune -o -type f \
    \( -name '.env' -o -name '.env.*' -o -iname '*.pem' -o -iname '*.key' \
       -o -iname '*.p12' -o -iname '*.pfx' -o -iname '*.cookies' \
       -o -iname '*.session' -o -iname '*.sqlite' -o -iname '*.sqlite3' \
       -o -iname '*.db' -o -iname '*.dump' -o -iname '*.sql.gz' \
       -o -iname '*.bak' -o -iname '*.bak.*' -o -iname '*.log' \
       -o -iname '*.log.*' \) -print0
)

for directory in .secrets .sensitive_backup storage uploads backups node_modules vendor; do
  if find . -path './.git' -prune -o -type f -path "*/${directory}/*" -print -quit | rg -q .; then
    fail "forbidden directory contains files: ${directory}"
  fi
done

while IFS= read -r -d '' path; do
  fail "file exceeds 20 MiB: ${path#./}"
done < <(find . -path './.git' -prune -o -type f -size +20M -print0)

if rg -l -I --hidden --glob '!.git/**' --pcre2 \
  '(?:-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----|(?:AKIA|ASIA)[A-Z0-9]{16}|AIza[0-9A-Za-z_-]{30,}|github_pat_[A-Za-z0-9_]{20,}|gh[opsu]_[A-Za-z0-9]{20,}|(?:sk|rk)_live_[A-Za-z0-9]{16,}|xox[baprs]-[A-Za-z0-9-]{16,}|[0-9]{8,12}:[A-Za-z0-9_-]{30,})' . >/dev/null; then
  fail 'high-confidence secret pattern detected'
fi

if rg -l -I --hidden --glob '!.git/**' --pcre2 \
  "getenv\\(['\"][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASS|PASSWORD)[A-Z0-9_]*['\"]\\)\\s*\\?:\\s*['\"][^'\"]+" . >/dev/null; then
  fail 'secret-like environment variable has a literal fallback'
fi

if [[ ! -f LICENSE && ! -f LICENSE.md && ! -f COPYING ]]; then
  fail 'no license file; choose the publication license before release'
fi

if ! command -v gitleaks >/dev/null 2>&1; then
  fail 'gitleaks is required for the release gate'
else
  if ! gitleaks dir . --no-banner --redact=100; then
    fail 'gitleaks detected a candidate secret'
  fi
fi

if (( failed != 0 )); then
  exit 1
fi

printf 'PASS\tpublic release checks completed\n'
