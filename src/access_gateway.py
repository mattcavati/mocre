"""MOCRE access gateway.

Serves the welcome UI, performs Google OAuth login, signs a session cookie, and applies small
in-memory rate limits. It is intentionally dependency-free so it can sit in front of the existing
static services.

Required environment for real OAuth:
  GOOGLE_CLIENT_ID
  GOOGLE_CLIENT_SECRET
  MOCRE_SESSION_SECRET
  MOCRE_BASE_URL=http://localhost:7900

Run locally:
  python3 src/access_gateway.py
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WELCOME = ROOT / "welcome"
OUT = ROOT / "out"
POLICY = json.loads((ROOT / "config" / "access_policy.json").read_text())

GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_TOKENINFO = "https://oauth2.googleapis.com/tokeninfo"

BUCKETS: dict[tuple[str, str], list[float]] = {}
BUCKETS_LOCK = threading.Lock()


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def cookie_flags() -> str:
    # Session/state cookies are Secure by default; set MOCRE_COOKIE_INSECURE=1 only for local http dev.
    return "" if env("MOCRE_COOKIE_INSECURE") == "1" else "; Secure"


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def unb64url(txt: str) -> bytes:
    return base64.urlsafe_b64decode(txt + "=" * (-len(txt) % 4))


def sign(payload: bytes) -> str:
    secret = env("MOCRE_SESSION_SECRET")
    if not secret:
        raise RuntimeError("MOCRE_SESSION_SECRET is not configured")
    return b64url(hmac.new(secret.encode(), payload, hashlib.sha256).digest())


def pack_session(user: dict) -> str:
    payload = json.dumps({"user": user, "iat": int(time.time())}, separators=(",", ":")).encode()
    body = b64url(payload)
    return f"{body}.{sign(payload)}"


def unpack_session(token: str | None) -> dict | None:
    if not token or "." not in token or not env("MOCRE_SESSION_SECRET"):
        return None
    body, sig = token.split(".", 1)
    payload = unb64url(body)
    if not hmac.compare_digest(sign(payload), sig):
        return None
    data = json.loads(payload)
    if int(time.time()) - int(data.get("iat", 0)) > 86400:
        return None
    return data.get("user")


def take_rate(key: str, bucket: str, limit: int, window_seconds: int) -> bool:
    now = time.time()
    k = (key, bucket)
    with BUCKETS_LOCK:
        hits = [t for t in BUCKETS.get(k, []) if now - t < window_seconds]
        if len(hits) >= limit:
            BUCKETS[k] = hits
            return False
        hits.append(now)
        BUCKETS[k] = hits
        return True


class Handler(BaseHTTPRequestHandler):
    server_version = "MOCREAccess/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def user(self) -> dict | None:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get(POLICY["session_cookie"])
        return unpack_session(morsel.value if morsel else None)

    def identity_key(self) -> str:
        u = self.user()
        return (u or {}).get("email") or self.client_address[0]

    def send_json(self, data: dict, status: int = 200) -> None:
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def text(self, body: str, status: int = 200) -> None:
        raw = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def require_rate(self, bucket: str, limit: int, window_seconds: int) -> bool:
        if take_rate(self.identity_key(), bucket, limit, window_seconds):
            return True
        self.send_json({"error": "rate_limited", "bucket": bucket}, HTTPStatus.TOO_MANY_REQUESTS)
        return False

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/auth/google":
            return self.auth_google()
        if path == "/auth/google/callback":
            return self.auth_callback(parsed)
        if path == "/api/session":
            return self.send_json({"user": self.user(), "authenticated": self.user() is not None})
        if not self.require_rate("ui", int(POLICY["ui_views_per_minute"]), 60):
            return
        return self.serve_static(path)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/contact":
            return self.send_json({"error": "not_found"}, 404)
        if not self.require_rate("contact", int(POLICY["contact_messages_per_day"]), 86400):
            return
        length = min(int(self.headers.get("Content-Length", "0")), 8192)
        body = self.rfile.read(length).decode("utf-8", "replace")
        OUT.mkdir(exist_ok=True)
        with (OUT / "contact_requests.jsonl").open("a") as f:
            f.write(json.dumps({"ts": int(time.time()), "user": self.user(), "body": body}) + "\n")
        return self.send_json({"ok": True})

    def auth_google(self) -> None:
        if not env("GOOGLE_CLIENT_ID") or not env("GOOGLE_CLIENT_SECRET") or not env("MOCRE_SESSION_SECRET"):
            return self.text("<h1>Google OAuth no configurado</h1><p>Define GOOGLE_CLIENT_ID, "
                             "GOOGLE_CLIENT_SECRET y MOCRE_SESSION_SECRET.</p>", 503)
        base_url = env("MOCRE_BASE_URL", "http://localhost:7900").rstrip("/")
        state = secrets.token_urlsafe(24)
        params = {
            "client_id": env("GOOGLE_CLIENT_ID"),
            "redirect_uri": f"{base_url}/auth/google/callback",
            "response_type": "code",
            "scope": " ".join(POLICY["required_google_scopes"]),
            "access_type": "online",
            "prompt": "select_account",
            "state": state,
        }
        # CSRF defense: bind the browser to this flow with a short-lived state cookie checked on callback.
        self.send_response(302)
        self.send_header("Location", GOOGLE_AUTH + "?" + urllib.parse.urlencode(params))
        self.send_header("Set-Cookie",
                         f"mocre_oauth_state={state}; HttpOnly; SameSite=Lax; Path=/auth; Max-Age=600{cookie_flags()}")
        self.end_headers()

    def auth_callback(self, parsed) -> None:
        qs = urllib.parse.parse_qs(parsed.query)
        code = (qs.get("code") or [""])[0]
        state = (qs.get("state") or [""])[0]
        state_cookie = SimpleCookie(self.headers.get("Cookie", "")).get("mocre_oauth_state")
        expected = state_cookie.value if state_cookie else ""
        if not code:
            return self.text("<h1>OAuth cancelado</h1>", 400)
        if not expected or not state or not hmac.compare_digest(state, expected):
            return self.text("<h1>Estado OAuth inválido</h1>", 400)
        base_url = env("MOCRE_BASE_URL", "http://localhost:7900").rstrip("/")
        payload = urllib.parse.urlencode({
            "code": code,
            "client_id": env("GOOGLE_CLIENT_ID"),
            "client_secret": env("GOOGLE_CLIENT_SECRET"),
            "redirect_uri": f"{base_url}/auth/google/callback",
            "grant_type": "authorization_code",
        }).encode()
        try:
            token = json.loads(urllib.request.urlopen(GOOGLE_TOKEN, payload, timeout=10).read())
            info = json.loads(urllib.request.urlopen(
                GOOGLE_TOKENINFO + "?" + urllib.parse.urlencode({"id_token": token["id_token"]}),
                timeout=10,
            ).read())
        except Exception as exc:
            self.log_message("oauth error: %s", exc)
            return self.text("<h1>OAuth falló</h1><p>No se pudo completar el inicio de sesión.</p>", 502)
        # Only trust an id_token minted for THIS app and a Google-verified email.
        if info.get("aud") != env("GOOGLE_CLIENT_ID") or str(info.get("email_verified")).lower() != "true":
            return self.text("<h1>Token no válido</h1>", 401)
        user = {"email": info.get("email"), "name": info.get("name"), "picture": info.get("picture")}
        session = pack_session(user)
        self.send_response(302)
        self.send_header("Location", "/")
        self.send_header("Set-Cookie",
                         f"{POLICY['session_cookie']}={session}; HttpOnly; SameSite=Lax; Path=/; Max-Age=86400{cookie_flags()}")
        self.send_header("Set-Cookie",
                         f"mocre_oauth_state=; HttpOnly; SameSite=Lax; Path=/auth; Max-Age=0{cookie_flags()}")
        self.end_headers()

    def serve_static(self, path: str) -> None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (WELCOME / rel).resolve()
        if not target.is_relative_to(WELCOME.resolve()) or not target.exists() or target.is_dir():
            return self.text("<h1>No encontrado</h1>", 404)
        raw = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    port = int(env("MOCRE_ACCESS_PORT", "7900"))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"MOCRE access gateway on http://127.0.0.1:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
