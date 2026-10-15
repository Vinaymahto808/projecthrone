"""
security.py - Authentication for the Face Attendance API (stdlib only).

Provides the three things an industry deployment needs:

1. Admin/kiosk **tokens** - compact JWT-style strings signed with HS256
   (hmac/sha256/base64 from the standard library, no extra packages). The
   signing key is generated once and stored in config.SECRET_KEY_PATH; the
   FACE_ATTENDANCE_SECRET environment variable overrides it (use that when
   running more than one server against the same trust domain).

2. **Device (kiosk) keys** - long random keys issued per kiosk via the
   dashboard/API. The phone exchanges the key for a short-lived token at
   POST /auth/device-token, so the key itself is rarely sent.

3. **Login rate limiting** - a per-IP sliding window over failed password
   attempts, used by /auth/login and /employee/login to slow credential
   stuffing on the LAN.

Roles
-----
    admin   - full API (register, reports, corrections, audit, settings)
    device  - kiosk operations only (verify, mark, history)

When config.REQUIRE_AUTH is False every request is treated as admin; that is
the escape hatch for wiring up a fresh install (and for desktop testing).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Any

import config

_lock = threading.Lock()
_SECRET: bytes | None = None


# ----------------------------------------------------------------- secret key
def _load_secret() -> bytes:
    """Signing key: env override > existing file > generate + persist."""
    global _SECRET
    if _SECRET is not None:
        return _SECRET
    with _lock:
        if _SECRET is None:
            env = os.environ.get("FACE_ATTENDANCE_SECRET", "").strip()
            if env:
                _SECRET = env.encode("utf-8")
            else:
                try:
                    if os.path.exists(config.SECRET_KEY_PATH):
                        with open(config.SECRET_KEY_PATH, encoding="utf-8") as fh:
                            _SECRET = fh.read().strip().encode("utf-8")
                    else:
                        _SECRET = secrets.token_hex(32).encode("ascii")
                        with open(config.SECRET_KEY_PATH, "w", encoding="utf-8") as fh:
                            fh.write(_SECRET.decode("ascii") + "\n")
                        try:  # best effort: keep the key readable only by us
                            os.chmod(config.SECRET_KEY_PATH, 0o600)
                        except OSError:
                            pass
                except OSError:  # read-only folder -> ephemeral key
                    _SECRET = secrets.token_hex(32).encode("ascii")
    return _SECRET


# ----------------------------------------------------------------- JWT (HS256)
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def create_token(claims: dict[str, Any], ttl_sec: int | None = None) -> tuple[str, int]:
    """Sign a token. Returns (token, expiry_unix_seconds)."""
    now = int(time.time())
    exp = now + int(ttl_sec if ttl_sec is not None else config.TOKEN_TTL_SEC)
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {**claims, "iat": now, "exp": exp, "jti": secrets.token_hex(6)}
    head = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{head}.{body}".encode("ascii")
    sig = hmac.new(_load_secret(), signing_input, hashlib.sha256).digest()
    return f"{head}.{body}.{_b64url(sig)}", exp


def verify_token(token: str) -> dict[str, Any] | None:
    """Validate signature + expiry. Returns the claims, or None if invalid."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    expected = hmac.new(_load_secret(), signing_input, hashlib.sha256).digest()
    try:
        actual = _b64url_decode(parts[2])
    except Exception:
        return None
    if not hmac.compare_digest(expected, actual):
        return None
    try:
        claims = json.loads(_b64url_decode(parts[1]))
    except Exception:
        return None
    if int(claims.get("exp", 0)) < time.time():
        return None
    return claims


# ---------------------------------------------------------------- passwords
def hash_password(password: str, salt: str | None = None) -> str:
    """PBKDF2-SHA256, same format as store.py (salt$digest)."""
    salt = salt or secrets.token_hex(8)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000
    ).hex()
    return f"{salt}${digest}"


def check_password(stored: str | None, password: str) -> bool:
    if not stored or "$" not in stored:
        return False
    salt, _ = stored.split("$", 1)
    return hmac.compare_digest(hash_password(password, salt), stored)


def hash_api_key(key: str) -> str:
    """Device keys are high-entropy, so a single SHA-256 is enough."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def check_api_key(stored_hash: str | None, key: str) -> bool:
    return bool(stored_hash) and hmac.compare_digest(stored_hash, hash_api_key(key))


# ------------------------------------------------------------- rate limiting
class RateLimiter:
    """Sliding-window limiter for failed attempts, keyed by client id (IP)."""

    def __init__(self, limit: int, window_sec: int) -> None:
        self.limit = limit
        self.window = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def blocked(self, key: str) -> bool:
        """True when the key has already used up its failed attempts."""
        now = time.time()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            return len(hits) >= self.limit

    def record_failure(self, key: str) -> None:
        now = time.time()
        with self._lock:
            self._hits[key].append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


# One limiter shared by every credential endpoint.
login_limiter = RateLimiter(config.LOGIN_RATE_LIMIT, config.LOGIN_WINDOW_SEC)
