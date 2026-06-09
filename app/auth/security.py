"""Password hashing (bcrypt) + JWT session tokens + session-cookie helpers.

bcrypt is used directly (passlib is unmaintained and breaks with bcrypt 5.x).
Passwords are sha256+base64 pre-hashed so any-length input fits bcrypt's
72-byte limit without silent truncation.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Response

from app.config import get_settings

COOKIE_NAME = "peops_session"
_ALGO = "HS256"


# ── passwords ────────────────────────────────────────────────────────────────

def _prehash(password: str) -> bytes:
    # Normalize to a fixed 44-byte token so bcrypt's 72-byte cap never truncates.
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str | None) -> bool:
    # OAuth-only accounts have no password hash — never authenticate by password.
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(_prehash(password), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ── JWT ──────────────────────────────────────────────────────────────────────

def create_access_token(user_id: str) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_ttl_min)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGO)


def decode_access_token(token: str) -> str | None:
    """Return the subject (user id) or None if the token is missing/invalid/expired."""
    if not token:
        return None
    try:
        claims = jwt.decode(token, get_settings().jwt_secret, algorithms=[_ALGO])
    except jwt.PyJWTError:
        return None
    sub = claims.get("sub")
    return sub if isinstance(sub, str) else None


# ── cookie ───────────────────────────────────────────────────────────────────

def set_session_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,  # type: ignore[arg-type]
        path="/",
        max_age=settings.jwt_ttl_min * 60,
        domain=settings.cookie_domain,
    )


def clear_session_cookie(response: Response) -> None:
    settings = get_settings()
    # delete_cookie must echo the same attributes used to set it, or browsers
    # keep the original cookie.
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        samesite=settings.cookie_samesite,  # type: ignore[arg-type]
        secure=settings.cookie_secure,
        httponly=True,
        domain=settings.cookie_domain,
    )
