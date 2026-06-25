"""Google OAuth 2.0 Authorization Code flow helpers.

Three small, separately-mockable functions: build the consent URL, exchange the
code for tokens, and verify the returned id_token. The router stitches them into
the login/callback endpoints and reuses the app's own session cookie.
"""

from __future__ import annotations

from urllib.parse import urlencode

import httpx

from app.config import get_settings

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
SCOPES = "openid email profile"


class GoogleAuthError(Exception):
    """Raised when the OAuth exchange or token verification fails."""


def build_auth_url(state: str) -> str:
    settings = get_settings()
    params = {
        "response_type": "code",
        "client_id": settings.google_client_id or "",
        "redirect_uri": settings.effective_google_redirect_uri,
        "scope": SCOPES,
        "state": state,
        "access_type": "online",
        "include_granted_scopes": "true",
        "prompt": "select_account",
    }
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"


def exchange_code(code: str) -> dict:
    """Trade the authorization code for tokens (contains id_token)."""
    settings = get_settings()
    data = {
        "code": code,
        "client_id": settings.google_client_id or "",
        "client_secret": settings.google_client_secret or "",
        "redirect_uri": settings.effective_google_redirect_uri,
        "grant_type": "authorization_code",
    }
    try:
        resp = httpx.post(TOKEN_ENDPOINT, data=data, timeout=15.0)
    except httpx.HTTPError as exc:  # network failure
        raise GoogleAuthError(f"token endpoint unreachable: {exc}") from exc
    if resp.status_code != 200:
        raise GoogleAuthError(f"token exchange failed: HTTP {resp.status_code} {resp.text[:200]}")
    payload = resp.json()
    if "id_token" not in payload:
        raise GoogleAuthError("token response missing id_token")
    return payload


def verify_id_token(id_token_str: str) -> dict:
    """Verify the id_token's signature + audience against Google's certs and
    return its claims (sub, email, email_verified, name, picture)."""
    settings = get_settings()
    try:
        # Imports live inside the try so a missing transport dep surfaces as a
        # graceful GoogleAuthError (→ /login?error=google), never a 500.
        from google.auth.transport import requests as g_requests
        from google.oauth2 import id_token as g_id_token

        claims = g_id_token.verify_oauth2_token(
            id_token_str,
            g_requests.Request(),
            settings.google_client_id,
            clock_skew_in_seconds=10,
        )
    except Exception as exc:  # noqa: BLE001 — normalize all verification errors
        raise GoogleAuthError(f"id_token verification failed: {exc}") from exc
    if claims.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise GoogleAuthError("unexpected token issuer")
    return claims
