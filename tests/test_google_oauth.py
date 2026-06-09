"""Google OAuth contract — the network calls (token exchange + id_token verify)
are mocked, so the redirect/state/provisioning/cookie logic is exercised without
hitting Google."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from app.auth import google
from app.config import get_settings

TEST_CLIENT_ID = "test-client-id.apps.googleusercontent.com"


def _enable_google(monkeypatch) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "google_client_id", TEST_CLIENT_ID)
    monkeypatch.setattr(s, "google_client_secret", "test-secret")
    monkeypatch.setattr(s, "google_redirect_uri",
                        "http://localhost:8080/api/auth/google/callback")


def _mock_google(monkeypatch, *, sub: str, email: str, name: str, verified: bool = True):
    monkeypatch.setattr(google, "exchange_code", lambda code: {"id_token": "fake.jwt"})
    monkeypatch.setattr(google, "verify_id_token", lambda tok: {
        "iss": "https://accounts.google.com",
        "sub": sub, "email": email, "email_verified": verified, "name": name,
    })


def _no_follow(client: TestClient) -> TestClient:
    return TestClient(client.app, follow_redirects=False)


# ── providers ─────────────────────────────────────────────────────────────────

def test_providers_default_google_disabled(client):
    body = client.get("/api/auth/providers").json()
    assert body["password"] is True
    assert body["google"] is False  # no client id/secret configured by default


def test_providers_google_enabled(client, monkeypatch):
    _enable_google(monkeypatch)
    assert client.get("/api/auth/providers").json()["google"] is True


# ── login redirect ────────────────────────────────────────────────────────────

def test_google_login_redirects_to_google(client, monkeypatch):
    _enable_google(monkeypatch)
    c = _no_follow(client)
    r = c.get("/api/auth/google/login")
    assert r.status_code == 302
    loc = r.headers["location"]
    assert loc.startswith("https://accounts.google.com/o/oauth2/v2/auth")
    q = parse_qs(urlparse(loc).query)
    assert q["client_id"] == [TEST_CLIENT_ID]
    assert q["response_type"] == ["code"]
    assert q["redirect_uri"] == ["http://localhost:8080/api/auth/google/callback"]
    assert "openid" in q["scope"][0] and "email" in q["scope"][0]
    assert q["state"][0]
    # state cookie is set for CSRF protection on the callback
    assert "peops_oauth_state" in r.cookies


def test_google_login_disabled_redirects_with_error(client):
    c = _no_follow(client)
    r = c.get("/api/auth/google/login")  # google disabled by default
    assert r.status_code == 302
    assert r.headers["location"] == "/login?error=google_disabled"


# ── callback ──────────────────────────────────────────────────────────────────

def _login_state(c: TestClient) -> str:
    r = c.get("/api/auth/google/login")
    return parse_qs(urlparse(r.headers["location"]).query)["state"][0]


def test_google_callback_creates_user_and_session(client, monkeypatch):
    _enable_google(monkeypatch)
    _mock_google(monkeypatch, sub="g-sub-001", email="newgoogle@peops.dev", name="Google User")
    c = _no_follow(client)
    state = _login_state(c)  # also sets the state cookie in c's jar

    r = c.get(f"/api/auth/google/callback?code=auth-code&state={state}")
    assert r.status_code == 302
    assert r.headers["location"] == "/dashboard"
    assert "peops_session" in r.cookies

    # Follow-up: the session cookie authenticates /me as the new Google user.
    me = c.get("/api/auth/me").json()
    assert me["email"] == "newgoogle@peops.dev"
    assert me["authProvider"] == "google"


def test_google_callback_links_existing_email(client, monkeypatch):
    # A password account exists first…
    pw = TestClient(client.app)
    pw.post("/api/auth/signup", json={
        "email": "linkme@peops.dev", "password": "pw-12345678", "name": "Linker"})
    id_pw = pw.get("/api/auth/me").json()["id"]

    # …then the same person signs in with Google (same verified email) → same account.
    _enable_google(monkeypatch)
    _mock_google(monkeypatch, sub="g-sub-link", email="linkme@peops.dev", name="Linker")
    c = _no_follow(client)
    state = _login_state(c)
    r = c.get(f"/api/auth/google/callback?code=x&state={state}")
    assert r.status_code == 302
    me = c.get("/api/auth/me").json()
    assert me["id"] == id_pw  # linked, not a duplicate account


def test_google_callback_state_mismatch_rejected(client, monkeypatch):
    _enable_google(monkeypatch)
    _mock_google(monkeypatch, sub="g-sub-x", email="x@peops.dev", name="X")
    c = _no_follow(client)
    _login_state(c)  # sets a state cookie
    r = c.get("/api/auth/google/callback?code=x&state=WRONG-STATE")
    assert r.status_code == 302
    assert r.headers["location"] == "/login?error=state"


def test_google_callback_unverified_email_rejected(client, monkeypatch):
    _enable_google(monkeypatch)
    _mock_google(monkeypatch, sub="g-sub-unv", email="unv@peops.dev", name="Unv",
                 verified=False)
    c = _no_follow(client)
    state = _login_state(c)
    r = c.get(f"/api/auth/google/callback?code=x&state={state}")
    assert r.status_code == 302
    assert r.headers["location"] == "/login?error=email_unverified"


def test_google_transport_dependency_installed():
    # google-auth's id_token verification needs the `requests` transport; this
    # caught a real prod bug where the dep was missing from the image.
    import google.auth.transport.requests  # noqa: F401


def test_verify_id_token_is_graceful_on_bad_token(monkeypatch):
    # The REAL verify path (not mocked) must raise GoogleAuthError — never an
    # ImportError / unhandled exception that would 500 the callback.
    _enable_google(monkeypatch)
    with pytest.raises(google.GoogleAuthError):
        google.verify_id_token("not.a.real.token")


def test_google_only_account_cannot_password_login(client, monkeypatch):
    _enable_google(monkeypatch)
    _mock_google(monkeypatch, sub="g-sub-only", email="googleonly@peops.dev", name="GO")
    c = _no_follow(client)
    state = _login_state(c)
    c.get(f"/api/auth/google/callback?code=x&state={state}")  # creates google-only user

    # Password login for a google-only account must fail (no password hash).
    r = client.post("/api/auth/login",
                    json={"email": "googleonly@peops.dev", "password": "anything-123"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "invalid_credentials"
