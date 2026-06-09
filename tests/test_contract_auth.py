"""Auth + multi-tenancy contract: cookie session lifecycle, structured errors,
and strict per-user data isolation."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _fresh(client: TestClient) -> TestClient:
    """A new client over the same app — its own (empty) cookie jar."""
    return TestClient(client.app)


def _signup(c: TestClient, email: str, name: str = "User", pw: str = "pw-12345678"):
    return c.post("/api/auth/signup", json={"email": email, "password": pw, "name": name})


# ── session lifecycle ─────────────────────────────────────────────────────────

def test_signup_me_logout_flow(client):
    c = _fresh(client)
    r = _signup(c, "alice@peops.dev", "Alice")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["email"] == "alice@peops.dev"
    assert body["name"] == "Alice"
    assert body["id"].startswith("u_")
    assert "password" not in body and "password_hash" not in body

    me = c.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "alice@peops.dev"

    assert c.post("/api/auth/logout").status_code == 200
    after = c.get("/api/auth/me")
    assert after.status_code == 401
    assert after.json()["detail"]["code"] == "not_authenticated"


def test_login_after_signup(client):
    c = _fresh(client)
    _signup(c, "bob@peops.dev", "Bob")
    c.post("/api/auth/logout")
    r = c.post("/api/auth/login", json={"email": "bob@peops.dev", "password": "pw-12345678"})
    assert r.status_code == 200
    assert c.get("/api/auth/me").json()["email"] == "bob@peops.dev"


def test_email_is_case_insensitive(client):
    c = _fresh(client)
    _signup(c, "Carol@Peops.dev", "Carol")
    c.post("/api/auth/logout")
    r = c.post("/api/auth/login", json={"email": "carol@peops.dev", "password": "pw-12345678"})
    assert r.status_code == 200


# ── structured errors ─────────────────────────────────────────────────────────

def test_login_wrong_password_is_structured_401(client):
    c = _fresh(client)
    _signup(c, "dave@peops.dev", "Dave")
    r = c.post("/api/auth/login", json={"email": "dave@peops.dev", "password": "wrongwrong"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "invalid_credentials"


def test_login_unknown_email_same_message(client):
    c = _fresh(client)
    r = c.post("/api/auth/login", json={"email": "nobody@peops.dev", "password": "whatever1"})
    assert r.status_code == 401
    # No user enumeration: identical code/message to wrong-password.
    assert r.json()["detail"]["code"] == "invalid_credentials"


def test_duplicate_email_409(client):
    c = _fresh(client)
    assert _signup(c, "erin@peops.dev").status_code == 200
    c2 = _fresh(client)
    r = _signup(c2, "erin@peops.dev")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "email_taken"


def test_weak_password_400(client):
    c = _fresh(client)
    r = c.post("/api/auth/signup",
               json={"email": "frank@peops.dev", "password": "short", "name": "Frank"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "weak_password"


def test_invalid_email_400(client):
    c = _fresh(client)
    r = c.post("/api/auth/signup",
               json={"email": "not-an-email", "password": "pw-12345678", "name": "X"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "invalid_email"


# ── gate ──────────────────────────────────────────────────────────────────────

def test_unauthenticated_requests_are_blocked(client):
    anon = _fresh(client)
    for path in ("/api/models", "/api/dashboard/summary", "/api/sdk/snippets"):
        r = anon.get(path)
        assert r.status_code == 401, path
        assert r.json()["detail"]["code"] == "not_authenticated"


def test_healthz_is_public(client):
    anon = _fresh(client)
    assert anon.get("/healthz").status_code == 200


# ── multi-tenancy isolation ───────────────────────────────────────────────────

def test_user_isolation(client):
    a = _fresh(client)
    _signup(a, "owner@peops.dev", "Owner")
    b = _fresh(client)
    _signup(b, "intruder@peops.dev", "Intruder")

    imp = a.post("/api/models/import", json={"fileName": "iso-test.onnx"})
    assert imp.status_code == 200
    model_id = imp.json()["modelId"]

    # Owner sees it.
    a_ids = {m["id"] for m in a.get("/api/models").json()}
    assert model_id in a_ids
    assert a.get(f"/api/models/{model_id}").status_code == 200

    # Intruder does not — not in the list, and 404 (not 403) on every scoped route.
    b_ids = {m["id"] for m in b.get("/api/models").json()}
    assert model_id not in b_ids
    assert b.get(f"/api/models/{model_id}").status_code == 404
    assert b.get(f"/api/models/{model_id}/architecture").status_code == 404
    assert b.get(f"/api/models/{model_id}/pareto").status_code == 404
    assert b.get(f"/api/models/{model_id}/telemetry/kpi").status_code == 404
    assert b.delete(f"/api/models/{model_id}").status_code == 404
    # The model still exists for its owner after the intruder's failed delete.
    assert a.get(f"/api/models/{model_id}").status_code == 200
