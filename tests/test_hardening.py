"""Hardening contract: readiness probe, upload validation, structured errors."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_and_readyz(client):
    assert client.get("/healthz").json()["ok"] is True
    r = client.get("/readyz")
    body = r.json()
    assert r.status_code == 200, body
    assert body["ok"] is True
    assert body["checks"]["db"] == "ok"
    assert body["checks"]["storage"] == "ok"


def test_upload_rejects_unsupported_extension(client):
    r = client.post(
        "/api/models/upload",
        files={"file": ("notamodel.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unsupported_format"


def test_request_id_header_present(client):
    r = client.get("/api/sdk/snippets")
    assert "x-request-id" in {k.lower() for k in r.headers}


def test_response_id_on_error(client):
    anon = TestClient(client.app)
    r = anon.get("/api/models")  # 401, structured
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "not_authenticated"
