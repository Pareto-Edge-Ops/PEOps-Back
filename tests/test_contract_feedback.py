"""Feedback intake — always persisted; the GitHub issue is a best-effort side
effect that stays off unless a repo+token are configured (the suite has neither).

The submit endpoint is multipart (Form fields + an optional image), so requests
go through `data=` (and `files=` when attaching a screenshot)."""

from __future__ import annotations

from app.config import get_settings

# Minimal stand-in image bytes — the serve path streams them back verbatim; it
# doesn't decode the image, so any bytes with an image extension work.
_PNG = b"\x89PNG\r\n\x1a\nfake-screenshot-bytes"


def test_submit_feedback_returns_open_status(client):
    r = client.post("/api/feedback", data={
        "kind": "feature", "message": "A dark-mode toggle would be great",
        "page": "/dashboard", "locale": "en",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"id", "status", "githubIssueUrl"}
    assert body["id"].startswith("fb_")
    assert body["status"] == "open"
    # No GitHub repo configured in the suite → no issue is opened.
    assert body["githubIssueUrl"] is None


def test_submit_feedback_defaults_kind_to_feature(client):
    r = client.post("/api/feedback", data={"message": "Just some thoughts"})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "open"


def test_submit_feedback_rejects_blank_message(client):
    r = client.post("/api/feedback", data={"kind": "bug", "message": "   "})
    assert r.status_code == 422


def test_submit_feedback_without_image_has_no_attachment(client):
    """A submission with no image gets no attachment — the serve endpoint 404s."""
    r = client.post("/api/feedback", data={"message": "no screenshot here"})
    assert r.status_code == 200, r.text
    fb_id = r.json()["id"]
    got = client.get(f"/api/feedback/{fb_id}/attachment")
    assert got.status_code == 404


def test_submit_feedback_with_image_round_trips(client):
    """An attached screenshot is stored and served back with the right type."""
    r = client.post(
        "/api/feedback",
        data={"kind": "bug", "message": "see the screenshot"},
        files={"image": ("shot.png", _PNG, "image/png")},
    )
    assert r.status_code == 200, r.text
    fb_id = r.json()["id"]

    got = client.get(f"/api/feedback/{fb_id}/attachment")
    assert got.status_code == 200, got.text
    assert got.headers["content-type"].startswith("image/png")
    assert got.content == _PNG


def test_submit_feedback_rejects_non_image(client):
    r = client.post(
        "/api/feedback",
        data={"message": "here are my notes"},
        files={"image": ("notes.txt", b"plain text", "text/plain")},
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unsupported_image"


def test_submit_feedback_rejects_oversized_image(client, monkeypatch):
    # Shrink the cap to 0 MB so any non-empty image trips the size guard, instead
    # of staging a real >10 MB buffer. get_settings() is cached, so patching the
    # live instance's attribute is what the size check reads.
    monkeypatch.setattr(get_settings(), "feedback_image_max_mb", 0)
    r = client.post(
        "/api/feedback",
        data={"message": "big picture"},
        files={"image": ("big.png", _PNG, "image/png")},
    )
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "image_too_large"


def test_feedback_requires_auth(client):
    """The router sits behind the session gate — a cookie-less client is bounced.

    Depends on `client` only to guarantee the suite env + app are configured;
    a fresh TestClient carries no session cookie."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as anon:
        r = anon.post("/api/feedback", data={"kind": "other", "message": "hi"})
        assert r.status_code == 401
