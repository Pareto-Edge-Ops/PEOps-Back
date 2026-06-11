"""PeopsClient + HttpSession against httpx.MockTransport."""

from __future__ import annotations

import httpx
import pytest

from peops_sdk import InferenceError, PeopsClient
from peops_sdk._http import ApiError, HttpSession


def _patch_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original_init = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)


def test_infer_happy_path(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/infer/dep_x"
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(200, json={"latencyMs": 1.2, "outputs": []})

    _patch_transport(monkeypatch, handler)
    with PeopsClient("http://t", "dep_x", "k") as c:
        out = c.infer({"input": [[1.0]]})
    assert out["latencyMs"] == 1.2


def test_infer_error_mapping(monkeypatch):
    def handler(request):
        return httpx.Response(404, json={"detail": {
            "code": "deployment_not_found", "message": "nope"}})

    _patch_transport(monkeypatch, handler)
    with PeopsClient("http://t", "dep_x", "k") as c:
        with pytest.raises(InferenceError) as exc:
            c.infer()
    assert exc.value.code == "deployment_not_found"
    assert exc.value.status == 404


def test_http_session_retries_503(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"ok": True})

    _patch_transport(monkeypatch, handler)
    monkeypatch.setattr("peops_sdk._http.time.sleep", lambda _s: None)
    s = HttpSession("http://t", "k", max_attempts=3)
    assert s.request("GET", "/x").json() == {"ok": True}
    assert calls["n"] == 3
    s.close()


def test_http_session_no_retry_on_401(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"detail": {"code": "invalid_api_key",
                                                    "message": "bad"}})

    _patch_transport(monkeypatch, handler)
    s = HttpSession("http://t", "k", max_attempts=3)
    with pytest.raises(ApiError) as exc:
        s.request("GET", "/x")
    assert exc.value.code == "invalid_api_key"
    assert calls["n"] == 1
    s.close()
