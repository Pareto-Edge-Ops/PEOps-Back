"""TelemetryReporter semantics against a fake in-process server."""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from peops_sdk.telemetry import TelemetryReporter


class FakeBackend:
    """Captures batches; can be told to fail N times."""

    def __init__(self) -> None:
        self.batches: list[dict] = []
        self.fail_next = 0
        self.lock = threading.Lock()

    def handler(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            if self.fail_next > 0:
                self.fail_next -= 1
                return httpx.Response(503, json={"detail": {"code": "unavailable"}})
            import json

            self.batches.append(json.loads(request.content))
        return httpx.Response(200, json={"accepted": {}, "dropped": 0})

    @property
    def events(self) -> list[dict]:
        return [e for b in self.batches for e in b.get("events", [])]


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend()
    transport = httpx.MockTransport(fake.handler)
    original_init = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)
    return fake


def _reporter(**kw) -> TelemetryReporter:
    return TelemetryReporter(
        "http://test", "dep_x", "peops_sk_test", sdk_version="0.2.0", **kw)


def test_events_flush_on_close(backend):
    rep = _reporter()
    for i in range(25):
        rep.record_event(latency_ms=float(i), pre_ms=0.1, post_ms=0.1)
    rep.close()
    assert len(backend.events) == 25
    ev = backend.events[0]
    assert {"ts", "latencyMs", "success", "batchSize", "region",
            "preMs", "postMs"} <= set(ev)


def test_recording_never_blocks_or_raises(backend):
    rep = _reporter()
    t0 = time.perf_counter()
    for _ in range(5000):
        rep.record_event(latency_ms=1.0)
    assert time.perf_counter() - t0 < 1.0, "hot path must be cheap"
    rep.close()


def test_failed_flush_requeues_then_recovers(backend):
    backend.fail_next = 1
    rep = _reporter()
    for i in range(10):
        rep.record_event(latency_ms=float(i))
    rep.close()  # close retries within its budget
    assert len(backend.events) == 10, "events must survive one failed flush"


def test_disabled_via_flag(backend):
    rep = _reporter(enabled=False)
    rep.record_event(latency_ms=1.0)
    rep.close()
    assert backend.batches == []


def test_disabled_via_env(backend, monkeypatch):
    monkeypatch.setenv("PEOPS_SDK_TELEMETRY", "0")
    rep = _reporter()
    rep.record_event(latency_ms=1.0)
    rep.close()
    assert backend.batches == []


def test_window_stats_emitted_on_close(backend):
    np = pytest.importorskip("numpy")
    rep = _reporter()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal((1, 8)).astype(np.float32)
        logits = rng.standard_normal((1, 5))
        rep.observe({"input": x}, logits)
        rep.record_event(latency_ms=1.0)
    rep.close()
    windows = [w for b in backend.batches for w in b.get("windows", [])]
    assert windows, "close() must flush the open window"
    w = windows[0]
    assert w["n"] == 20
    assert "input" in w["inputs"]
    assert abs(w["inputs"]["input"]["mean"]) < 1.0
    assert "classDist" in w["output"]
    assert len(w["output"]["hist"]) == 16
