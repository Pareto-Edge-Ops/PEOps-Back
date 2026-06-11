"""Background telemetry reporter — fault-tolerant by construction.

Design contract: NOTHING in this module may ever raise into the caller's
serving path. Events queue into a bounded deque (drop-oldest under pressure,
with the drop count itself reported); a daemon thread flushes batches to
POST /api/v1/telemetry/{deployment_id}/batch with backoff; atexit performs a
final best-effort flush with a small time budget.

Disable entirely with report_telemetry=False or PEOPS_SDK_TELEMETRY=0.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

from ._http import HttpSession
from .stats import WindowAggregator
from .system import runtime_fingerprint, system_sample

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


_QUEUE_MAX = 10_000
_BATCH_MAX = 450            # below the server's 500-item cap
_FLUSH_INTERVAL_S = _env_float("PEOPS_SDK_FLUSH_INTERVAL_S", 5.0)
_SNAPSHOT_INTERVAL_S = _env_float("PEOPS_SDK_SNAPSHOT_INTERVAL_S", 30.0)
_WINDOW_INTERVAL_S = _env_float("PEOPS_SDK_WINDOW_INTERVAL_S", 60.0)
_WINDOW_MAX_REQUESTS = int(_env_float("PEOPS_SDK_WINDOW_MAX_REQUESTS", 200))
_ATEXIT_BUDGET_S = 3.0


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def telemetry_enabled(flag: bool | None = None) -> bool:
    if flag is not None and not flag:
        return False
    return os.environ.get("PEOPS_SDK_TELEMETRY", "1") not in ("0", "false", "no")


class TelemetryReporter:
    """Collects events/snapshots/windows and ships them in the background."""

    def __init__(
        self,
        base_url: str,
        deployment_id: str,
        api_key: str,
        *,
        sdk_version: str,
        enabled: bool = True,
    ) -> None:
        self.enabled = telemetry_enabled(enabled)
        self.client_id = f"sdk_{uuid.uuid4().hex[:10]}"
        self.deployment_id = deployment_id
        self._events: deque[dict] = deque(maxlen=_QUEUE_MAX)
        self._snapshots: deque[dict] = deque(maxlen=64)
        self._windows: deque[dict] = deque(maxlen=64)
        self._dropped = 0
        self._sent_events = 0
        self._lock = threading.Lock()
        self._window_lock = threading.Lock()
        self._aggregator = WindowAggregator()
        self._window_requests = 0
        self._fingerprint = runtime_fingerprint(sdk_version)
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._http: HttpSession | None = None
        self._throughput_marker = (time.monotonic(), 0)

        if self.enabled:
            self._http = HttpSession(
                base_url, api_key, timeout=10.0, max_attempts=2, max_backoff=30.0)
            self._thread = threading.Thread(
                target=self._loop, name="peops-telemetry", daemon=True)
            self._thread.start()
            atexit.register(self.close)

    # ── recording (hot path — must be cheap and never raise) ────────────────

    def record_event(
        self,
        *,
        latency_ms: float,
        pre_ms: float | None = None,
        post_ms: float | None = None,
        success: bool = True,
        error_code: str | None = None,
        batch_size: int = 1,
        region: str = "local",
        input_sig: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            event = {
                "ts": _iso_now(),
                "latencyMs": round(float(latency_ms), 3),
                "success": success,
                "batchSize": int(batch_size),
                "region": region,
            }
            if pre_ms is not None:
                event["preMs"] = round(float(pre_ms), 3)
            if post_ms is not None:
                event["postMs"] = round(float(post_ms), 3)
            if error_code:
                event["errorCode"] = error_code
            if input_sig:
                event["inputSig"] = input_sig
            with self._lock:
                if len(self._events) == self._events.maxlen:
                    self._dropped += 1
                self._events.append(event)
                self._sent_events += 1
                self._window_requests += 1
                cut_window = self._window_requests >= _WINDOW_MAX_REQUESTS
            if cut_window:
                # Request-driven window cut: keeps window sizes deterministic
                # under burst load instead of waiting for the next loop tick.
                self._take_window()
        except Exception:
            pass

    def observe(self, inputs: dict[str, Any] | None, output: Any | None) -> None:
        if not self.enabled:
            return
        try:
            self._aggregator.observe(inputs, output)
        except Exception:
            pass

    # ── background loop ──────────────────────────────────────────────────────

    def _loop(self) -> None:
        last_snapshot = last_window = time.monotonic()
        while not self._closed.wait(_FLUSH_INTERVAL_S):
            now = time.monotonic()
            try:
                if now - last_snapshot >= _SNAPSHOT_INTERVAL_S:
                    self._take_snapshot()
                    last_snapshot = now
                window_due = (
                    now - last_window >= _WINDOW_INTERVAL_S
                    or self._window_requests >= _WINDOW_MAX_REQUESTS
                )
                if window_due:
                    self._take_window()
                    last_window = now
                self._flush()
            except Exception:
                pass  # the loop must survive anything

    def _take_snapshot(self) -> None:
        now = time.monotonic()
        marker_t, marker_n = self._throughput_marker
        with self._lock:
            sent = self._sent_events
            dropped = self._dropped
        elapsed_min = max(1e-6, (now - marker_t) / 60.0)
        rpm = (sent - marker_n) / elapsed_min
        self._throughput_marker = (now, sent)
        self._snapshots.append({
            "ts": _iso_now(),
            **system_sample(),
            "throughputRpm": round(rpm, 2),
            "droppedEvents": dropped,
            **self._fingerprint,
        })

    def _take_window(self) -> None:
        # Serialized: callable from both the hot path (request-cap cut) and
        # the background loop (time-based cut).
        with self._window_lock:
            with self._lock:
                self._window_requests = 0
            window = self._aggregator.flush()
            if window:
                self._windows.append(window)

    def _flush(self) -> bool:
        if self._http is None:
            return True
        with self._lock:
            events = [self._events.popleft()
                      for _ in range(min(_BATCH_MAX, len(self._events)))]
        snapshots = [self._snapshots.popleft() for _ in range(len(self._snapshots))]
        windows = [self._windows.popleft() for _ in range(len(self._windows))]
        if not events and not snapshots and not windows:
            return True
        try:
            self._http.request(
                "POST", f"/api/v1/telemetry/{self.deployment_id}/batch",
                json={
                    "clientId": self.client_id,
                    "events": events,
                    "snapshots": snapshots,
                    "windows": windows,
                },
            )
            return True
        except Exception:
            # Re-queue at the FRONT so ordering survives one failed flush;
            # the bounded deque drops oldest under sustained failure.
            with self._lock:
                for ev in reversed(events):
                    if len(self._events) == self._events.maxlen:
                        self._dropped += 1
                        break
                    self._events.appendleft(ev)
            for sn in reversed(snapshots):
                self._snapshots.appendleft(sn)
            for w in reversed(windows):
                self._windows.appendleft(w)
            return False

    # ── shutdown ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Final best-effort flush within a small budget; idempotent."""
        if not self.enabled or self._closed.is_set():
            return
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            # Always ship at least one snapshot per session — it carries the
            # runtime fingerprint the dashboard's client-hosts table shows
            # (short sessions would otherwise never hit the 30s cadence).
            self._take_snapshot()
            self._take_window()
            deadline = time.monotonic() + _ATEXIT_BUDGET_S
            while time.monotonic() < deadline:
                if self._flush() and not self._events:
                    break
        except Exception:
            pass
        finally:
            if self._http is not None:
                self._http.close()
