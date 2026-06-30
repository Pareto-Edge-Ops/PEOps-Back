#!/usr/bin/env python3
"""End-to-end proof that the telemetry closed loop ACTUALLY works.

Run against a live backend (default http://localhost:8000) started with the
inline drift monitor on a short interval, e.g.:

    ASTRA_DB_PATH=/tmp/astra-verify/db.sqlite \
    ASTRA_STORAGE_DIR=/tmp/astra-verify/storage \
    ASTRA_WORK_DIR=/tmp/astra-verify/work \
    ASTRA_FAST_PIPELINE=1 ASTRA_INLINE_JOBS=1 \
    ASTRA_MONITOR_INLINE_ENABLED=1 ASTRA_MONITOR_INTERVAL_SEC=5 \
    ASTRA_COOKIE_SECURE=0 ASTRA_RATE_LIMIT_ENABLED=0 \
    uvicorn app.main:app --port 8000

    python3 scripts/verify_closed_loop.py --base http://localhost:8000

Checklist proven (each step asserts against the public API only):
  1.  signup → session cookie
  2.  model import → real fast pipeline completes
  3.  deployment + API key minted
  4.  60 real inferences through POST /api/v1/infer (Bearer key)
  5.  telemetry flips to source=live; KPI/series/percentiles are consistent
  6.  SSE stream delivers a snapshot frame
  7.  bad-input requests are recorded as failed events
  8.  the drift monitor pass raises a REAL "5xx error spike" alert
  9.  deployment live metrics (qps / errorsPct / lastEventAt) are maintained

Exit code 0 = closed loop verified; 1 = a step failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

import httpx

CHECK: list[tuple[str, bool]] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    CHECK.append((name, ok))
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        finish()


def finish() -> None:
    failed = [n for n, ok in CHECK if not ok]
    print("\n" + ("CLOSED LOOP VERIFIED" if not failed else f"FAILED: {failed}"))
    sys.exit(1 if failed else 0)


def wait_until(fn, timeout: float, interval: float = 0.5):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(interval)
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--monitor-interval", type=float, default=5.0,
                    help="ASTRA_MONITOR_INTERVAL_SEC the server was started with")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    c = httpx.Client(base_url=base, timeout=60.0)

    # 1 — auth
    email = f"verify+{uuid.uuid4().hex[:8]}@astra.dev"
    r = c.post("/api/auth/signup", json={
        "email": email, "password": "verify-pass-1234", "name": "Verify"})
    step("signup issues a session", r.status_code == 200, email)

    # 2 — import a model, real pipeline
    r = c.post("/api/models/import", json={"fileName": "closed-loop-verify.onnx"})
    step("model import accepted", r.status_code == 200)
    body = r.json()
    mid, rid = body["modelId"], body["runId"]

    deadline = time.time() + 180
    status = c.get(f"/api/models/{mid}/ingestion/{rid}").json()
    while status.get("status") == "streaming" and time.time() < deadline:
        time.sleep(0.5)
        status = c.get(f"/api/models/{mid}/ingestion/{rid}").json()
    step("pipeline completed", bool(status) and status["status"] == "completed",
         f"status={status.get('status') if status else 'n/a'}")
    c.post(f"/api/models/{mid}/ingestion/complete")

    # 3 — deployment + key
    r = c.post(f"/api/models/{mid}/deployments", json={"region": "ap-northeast-2"})
    step("deployment created + key minted", r.status_code == 200)
    dep = r.json()
    dep_id, api_key = dep["deployment"]["id"], dep["apiKey"]
    auth = {"Authorization": f"Bearer {api_key}"}

    # 4 — real traffic
    ok_count = 0
    for _ in range(60):
        rr = c.post(f"/api/v1/infer/{dep_id}", headers=auth,
                    json={"inputs": None, "batch": 1})
        ok_count += rr.status_code == 200
    step("60 real inferences served", ok_count == 60, f"{ok_count}/60 ok")

    # 5 — live aggregation
    meta = c.get(f"/api/models/{mid}/telemetry/meta").json()
    step("telemetry source flips to live", meta["source"] == "live",
         f"sources={meta.get('sources')}")
    kpi = c.get(f"/api/models/{mid}/telemetry/kpi?range=1h").json()
    step("KPI requests/min > 0", kpi["requestsPerMin"]["value"] > 0,
         f"req/min={kpi['requestsPerMin']['value']}")
    series = c.get(f"/api/models/{mid}/telemetry/series?range=1h").json()
    step("series has non-empty buckets", any(p["requests"] > 0 for p in series))
    pct = c.get(f"/api/models/{mid}/telemetry/percentiles?range=1h").json()
    v = pct["values"]
    step("percentiles ordered p50<=p95<=p99",
         v["p50"] <= v["p95"] <= v["p99"],
         f"p50={v['p50']} p95={v['p95']} p99={v['p99']}")

    # 6 — SSE
    got_snapshot = False
    try:
        with c.stream("GET", f"/api/models/{mid}/telemetry/stream",
                      timeout=15.0) as resp:
            event = None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event == "snapshot":
                    payload = json.loads(line.split(":", 1)[1])
                    got_snapshot = "source" in payload
                    break
    except httpx.HTTPError:
        pass
    step("SSE stream delivers a snapshot frame", got_snapshot)

    # 7 — induce real failures (bad input name → recorded failed events)
    bad = 0
    for _ in range(10):
        rr = c.post(f"/api/v1/infer/{dep_id}", headers=auth,
                    json={"inputs": {"nonexistent_input": [[1.0]]}})
        bad += rr.status_code >= 400
    step("bad inputs rejected AND recorded", bad == 10, f"{bad}/10 failed as expected")

    # 8 — the monitor pass raises a real alert (error rate > threshold)
    def find_alert():
        alerts = c.get(f"/api/models/{mid}/telemetry/alerts").json()
        return [a for a in alerts if a["title"] == "5xx error spike"] or None

    alerts = wait_until(find_alert, timeout=args.monitor_interval * 4 + 10)
    step("drift monitor raised '5xx error spike'", bool(alerts),
         alerts[0]["body"][:70] if alerts else "no alert within window")

    # 9 — live deployment metrics maintained by the monitor
    deps = c.get(f"/api/models/{mid}/telemetry/deployments").json()
    d = deps[0] if deps else {}
    step("deployment live metrics updated",
         bool(d) and d["qps"] > 0 and d["errorsPct"] > 1.0,
         f"qps={d.get('qps')} errors%={d.get('errorsPct')}")

    finish()


if __name__ == "__main__":
    main()
