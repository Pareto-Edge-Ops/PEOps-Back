#!/usr/bin/env python3
"""Drive REAL HTTP inference against a separate `peops serve` server.

This talks to the standalone HTTP server the SDK started on another port
(127.0.0.1:8765/infer) — a genuinely separate Python process — over plain HTTP,
exactly as any application would. The server synthesizes a valid input probe
when the body omits `inputs`, so this stays model-agnostic. stdlib only.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def _post(url: str, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — localhost
        return json.loads(resp.read() or b"{}")


def _wait_up(url: str, deadline: float) -> None:
    while time.time() < deadline:
        try:
            _post(url, {}, timeout=5.0)
            return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    raise SystemExit(f"serve never came up at {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/infer")
    ap.add_argument("-n", type=int, default=150)
    args = ap.parse_args()

    _wait_up(args.url, time.time() + 30)
    lats: list[float] = []
    t0 = time.time()
    for i in range(args.n):
        out = _post(args.url, {})  # empty body → server synthesizes a valid probe
        lats.append(float(out.get("latencyMs", 0.0)))
        # Spread the calls out a little so the SDK ships ≥1 system snapshot
        # (snapshot cadence is set low by the orchestrator's env).
        if i % 25 == 24:
            time.sleep(0.6)
    lats.sort()
    p = lambda q: lats[min(len(lats) - 1, int(q * len(lats)))]  # noqa: E731
    print(json.dumps({
        "served": len(lats),
        "wallSec": round(time.time() - t0, 1),
        "p50Ms": round(p(0.50), 3),
        "p95Ms": round(p(0.95), 3),
    }, indent=2))


if __name__ == "__main__":
    main()
