#!/usr/bin/env python3
"""SDK e2e step 5: assert the dashboard observed the local serving session."""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--handoff", required=True)
    ap.add_argument("--monitor-wait", type=float, default=25.0)
    args = ap.parse_args()
    with open(args.handoff) as f:
        h = json.load(f)

    c = httpx.Client(base_url=args.base.rstrip("/"), timeout=30.0,
                     cookies=h["cookies"])
    mid = h["modelId"]
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"   {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    meta = c.get(f"/api/models/{mid}/telemetry/meta").json()
    check("telemetry source is live", meta["source"] == "live")
    check("client-sourced events >= 160",
          meta["sources"]["client"] >= 160, f"sources={meta['sources']}")
    check("system snapshot received", meta.get("lastSnapshotAt") is not None)

    kpi = c.get(f"/api/models/{mid}/telemetry/kpi?range=1h").json()
    check("KPI fed by client events", kpi["requestsPerMin"]["value"] > 0)

    hosts = c.get(f"/api/models/{mid}/telemetry/clients").json()
    check("client host visible", bool(hosts),
          f"{hosts[0]['host']} sdk {hosts[0]['sdkVersion']}" if hosts else "")

    bd = c.get(f"/api/models/{mid}/telemetry/breakdown?range=1h").json()
    check("latency breakdown points exist", bool(bd["points"]))

    stats = c.get(f"/api/models/{mid}/telemetry/output-stats?range=1h").json()
    check("output stats windows aggregated", stats["windows"] >= 5,
          f"windows={stats['windows']}")

    # Drift: wait for monitor passes to see the shifted windows.
    deadline = time.time() + args.monitor_wait
    titles: set[str] = set()
    while time.time() < deadline:
        alerts = c.get(f"/api/models/{mid}/telemetry/alerts").json()
        titles = {a["title"] for a in alerts}
        if "input distribution shift" in titles:
            break
        time.sleep(2)
    check("input drift alert raised", "input distribution shift" in titles,
          f"alerts={sorted(titles)}")

    if failures:
        print(f"\nSDK E2E FAILED: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
