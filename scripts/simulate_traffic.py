#!/usr/bin/env python3
"""Drive REAL traffic at a deployed model's inference endpoint over HTTP.

Unlike the in-app demo burst (which writes events in-process), this hits the
genuine public path — POST /api/v1/infer/{deployment_id} with a bearer API key —
exactly as an external app or device would. Use it to watch the Telemetry
Dashboard come alive end-to-end, including drift alerts during incident bursts.

Example:
    python scripts/simulate_traffic.py \
        --base-url http://localhost:8000 \
        --deployment dep_ab12cd34ef \
        --api-key peops_sk_live_xxxxxxxx \
        --rate 5 --duration 120 --incidents
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import httpx


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PEOps inference traffic generator")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--deployment", required=True, help="deployment id (dep_…)")
    p.add_argument("--api-key", required=True, help="bearer key (peops_sk_live_…)")
    p.add_argument("--rate", type=float, default=5.0, help="requests per second")
    p.add_argument("--duration", type=float, default=120.0, help="seconds to run")
    p.add_argument("--incidents", action="store_true",
                   help="inject occasional latency/error incident bursts")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    url = f"{args.base_url.rstrip('/')}/api/v1/infer/{args.deployment}"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    interval = 1.0 / args.rate if args.rate > 0 else 0.2
    deadline = time.monotonic() + args.duration
    sent = ok = failed = 0
    incident_until = 0.0

    print(f"→ {url}  rate={args.rate}/s  duration={args.duration}s "
          f"incidents={args.incidents}")
    with httpx.Client(timeout=10.0) as client:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if args.incidents and now > incident_until and random.random() < 0.01:
                incident_until = now + random.uniform(5, 12)  # ~5–12s incident
            in_incident = now < incident_until
            # During an error incident, send a deliberately malformed input so the
            # server records a real failed event (drives the 5xx/error alert).
            if in_incident and random.random() < 0.4:
                payload = {"inputs": {"__nonexistent__": [[1.0]]}}
            else:
                payload = {"inputs": None}
            try:
                r = client.post(url, json=payload, headers=headers)
                sent += 1
                if r.status_code < 400:
                    ok += 1
                else:
                    failed += 1
            except httpx.HTTPError:
                sent += 1
                failed += 1
            if sent % 25 == 0:
                print(f"  sent={sent} ok={ok} failed={failed}", flush=True)
            time.sleep(interval * (random.uniform(0.5, 1.5) if not in_incident else 0.4))

    print(f"done — sent={sent} ok={ok} failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
