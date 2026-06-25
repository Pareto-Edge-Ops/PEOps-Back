#!/usr/bin/env python3
"""Assert the COST & SAVINGS lens reflects the live serving telemetry.

Drives a multi-accelerator fleet at a deployed model, then verifies:
  • /telemetry/cost — live source, per-hardware $/1M, the GPU cheaper than the
    CPU, the original-vs-compressed counterfactual + savings math internally
    consistent, and a monthly bill asserted from the measured QPS.
  • /telemetry/cost?projectQps=… — a labeled projection without clobbering the
    measured monthly.
  • /dashboard/cost-savings — the workspace rollup reconciles and reports live.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

PASS, FAIL = "✓", "✗"
_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {PASS if ok else FAIL} {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--handoff", required=True)
    args = ap.parse_args()
    with open(args.handoff) as f:
        h = json.load(f)
    mid = h["modelId"]
    c = httpx.Client(base_url=args.base.rstrip("/"), cookies=h["cookies"], timeout=60.0)

    def get(path: str):
        r = c.get(path)
        r.raise_for_status()
        return r.json()

    print("── inject a multi-accelerator fleet so the cost lens has live traffic")
    r = c.post(f"/api/models/{mid}/telemetry/simulate",
               json={"count": 600, "hours": 6, "incidents": False, "fleet": True})
    r.raise_for_status()
    time.sleep(1.0)

    print("\nPhase A — per-model cost (live)")
    cost = get(f"/api/models/{mid}/telemetry/cost")
    check("source is live", cost.get("source") == "live", cost.get("source"))
    check("per-hardware cost rows present", len(cost.get("perHardware", [])) >= 1,
          f'{len(cost.get("perHardware", []))} rows')
    check("compressed $/1M is priced", cost.get("compressedPer1M", 0) > 0,
          f'${cost.get("compressedPer1M")}/1M')
    check("a monthly bill is asserted from measured QPS",
          cost.get("measuredQps", 0) > 0 and cost.get("monthlyCompressed") is not None,
          f'qps={cost.get("measuredQps")}, monthly=${cost.get("monthlyCompressed")}')

    rows = cost.get("perHardware", [])
    gpu = [r["compressedPer1M"] for r in rows if r["accelerator"] == "gpu"]
    cpu = [r["compressedPer1M"] for r in rows if r["accelerator"] in ("cpu", "hosted")]
    if gpu and cpu:
        check("GPU serves the same artifact cheaper than CPU", min(gpu) < max(cpu),
              f"GPU ${min(gpu)}/1M < CPU ${max(cpu)}/1M")

    ratio = cost.get("assumedLatencyRatio")
    if ratio:
        check("original cost is the disclosed counterfactual",
              cost.get("originalPer1M") == round(cost["compressedPer1M"] * ratio, 4),
              f'orig ${cost.get("originalPer1M")} == ${cost["compressedPer1M"]}×{ratio}')
        check("savings% follows the latency ratio",
              cost.get("savingsPct") == round(100 * (1 - 1 / ratio), 1),
              f'{cost.get("savingsPct")}%')
    if cost.get("monthlySavings") is not None:
        check("monthly savings reconciles",
              cost["monthlySavings"] == round(cost["monthlyOriginal"] - cost["monthlyCompressed"], 2),
              f'${cost["monthlySavings"]}/mo')

    print("\nPhase B — projection")
    proj = get(f"/api/models/{mid}/telemetry/cost?projectQps=200")
    check("a target QPS yields a labeled projection",
          proj.get("projected") is True and proj.get("projectedMonthlyCompressed") is not None,
          f'projected=${proj.get("projectedMonthlyCompressed")}/mo')

    print("\nPhase C — workspace rollup")
    ws = get("/api/dashboard/cost-savings")
    check("workspace reports live traffic", ws.get("hasLiveTraffic") is True,
          f'liveModels={ws.get("liveModelCount")}')
    check("workspace savings reconciles",
          ws.get("monthlySavings") is not None
          and ws["monthlySavings"] == round(ws["monthlyOriginal"] - ws["monthlyCompressed"], 2),
          f'${ws.get("monthlySavings")}/mo across {ws.get("liveModelCount")} models')

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        sys.exit(1)
    print("COST & SAVINGS E2E VERIFIED — live per-hardware $/1M, original-vs-compressed "
          "counterfactual, monthly bill from measured QPS, and the workspace rollup")


if __name__ == "__main__":
    main()
