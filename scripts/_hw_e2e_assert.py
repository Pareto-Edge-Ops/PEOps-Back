#!/usr/bin/env python3
"""Assert the dashboard reflects the hardware-aware telemetry closed loop.

Two phases:
  A. REAL serve — the separate `peops serve` process reported genuine telemetry
     from THIS host, so the dashboard must show live client traffic, snapshots
     carrying real hardware identity (CPU model/cores/RAM + the bound ORT
     provider), a per-hardware group, and a CPU/memory resource series.
  B. FLEET sim — inject a multi-accelerator fleet so the GPU views (per-hardware
     speed incl. a GPU group + GPU resource utilization) are populated and
     verifiable on a box without an NVIDIA GPU.
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

    # ── Phase A: the REAL separate-server telemetry ───────────────────────────
    print("Phase A — real `peops serve` telemetry (this host)")
    meta = get(f"/api/models/{mid}/telemetry/meta")
    check("telemetry source is live", meta.get("source") == "live", meta.get("source"))
    check("client-sourced events ingested",
          (meta.get("sources", {}).get("client", 0)) > 0,
          f"client={meta.get('sources', {}).get('client')}")
    check("a system snapshot arrived", bool(meta.get("lastSnapshotAt")),
          str(meta.get("lastSnapshotAt")))

    clients = get(f"/api/models/{mid}/telemetry/clients")
    check("a serving host is visible", len(clients) >= 1, f"{len(clients)} host(s)")
    if clients:
        cl = clients[0]
        check("snapshot carries real CPU identity",
              bool(cl.get("cpuModel")) and cl.get("cpuCores", 0) > 0,
              f"{cl.get('cpuModel')} · {cl.get('cpuCores')} cores")
        check("snapshot records the bound ORT provider",
              bool(cl.get("activeProvider")), cl.get("activeProvider"))

    hw = get(f"/api/models/{mid}/telemetry/hardware")
    check("per-hardware breakdown has a group", len(hw) >= 1, f"{len(hw)} group(s)")
    if hw:
        g = hw[0]
        check("group reports latency + throughput + cost",
              g["p95"] > 0 and g["throughputPerSec"] > 0 and g["estCostPer1M"] >= 0,
              f'{g["label"]}: p95={g["p95"]}ms, {g["throughputPerSec"]}/s, ${g["estCostPer1M"]}/1M')

    res = get(f"/api/models/{mid}/telemetry/resources")
    check("resource series has CPU/memory points", len(res.get("points", [])) >= 1,
          f'{len(res.get("points", []))} bucket(s), hasGpu={res.get("hasGpu")}')

    # ── Phase B: the simulated multi-accelerator fleet ────────────────────────
    print("\nPhase B — multi-hardware fleet (GPU views)")
    r = c.post(f"/api/models/{mid}/telemetry/simulate",
               json={"count": 600, "hours": 6, "incidents": False, "fleet": True})
    r.raise_for_status()
    fleet = r.json().get("fleet", {})
    check("fleet injected GPU + CPU hosts", fleet.get("gpuHosts", 0) >= 1,
          f'{fleet.get("hosts")} hosts, {fleet.get("gpuHosts")} with GPU')
    time.sleep(1.0)

    hw2 = get(f"/api/models/{mid}/telemetry/hardware")
    accels = {g["accelerator"] for g in hw2}
    check("≥3 hardware groups after fleet", len(hw2) >= 3, f"{len(hw2)} groups: {sorted(accels)}")
    gpu_groups = [g for g in hw2 if g["accelerator"] == "gpu"]
    check("a GPU group is present", len(gpu_groups) >= 1,
          ", ".join(g["gpuName"] for g in gpu_groups))
    check("GPU group reports live utilization + cost",
          any((g["avgGpuUtilPct"] or 0) > 0 and g["estCostPer1M"] > 0 for g in gpu_groups))
    # The core promise: a GPU serves the SAME artifact faster than the CPU.
    cpu_groups = [g for g in hw2 if g["accelerator"] in ("cpu", "hosted")]
    if gpu_groups and cpu_groups:
        fastest_gpu = min(g["p95"] for g in gpu_groups)
        slowest_cpu = max(g["p95"] for g in cpu_groups)
        check("GPU is faster than CPU on the same model", fastest_gpu < slowest_cpu,
              f"GPU p95 {fastest_gpu}ms < CPU p95 {slowest_cpu}ms")

    res2 = get(f"/api/models/{mid}/telemetry/resources")
    check("resource series now reports GPU", res2.get("hasGpu") is True)
    check("a bucket carries GPU utilization",
          any(p.get("gpuUtilPct") for p in res2.get("points", [])))

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        sys.exit(1)
    print("HARDWARE TELEMETRY E2E VERIFIED — real separate-server serving + "
          "per-hardware speed + GPU resource views, all live on the dashboard")


if __name__ == "__main__":
    main()
