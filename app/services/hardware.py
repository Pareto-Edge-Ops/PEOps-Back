"""Hardware-aware telemetry — turn raw events + client snapshots into the two
views the platform's value prop actually needs:

  • per-hardware inference speed  — the same compressed artifact is served on
    heterogeneous hardware (CPU / CUDA GPU / Apple CoreML / edge ARM). Group the
    measured latencies by the hardware that served them so the dashboard answers
    "how fast is this model on each accelerator, and which is cheapest?".
  • resource utilization over time — CPU%, host memory, GPU util%, and GPU VRAM
    sampled from the SDK snapshots, time-bucketed over the selected range.

Events themselves don't carry hardware, but every SDK client_id has exactly one
hardware identity (from its snapshots), so we build a client_id → hardware map
from the latest snapshot per client and group events through it. Server-sourced
events (hosted /v1/infer, no client_id) fall into a "PEOps hosted" bucket.
"""

from __future__ import annotations

import json

from sqlmodel import Session, select

from app.config import iso
from app.dbmodels import InferenceEventRow, TelemetrySnapshotRow
from app.services.telemetry_agg import _bucket_bounds, _parse, _pct, _window

# ── cost model ────────────────────────────────────────────────────────────────
# Indicative on-demand $/hr for a representative single-accelerator instance.
# Used only to put inference latency into a "$ per 1M inferences" frame — an
# estimate (single-stream), clearly labeled as such in the UI.
_GPU_HOURLY: list[tuple[str, float]] = [
    ("a100", 3.67),    # p4d-class
    ("a10g", 1.006),   # g5.xlarge
    ("l4", 0.70),      # g6.xlarge
    ("l40", 1.95),
    ("v100", 3.06),    # p3.2xlarge
    ("t4", 0.526),     # g4dn.xlarge
    ("h100", 12.29),
]
_CPU_X86_HOURLY = 0.085   # ~c6i.large (2 vCPU)
_CPU_ARM_HOURLY = 0.068   # ~c7g.large (2 vCPU, Graviton)
_ONDEVICE_HOURLY = 0.0    # Apple/edge on-device — no marginal cloud cost


def _gpu_hourly(gpu_name: str) -> float:
    low = gpu_name.lower()
    for needle, rate in _GPU_HOURLY:
        if needle in low:
            return rate
    return 1.0  # unknown discrete GPU


def classify(hw: dict) -> dict:
    """Map a hardware identity → {deviceClass, accelerator, hourlyUsd}.

    `hw` carries provider/arch/gpuName/cpuModel (already parsed from a snapshot's
    runtime_json). `accelerator` is one of gpu | coreml | cpu | hosted."""
    gpu_name = (hw.get("gpuName") or "").strip()
    provider = (hw.get("activeProvider") or hw.get("provider") or "").strip()
    arch = (hw.get("arch") or "").lower()
    plow = provider.lower()

    if gpu_name and ("cuda" in plow or "tensorrt" in plow or "rocm" in plow or
                     hw.get("gpuCount")):
        return {"deviceClass": gpu_name, "accelerator": "gpu",
                "hourlyUsd": _gpu_hourly(gpu_name)}
    if "coreml" in plow:
        label = hw.get("cpuModel") or "Apple Silicon"
        return {"deviceClass": f"{label} · CoreML", "accelerator": "coreml",
                "hourlyUsd": _ONDEVICE_HOURLY}
    if "cuda" in plow or "tensorrt" in plow:  # GPU provider, name unknown
        return {"deviceClass": "CUDA GPU", "accelerator": "gpu",
                "hourlyUsd": 1.0}
    # CPU.
    is_arm = arch.startswith("arm") or arch.startswith("aarch")
    label = hw.get("cpuModel") or ("ARM64 CPU" if is_arm else "x86-64 CPU")
    return {"deviceClass": label, "accelerator": "cpu",
            "hourlyUsd": _CPU_ARM_HOURLY if is_arm else _CPU_X86_HOURLY}


def est_cost_per_million(mean_latency_ms: float, hourly_usd: float) -> float:
    """Single-stream estimate: 1M inferences at L ms each take 1000·L seconds;
    cost = hourly · (1000·L / 3600) = hourly · L / 3.6 (USD)."""
    if mean_latency_ms <= 0 or hourly_usd <= 0:
        return 0.0
    return round(hourly_usd * mean_latency_ms / 3.6, 4)


# ── client_id → hardware map ──────────────────────────────────────────────────


def _client_hardware(session: Session, model_id: str) -> dict[str, dict]:
    """Latest snapshot per client_id → its parsed hardware identity (+ live
    resource sample)."""
    rows = session.exec(
        select(TelemetrySnapshotRow)
        .where(TelemetrySnapshotRow.model_id == model_id)
        .order_by(TelemetrySnapshotRow.ts.desc())  # type: ignore[attr-defined]
        .limit(1000)
    ).all()
    latest: dict[str, dict] = {}
    for r in rows:
        if r.client_id in latest:
            continue
        try:
            rt = json.loads(r.runtime_json)
        except ValueError:
            rt = {}
        rt["arch"] = rt.get("arch", "")
        latest[r.client_id] = {
            "hw": rt,
            "gpuUtilPct": r.gpu_util_pct,
            "gpuMemUsedMb": r.gpu_mem_used_mb,
            "cpuPct": r.cpu_pct,
        }
    return latest


# ── per-hardware inference speed ──────────────────────────────────────────────


def hardware_breakdown(session: Session, model_id: str, range_str: str = "24h") -> list[dict]:
    """Group measured inference latencies by the hardware that served them."""
    window, _ = _window(range_str)
    start, _, _ = _bucket_bounds(range_str)
    end = start + window
    events = session.exec(
        select(InferenceEventRow).where(
            InferenceEventRow.model_id == model_id,
            InferenceEventRow.ts >= iso(start),
            InferenceEventRow.ts < iso(end),
        )
    ).all()
    if not events:
        return []

    cmap = _client_hardware(session, model_id)
    win_min = max(1e-6, window.total_seconds() / 60.0)

    # Accumulate per group.
    groups: dict[str, dict] = {}
    for e in events:
        cid = e.client_id or ""
        if e.source == "client" and cid in cmap:
            hw = cmap[cid]["hw"]
            info = classify(hw)
            key = f'{info["accelerator"]}:{info["deviceClass"]}'
            gpu_name = hw.get("gpuName", "")
            provider = hw.get("activeProvider") or hw.get("provider", "")
        else:
            # Hosted serving path (CPUExecutionProvider, no SDK snapshot).
            info = {"deviceClass": "PEOps hosted · CPU", "accelerator": "hosted",
                    "hourlyUsd": _CPU_X86_HOURLY}
            key = "hosted:PEOps hosted · CPU"
            gpu_name = ""
            provider = "CPUExecutionProvider"
        g = groups.setdefault(key, {
            "key": key, "label": info["deviceClass"], "deviceClass": info["deviceClass"],
            "accelerator": info["accelerator"], "provider": provider, "gpuName": gpu_name,
            "hourlyUsd": info["hourlyUsd"], "clients": set(), "lat": [], "errors": 0,
            "samples": 0,
        })
        g["samples"] += 1
        if cid:
            g["clients"].add(cid)
        if e.success:
            g["lat"].append(e.latency_ms)
        else:
            g["errors"] += 1

    out: list[dict] = []
    for g in groups.values():
        lat = g["lat"]
        mean_lat = round(sum(lat) / len(lat), 3) if lat else 0.0
        # capacity proxy: a single serving stream does 1000/mean_latency req/s.
        tput = round(1000.0 / mean_lat, 1) if mean_lat > 0 else 0.0
        # average live resource across this group's clients.
        cpu_vals = [cmap[c]["cpuPct"] for c in g["clients"] if c in cmap]
        gpu_u = [cmap[c]["gpuUtilPct"] for c in g["clients"]
                 if c in cmap and cmap[c]["gpuUtilPct"] is not None]
        gpu_m = [cmap[c]["gpuMemUsedMb"] for c in g["clients"]
                 if c in cmap and cmap[c]["gpuMemUsedMb"] is not None]
        out.append({
            "key": g["key"],
            "label": g["label"],
            "deviceClass": g["deviceClass"],
            "accelerator": g["accelerator"],
            "provider": g["provider"],
            "gpuName": g["gpuName"],
            "hostCount": max(1, len(g["clients"])) if g["accelerator"] != "hosted" else 1,
            "samples": g["samples"],
            "reqPerMin": round(g["samples"] / win_min, 1),
            "p50": round(_pct(lat, 50), 3),
            "p95": round(_pct(lat, 95), 3),
            "inferenceMs": mean_lat,
            "throughputPerSec": tput,
            "errorRate": round(100.0 * g["errors"] / g["samples"], 3) if g["samples"] else 0.0,
            "avgCpuPct": round(sum(cpu_vals) / len(cpu_vals), 1) if cpu_vals else None,
            "avgGpuUtilPct": round(sum(gpu_u) / len(gpu_u), 1) if gpu_u else None,
            "avgGpuMemUsedMb": round(sum(gpu_m) / len(gpu_m), 1) if gpu_m else None,
            "estCostPer1M": est_cost_per_million(mean_lat, g["hourlyUsd"]),
        })
    # Fastest first (lowest p95); hosted/cpu naturally sink below accelerators.
    out.sort(key=lambda r: (r["p95"] if r["p95"] > 0 else 1e12))
    return out


# ── resource utilization time-series ──────────────────────────────────────────


def resource_series(session: Session, model_id: str, range_str: str = "24h") -> dict:
    """Time-bucketed CPU%, host memory, GPU util% and GPU VRAM from snapshots."""
    start, size, n = _bucket_bounds(range_str)
    end = start + size * n
    rows = session.exec(
        select(TelemetrySnapshotRow).where(
            TelemetrySnapshotRow.model_id == model_id,
            TelemetrySnapshotRow.ts >= iso(start),
            TelemetrySnapshotRow.ts < iso(end),
        )
    ).all()
    if not rows:
        return {"points": [], "hasGpu": False}

    size_sec = size.total_seconds()
    buckets: list[dict] = [
        {"cpu": [], "mem": [], "gpu": [], "gmem": []} for _ in range(n)
    ]
    has_gpu = False
    for r in rows:
        try:
            dt = _parse(r.ts)
        except ValueError:
            continue
        idx = int((dt - start).total_seconds() / size_sec) if size_sec > 0 else 0
        if not (0 <= idx < n):
            continue
        b = buckets[idx]
        b["cpu"].append(r.cpu_pct)
        b["mem"].append(r.rss_mb)
        if r.gpu_util_pct is not None:
            b["gpu"].append(r.gpu_util_pct)
            has_gpu = True
        if r.gpu_mem_used_mb is not None:
            b["gmem"].append(r.gpu_mem_used_mb)

    def _avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 1) if xs else None

    points = []
    for i, b in enumerate(buckets):
        if not (b["cpu"] or b["mem"] or b["gpu"]):
            continue
        points.append({
            "t": iso(start + size * i),
            "cpuPct": _avg(b["cpu"]) or 0.0,
            "memMb": _avg(b["mem"]) or 0.0,
            "gpuUtilPct": _avg(b["gpu"]),
            "gpuMemUsedMb": _avg(b["gmem"]),
        })
    return {"points": points, "hasGpu": has_gpu}
