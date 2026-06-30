"""Cost & savings lens — translate compression into dollars.

Every other telemetry view answers "how fast / how small"; this one answers
"how much money". It is built on the same hardware cost primitives
(``hardware.est_cost_per_million`` / ``hardware.hardware_breakdown``) and the
same live-vs-benchmark discipline as the rest of telemetry.

Honesty rules (the platform's "measured, not guessed" brand):
  • A monthly $ figure is asserted ONLY from real measured QPS (a live
    deployment's monitor-maintained metrics). With no live traffic we expose the
    per-1M unit cost + savings% and let the caller PROJECT a monthly figure at a
    target QPS (labeled as a projection).
  • Original cost is a counterfactual — we never run the uncompressed model on
    the user's hardware. We scale the measured compressed cost by the
    benchmarked latency ratio (original.p95 / compressed.p95). The ratio is
    surfaced (``assumedLatencyRatio``) so the UI can disclose the assumption.
"""

from __future__ import annotations

from sqlmodel import Session, select

from app.dbmodels import DeploymentRow, ModelRow
from app.repositories import get_cached_result, user_artifact_metas
from app.services import hardware as hw
from app.services.telemetry_agg import has_any_events

SEC_PER_MONTH = 2_592_000  # 30 * 86400
PER_M = 1_000_000


def _latency_ratio(bench: dict | None) -> float | None:
    """original.p95 / compressed.p95 — the measured speedup the compression
    bought. >1 is the expected (good) case. None when the benchmark lacks usable
    latencies (then original cost / savings% are honestly absent, never faked)."""
    if not bench:
        return None
    try:
        oc = float(bench["original"]["p95"])
        cc = float(bench["compressed"]["p95"])
    except (KeyError, TypeError, ValueError):
        return None
    if oc <= 0 or cc <= 0:
        return None
    return round(oc / cc, 4)


def monthly_cost(per_1m_usd: float, qps: float) -> float:
    """Monthly $ to serve ``qps`` inferences/sec at ``per_1m_usd`` per million.
    inferences/month = qps · SEC_PER_MONTH; cost = per_1m · that / 1e6."""
    if per_1m_usd <= 0 or qps <= 0:
        return 0.0
    return round(per_1m_usd * qps * SEC_PER_MONTH / PER_M, 2)


def _savings_pct(ratio: float | None) -> float | None:
    """% cheaper the compressed model is vs the original on equal hardware. Cost
    scales with single-stream latency, so savings% = 1 − compressed/original =
    1 − 1/ratio."""
    if not ratio or ratio <= 0:
        return None
    return round(100.0 * (1.0 - 1.0 / ratio), 1)


def _measured_qps(session: Session, model_id: str, groups: list[dict]) -> float:
    """Canonical live QPS: sum of non-paused deployments' monitor-maintained qps
    (the authoritative live number), falling back to the served reqPerMin when
    the monitor has not populated qps yet."""
    deps = session.exec(
        select(DeploymentRow).where(
            DeploymentRow.model_id == model_id,
            DeploymentRow.status != "paused",
        )
    ).all()
    qps = sum(d.qps for d in deps)
    if qps <= 0:
        qps = sum(g["reqPerMin"] for g in groups) / 60.0
    return round(qps, 3)


def _representative(groups: list[dict]) -> dict | None:
    """The hardware row the headline cost speaks for: the cheapest real paid
    accelerator (groups are already sorted fastest-first), else the first group.
    The hosted reference bucket is skipped so a trivial bench row never wins."""
    paid = [g for g in groups if g["accelerator"] != "hosted" and g["estCostPer1M"] > 0]
    if paid:
        return paid[0]
    return groups[0] if groups else None


def _empty_cost_summary() -> dict:
    """The cost lens for a deployed-but-untrafficked model: no numbers at all
    (the SPA renders the CardEmpty state)."""
    return {
        "source": "none",
        "compressedPer1M": 0.0,
        "originalPer1M": None,
        "savingsPer1M": None,
        "savingsPct": None,
        "assumedLatencyRatio": None,
        "measuredQps": 0.0,
        "monthlyCompressed": None,
        "monthlyOriginal": None,
        "monthlySavings": None,
        "projected": False,
        "projectedMonthlyCompressed": None,
        "projectedMonthlyOriginal": None,
        "projectedMonthlySavings": None,
        "perHardware": [],
    }


def model_cost_summary(
    session: Session,
    model: ModelRow,
    range_str: str = "24h",
    project_qps: float | None = None,
) -> dict:
    """Per-model $ summary. Empty (source:"none") until real traffic exists, so
    no benchmark-derived numbers leak in as if they were measured. A weights-only
    checkpoint still gets the structured 404 (the SPA gates on /meta before ever
    calling this, so the 404 is just defensive)."""
    from fastapi import HTTPException

    live = has_any_events(session, model.id)
    if not live:
        if model.weights_only:
            raise HTTPException(status_code=404, detail={
                "code": "weights_only_checkpoint",
                "message": "This checkpoint is weights-only (state_dict) — it "
                           "cannot be executed, so no cost estimate exists.",
            })
        return _empty_cost_summary()

    bench = get_cached_result(session, model.id, "benchmark", user_id=model.user_id)
    ratio = _latency_ratio(bench)
    savings_pct = _savings_pct(ratio)

    per_hardware: list[dict] = []
    compressed_per1m: float | None = None
    original_per1m: float | None = None
    measured_qps = 0.0

    def _orig(cp: float) -> float | None:
        return round(cp * ratio, 4) if (ratio and cp > 0) else None

    groups = hw.hardware_breakdown(session, model.id, range_str)
    for g in groups:
        cp = g["estCostPer1M"]
        op = _orig(cp)
        per_hardware.append({
            "key": g["key"], "label": g["label"], "accelerator": g["accelerator"],
            "p95": g["p95"], "throughputPerSec": g["throughputPerSec"],
            "compressedPer1M": cp, "originalPer1M": op,
            "savingsPer1M": round(op - cp, 4) if op is not None else None,
        })
    rep = _representative(groups)
    if rep is not None:
        compressed_per1m = rep["estCostPer1M"]
        original_per1m = _orig(compressed_per1m)
    measured_qps = _measured_qps(session, model.id, groups)

    savings_per1m = (
        round(original_per1m - compressed_per1m, 4)
        if (original_per1m is not None and compressed_per1m is not None)
        else None
    )

    monthly_compressed = monthly_original = monthly_savings = None
    if measured_qps > 0 and compressed_per1m is not None:
        monthly_compressed = monthly_cost(compressed_per1m, measured_qps)
        if original_per1m is not None:
            monthly_original = monthly_cost(original_per1m, measured_qps)
            monthly_savings = round(monthly_original - monthly_compressed, 2)

    projected = False
    proj_compressed = proj_original = proj_savings = None
    if project_qps and project_qps > 0 and compressed_per1m is not None:
        projected = True
        proj_compressed = monthly_cost(compressed_per1m, project_qps)
        if original_per1m is not None:
            proj_original = monthly_cost(original_per1m, project_qps)
            proj_savings = round(proj_original - proj_compressed, 2)

    return {
        "source": "live",
        "compressedPer1M": compressed_per1m or 0.0,
        "originalPer1M": original_per1m,
        "savingsPer1M": savings_per1m,
        "savingsPct": savings_pct,
        "assumedLatencyRatio": ratio,
        "measuredQps": measured_qps,
        "monthlyCompressed": monthly_compressed,
        "monthlyOriginal": monthly_original,
        "monthlySavings": monthly_savings,
        "projected": projected,
        "projectedMonthlyCompressed": proj_compressed,
        "projectedMonthlyOriginal": proj_original,
        "projectedMonthlySavings": proj_savings,
        "perHardware": per_hardware,
    }


def workspace_cost_savings(session: Session, user_id: str) -> dict:
    """Workspace $ rollup across optimized models. Both monthly $ and
    ``avgSavingsPct`` accumulate ONLY from models with real serving traffic, so
    the workspace dashboard never shows benchmark-derived savings before any
    model has actually been deployed and used."""
    metas = user_artifact_metas(session, user_id)
    monthly_compressed = monthly_original = 0.0
    has_live = False
    live_models = 0
    pcts: list[float] = []

    for model, _meta in metas:
        bench = get_cached_result(session, model.id, "benchmark", user_id=user_id)
        ratio = _latency_ratio(bench)
        if ratio is None or not has_any_events(session, model.id):
            continue
        sp = _savings_pct(ratio)
        if sp is not None:
            pcts.append(sp)
        groups = hw.hardware_breakdown(session, model.id, "24h")
        rep = _representative(groups)
        if rep is None:
            continue
        cp = rep["estCostPer1M"]
        if cp <= 0:
            continue
        qps = _measured_qps(session, model.id, groups)
        if qps <= 0:
            continue
        mc = monthly_cost(cp, qps)
        mo = monthly_cost(cp * ratio, qps)
        if mc > 0 or mo > 0:
            monthly_compressed += mc
            monthly_original += mo
            has_live = True
            live_models += 1

    avg_pct = round(sum(pcts) / len(pcts), 1) if pcts else None
    return {
        "hasLiveTraffic": has_live,
        "monthlyCompressed": round(monthly_compressed, 2) if has_live else None,
        "monthlyOriginal": round(monthly_original, 2) if has_live else None,
        "monthlySavings": round(monthly_original - monthly_compressed, 2) if has_live else None,
        "avgSavingsPct": avg_pct,
        "modelCount": len(metas),
        "liveModelCount": live_models,
    }
