"""Cost & savings lens — compression translated into dollars.

Pure-math unit checks for the cost formulas, plus end-to-end checks over the
fleet simulator (live path) and the empty (no-traffic) path, asserting the
honesty rules: a monthly $ is only asserted from measured QPS, the original cost
is the disclosed counterfactual (compressed × latency ratio), and an unserved
model exposes no cost numbers at all (no benchmark fallback).
"""

from __future__ import annotations

from app.services import cost


def _simulate_fleet(client, mid: str) -> dict:
    r = client.post(
        f"/api/models/{mid}/telemetry/simulate",
        json={"count": 480, "hours": 6, "incidents": False, "fleet": True},
    )
    assert r.status_code == 200, r.text
    return r.json()


# ── pure math ─────────────────────────────────────────────────────────────────


def test_monthly_cost_formula():
    # $0.73/1M at 100 req/s for a 30-day month.
    assert cost.monthly_cost(0.73, 100) == round(0.73 * 100 * cost.SEC_PER_MONTH / 1e6, 2)
    assert cost.monthly_cost(0.73, 0) == 0.0     # no traffic → no cost
    assert cost.monthly_cost(0.0, 100) == 0.0    # free hardware → no cost


def test_latency_ratio_and_savings_pct():
    assert cost._latency_ratio({"original": {"p95": 40}, "compressed": {"p95": 10}}) == 4.0
    assert cost._latency_ratio({"original": {"p95": 0}, "compressed": {"p95": 10}}) is None
    assert cost._latency_ratio(None) is None
    assert cost._latency_ratio({"compressed": {"p95": 10}}) is None
    # 4× faster ⇒ 75% cheaper on equal hardware.
    assert cost._savings_pct(4.0) == 75.0
    assert cost._savings_pct(None) is None
    assert cost._savings_pct(0.0) is None


# ── live path (fleet simulator) ───────────────────────────────────────────────


def test_model_cost_live_compressed_vs_original(make_live_model, deploy_model, client):
    mid = make_live_model("cost-live.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)

    c = client.get(f"/api/models/{mid}/telemetry/cost").json()
    assert c["source"] == "live"
    assert c["perHardware"], "live path must expose per-hardware cost rows"
    assert c["compressedPer1M"] > 0
    assert c["measuredQps"] > 0

    # The accelerator serves the same artifact cheaper than the CPU.
    gpu = [r["compressedPer1M"] for r in c["perHardware"] if r["accelerator"] == "gpu"]
    cpu = [r["compressedPer1M"] for r in c["perHardware"] if r["accelerator"] in ("cpu", "hosted")]
    assert gpu and cpu
    assert min(gpu) < max(cpu)

    # Counterfactual + savings math is internally consistent (any ratio sign).
    ratio = c["assumedLatencyRatio"]
    if ratio:
        assert c["originalPer1M"] == round(c["compressedPer1M"] * ratio, 4)
        assert c["savingsPct"] == round(100.0 * (1 - 1 / ratio), 1)
    # Monthly is asserted (measured QPS exists) and reconciles.
    assert c["monthlyCompressed"] is not None
    if c["monthlySavings"] is not None:
        assert c["monthlySavings"] == round(c["monthlyOriginal"] - c["monthlyCompressed"], 2)


def test_workspace_cost_savings_live(make_live_model, deploy_model, client):
    mid = make_live_model("cost-ws.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)

    ws = client.get("/api/dashboard/cost-savings").json()
    assert ws["hasLiveTraffic"] is True
    assert ws["liveModelCount"] >= 1
    assert ws["modelCount"] >= 1
    assert ws["monthlySavings"] == round(ws["monthlyOriginal"] - ws["monthlyCompressed"], 2)


# ── empty (no traffic) ────────────────────────────────────────────────────────


def test_model_cost_empty_until_traffic(real_model, client):
    """A never-served model exposes NO cost numbers — no benchmark reference row,
    no $/1M, no monthly figure (honesty — nothing has been measured)."""
    mid = real_model["modelId"]
    c = client.get(f"/api/models/{mid}/telemetry/cost").json()
    assert c["source"] == "none"
    assert c["perHardware"] == []
    assert c["compressedPer1M"] == 0.0
    assert c["measuredQps"] == 0.0
    assert c["monthlyCompressed"] is None
    assert c["savingsPct"] is None


def test_model_cost_projection_live(make_live_model, deploy_model, client):
    """A target QPS projects a monthly figure (labeled projected) on the LIVE
    path — projection from the benchmark no longer exists."""
    mid = make_live_model("cost-proj.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)
    c = client.get(f"/api/models/{mid}/telemetry/cost?projectQps=200").json()
    assert c["projected"] is True
    assert c["projectedMonthlyCompressed"] is not None


def test_model_cost_weights_only_404(statedict_model, client):
    """A weights-only checkpoint can't be executed → structured 404, same gate as
    the sibling telemetry endpoints."""
    mid = statedict_model["modelId"]
    r = client.get(f"/api/models/{mid}/telemetry/cost")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "weights_only_checkpoint"


# ── compression-map cost annotation ───────────────────────────────────────────


def test_compression_map_carries_cost_chip(real_model, client):
    cmap = client.get("/api/dashboard/compression-map").json()
    priced = [p for p in cmap["points"] if p.get("estCostPer1M") is not None]
    # Pareto points carry a latency → a $/1M chip; assert it's a sane positive.
    for p in priced:
        assert p["estCostPer1M"] > 0
