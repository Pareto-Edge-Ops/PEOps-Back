"""Dashboard endpoints — real DB aggregates + honest zero state."""

from __future__ import annotations

RUN_STATUSES = {"running", "queued", "done", "failed"}
ACTIVITY_KINDS = {
    "run_started", "run_completed", "deploy_promoted", "accuracy_drift", "model_uploaded",
}


# ── zero state (captured before any model existed) ──────────────────────────

def test_empty_db_is_honestly_empty(empty_state):
    assert empty_state["models"] == []
    assert empty_state["runs"] == []
    assert empty_state["top_models"] == []
    assert empty_state["activity"] == []

    s = empty_state["summary"]
    assert s["activeRuns"]["value"] == 0
    assert s["completedThisWeek"]["value"] == 0
    assert s["liveDeployments"]["value"] == 0
    assert s["computeUsed"]["used"] == 0
    # sparks are real daily counts — all zero on an empty DB
    assert all(p["value"] == 0 for p in s["activeRuns"]["spark"])

    snap = empty_state["snapshot_resp"]
    assert snap["status"] == 404
    assert snap["body"]["detail"]["code"] == "no_completed_runs"

    cost = empty_state["compute_cost"]
    assert cost["usedGpuHours"] == 0
    assert cost["segments"] == []
    # No fabricated billing — keys must be absent entirely.
    for fake in ("costUsd", "region", "resetDateText", "noteText"):
        assert fake not in cost


# ── populated state (after the real fast-pipeline fixture ran) ──────────────

def test_summary_shape_and_real_counts(client, real_model):
    data = client.get("/api/dashboard/summary").json()
    for key in ("activeRuns", "completedThisWeek", "liveDeployments"):
        block = data[key]
        assert isinstance(block["value"], (int, float))
        assert isinstance(block["deltaText"], str)
        assert len(block["spark"]) == 16
        assert {"t", "value"} <= set(block["spark"][0])
    assert data["completedThisWeek"]["value"] >= 1   # the real fixture run
    # the run completed today — today's spark bucket counts it
    assert data["completedThisWeek"]["spark"][-1]["value"] >= 1
    cu = data["computeUsed"]
    assert {"used", "quota", "label", "progressNote"} == set(cu)
    assert "compute·h" in cu["label"]                # no GPU·h fiction
    assert cu["used"] > 0                            # real measured duration


def test_runs_reflect_real_pipeline(client, real_model):
    all_runs = client.get("/api/dashboard/runs").json()
    assert any(r["id"] == f"run_{real_model['runId']}" for r in all_runs)
    for run in all_runs:
        assert set(run) == {
            "id", "modelId", "name", "status", "progressPct", "iter", "bestAcc", "deltaAcc",
        }
        assert run["status"] in RUN_STATUSES
    done = client.get("/api/dashboard/runs", params={"status": "done"}).json()
    assert all(r["status"] == "done" for r in done)


def test_pareto_snapshot_uses_latest_real_model(client, real_model):
    snap = client.get("/api/dashboard/pareto-snapshot").json()
    assert snap["modelId"] == real_model["modelId"]
    assert "pareto" in snap["modelName"]
    assert snap["points"]
    assert any(p["onFrontier"] for p in snap["points"])
    for p in snap["points"]:
        assert set(p) == {"id", "accuracy", "latency", "size", "onFrontier"}
    # real best accuracy of the snapshot model is exposed (optional field)
    assert "bestAccuracy" in snap and snap["bestAccuracy"] > 0


def test_top_models_real_coverage_and_spark(client, real_model):
    top = client.get("/api/dashboard/top-models").json()
    assert any(m["modelId"] == real_model["modelId"] for m in top)
    assert [m["rank"] for m in top] == list(range(1, len(top) + 1))
    accs = [m["bestAccuracy"] for m in top]
    assert accs == sorted(accs, reverse=True)
    mine = next(m for m in top if m["modelId"] == real_model["modelId"])
    # coverage = real frontier share; spark = real trial accuracy progression
    pareto = client.get(f"/api/models/{real_model['modelId']}/pareto").json()
    trials = pareto["trials"]
    expected_cov = round(100 * sum(1 for t in trials if t["onFrontier"]) / len(trials), 1)
    assert mine["paretoCoverage"] == expected_cov
    assert mine["spark"] == [round(float(t["accuracy"]), 2) for t in trials[:16]]


def test_compute_cost_real_phases(client, real_model):
    cost = client.get("/api/dashboard/compute-cost").json()
    assert cost["usedGpuHours"] > 0
    assert cost["segments"], "expected real per-phase timings"
    labels = {seg["label"] for seg in cost["segments"]}
    assert "Benchmark" in labels
    for seg in cost["segments"]:
        assert seg["color"].startswith("#")
        assert seg["value"] >= 0
    for fake in ("costUsd", "region", "resetDateText", "noteText"):
        assert fake not in cost


def test_activity_real_events_only(client, real_model):
    full = client.get("/api/dashboard/activity", params={"limit": 50}).json()
    texts = " | ".join(a["text"] for a in full)
    assert "fixture-model" in texts
    timestamps = [a["timestamp"] for a in full]
    assert timestamps == sorted(timestamps, reverse=True)
    for a in full:
        assert a["kind"] in ACTIVITY_KINDS
