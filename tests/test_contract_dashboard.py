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
    # nothing optimized yet → zero savings, honestly
    assert s["sizeReduced"]["bytesSaved"] == 0
    assert s["sizeReduced"]["modelCount"] == 0
    assert all(p["value"] == 0 for p in s["sizeReduced"]["spark"])
    # sparks are real daily counts — all zero on an empty DB
    assert all(p["value"] == 0 for p in s["activeRuns"]["spark"])

    cmap = empty_state["compression_map"]
    assert cmap["points"] == []
    assert cmap["modelCount"] == 0
    assert cmap["certifiedCount"] == 0
    assert "best" not in cmap   # excluded when absent

    cov = empty_state["guarantee_coverage"]
    assert cov["totalModels"] == 0
    assert cov["certifiedCount"] == 0
    assert cov["segments"] == []
    assert "avgFidelity" not in cov   # excluded when no fidelity recorded


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
    sr = data["sizeReduced"]
    assert {"bytesSaved", "avgReductionX", "modelCount", "deltaText", "spark"} == set(sr)
    assert len(sr["spark"]) == 16
    assert sr["bytesSaved"] >= 0 and sr["modelCount"] >= 0


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


def test_run_enqueued_shows_queued(client, monkeypatch):
    """A freshly enqueued run is honestly "queued" until a worker picks it up.

    Patch the inline executor to a no-op so the job never runs, then assert the
    dashboard surfaces the run under the "queued" filter (this tab/KPI branch was
    dead before — runs were created "running" the instant they existed). The
    worker normally flips queued → running when it starts processing.
    """
    import app.services.queue as queue_mod

    monkeypatch.setattr(queue_mod, "_run_inline", lambda payload: None)
    body = client.post(
        "/api/models/import", json={"fileName": "queued-probe.onnx"}
    ).json()
    run_id = f"run_{body['runId']}"
    try:
        queued = client.get("/api/dashboard/runs", params={"status": "queued"}).json()
        assert any(r["id"] == run_id for r in queued), [r["id"] for r in queued]
        assert all(r["status"] == "queued" for r in queued)
        # And it must NOT yet appear as running.
        running = client.get("/api/dashboard/runs", params={"status": "running"}).json()
        assert all(r["id"] != run_id for r in running)
    finally:
        # Don't leak this never-executed model into other session-scoped tests.
        client.delete(f"/api/models/{body['modelId']}")


def test_compression_map_real_model(client, real_model):
    cmap = client.get("/api/dashboard/compression-map").json()
    # the real pipeline produced one optimized model with recorded provenance
    assert cmap["modelCount"] >= 1
    assert 0 <= cmap["certifiedCount"] <= cmap["modelCount"]
    for p in cmap["points"]:
        assert {"modelId", "name", "reductionX", "sizeRatio", "accuracyRetained",
                "accuracyDrop", "withinTolerance", "certified"} <= set(p)
        assert p["reductionX"] > 0
        assert 0 <= p["accuracyRetained"] <= 100
        assert isinstance(p["certified"], bool)
    if cmap["points"]:
        # a Pareto pick is plottable → a best (most reduction within tolerance) exists
        assert cmap["best"]["reductionX"] > 0


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


def test_guarantee_coverage_real_model(client, real_model):
    cov = client.get("/api/dashboard/guarantee-coverage").json()
    assert cov["totalModels"] >= 1
    assert 0 <= cov["certifiedCount"] <= cov["totalModels"]
    # rung buckets partition the optimized models exactly
    assert sum(seg["value"] for seg in cov["segments"]) == cov["totalModels"]
    for seg in cov["segments"]:
        assert seg["color"].startswith("#")
        assert seg["value"] >= 1


def test_activity_real_events_only(client, real_model):
    full = client.get("/api/dashboard/activity", params={"limit": 50}).json()
    texts = " | ".join(a["text"] for a in full)
    assert "fixture-model" in texts
    timestamps = [a["timestamp"] for a in full]
    assert timestamps == sorted(timestamps, reverse=True)
    for a in full:
        assert a["kind"] in ACTIVITY_KINDS
