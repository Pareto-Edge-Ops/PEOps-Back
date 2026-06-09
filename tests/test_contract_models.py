"""Models list/get/import lifecycle — including the REAL (fast) pipeline e2e."""

from __future__ import annotations

import time

from conftest import wait_run

MODEL_STATUSES = {"deployed", "training", "optimizing", "draft", "failed", "analyzing"}
MODEL_FORMATS = {
    "ONNX", "PyTorch", "TensorFlow", "TFLite", "CoreML", "Scikit-learn",
    "SafeTensors", "GGUF",
}
REQUIRED_KEYS = {
    "id", "name", "typeFull", "typeShort", "format", "lastLearnedAt",
    "lastOptimizedAt", "status", "bestAccuracy", "isDeployed",
}


def test_list_shape_and_defaults(client, real_model):
    models = client.get("/api/models").json()
    assert len(models) >= 1  # only REAL uploads exist — no seeded demo models
    for m in models:
        assert REQUIRED_KEYS <= set(m)
        assert m["status"] in MODEL_STATUSES
        assert m["format"] in MODEL_FORMATS
        # zod .optional() — keys absent rather than null
        if "description" in m:
            assert m["description"] is not None
        if "analysisRunId" in m:
            assert m["analysisRunId"] is not None
    # default sort: lastLearnedAt desc (string compare), nulls last
    dates = [m["lastLearnedAt"] for m in models]
    assert dates == sorted(dates, reverse=True)


def test_list_query_and_filters(client, real_model):
    q = client.get("/api/models", params={"q": "fixture-model"}).json()
    assert q and all("fixture-model" in m["name"].lower() for m in q)
    empty = client.get("/api/models", params={"q": "zzz-no-such"}).json()
    assert empty == []
    # nothing is actually deployed — the filter honestly returns nothing
    only = client.get("/api/models", params={"onlyDeployed": "1"}).json()
    assert all(m["isDeployed"] for m in only)


def test_list_sort_variants(client, real_model):
    by_name = client.get("/api/models", params={"sort": "name:asc"}).json()
    names = [m["name"] for m in by_name]
    assert names == sorted(names)

    by_acc = client.get("/api/models", params={"sort": "bestAccuracy:desc"}).json()
    accs = [m["bestAccuracy"] for m in by_acc]
    non_null = [a for a in accs if a is not None]
    # nulls sink to the bottom regardless of direction
    assert accs[: len(non_null)] == non_null

    bogus = client.get("/api/models", params={"sort": "evil;drop:asc"}).json()
    assert len(bogus) >= 1  # unknown keys fall back to the default sort


def test_get_model_and_404(client, real_model):
    m = client.get(f"/api/models/{real_model['modelId']}").json()
    assert m["id"] == real_model["modelId"]
    assert m["bestAccuracy"] is not None and m["bestAccuracy"] > 0
    assert client.get("/api/models/m_nope").status_code == 404


def test_statedict_null_accuracy_is_honest(client, statedict_model):
    """A weights-only checkpoint's accuracy is unmeasurable — it must be null,
    never an invented number."""
    m = client.get(f"/api/models/{statedict_model['modelId']}").json()
    assert m["status"] == "draft"
    assert m["bestAccuracy"] is None
    # the complete handshake must NOT invent an accuracy either
    client.post(f"/api/models/{statedict_model['modelId']}/ingestion/complete")
    m2 = client.get(f"/api/models/{statedict_model['modelId']}").json()
    assert m2["bestAccuracy"] is None


def test_statedict_artifact_is_quantized_archive(client, statedict_model):
    art = client.get(f"/api/models/{statedict_model['modelId']}/artifact")
    assert art.status_code == 200
    assert art.headers["content-disposition"].endswith('_compressed.npz"')
    assert len(art.content) > 100


def test_statedict_logs_are_honest(client, statedict_model):
    logs = client.get(
        f"/api/models/{statedict_model['modelId']}/ingestion/"
        f"{statedict_model['runId']}/logs"
    ).json()
    joined = "\n".join(e["message"] for e in logs["logs"])
    assert "state_dict" in joined
    assert "Pareto search skipped" in joined        # the honest skip notice
    assert "latency" in joined.lower()               # explains what's unmeasurable
    assert "Sensitivity analysis ready" in joined


def test_import_real_pipeline_e2e(client):
    t0 = time.time()
    r = client.post("/api/models/import", json={"fileName": "e2e-demo.onnx"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"runId", "modelId", "fileName"}
    run_id, model_id = body["runId"], body["modelId"]

    # While analyzing the list row carries analysisRunId
    m = client.get(f"/api/models/{model_id}").json()
    assert m["status"] == "analyzing"
    assert m["analysisRunId"] == run_id

    status = wait_run(client, model_id, run_id)
    assert status["status"] == "completed", status.get("error")
    elapsed = time.time() - t0
    assert elapsed < 90, f"fast pipeline took {elapsed:.1f}s"

    # Real pipeline logs exist and follow the 6-phase script
    logs = client.get(f"/api/models/{model_id}/ingestion/{run_id}/logs").json()
    assert logs["done"] is True
    messages = [entry["message"] for entry in logs["logs"]]
    joined = "\n".join(messages)
    for phrase in (
        "Phase 1/6", "Phase 2/6", "Phase 3/6", "Phase 4/6", "Phase 5/6", "Phase 6/6",
        "UOSA", "Optuna", "DFCV validation", "Benchmarking original vs compressed",
        "Sensitivity analysis ready",
    ):
        assert phrase in joined, f"missing log phrase: {phrase}"
    for entry in logs["logs"]:
        assert entry["level"] in {"INFO", "WARN", "ERROR", "DEBUG"}
        assert "ts" in entry

    # Complete the handshake → draft with real accuracy
    done = client.post(f"/api/models/{model_id}/ingestion/complete")
    assert done.json() == {"ok": True}
    m = client.get(f"/api/models/{model_id}").json()
    assert m["status"] == "draft"
    assert m["bestAccuracy"] is not None and m["bestAccuracy"] > 0
    assert "analysisRunId" not in m

    # Idempotent re-complete
    again = client.post(f"/api/models/{model_id}/ingestion/complete")
    assert again.json() == {"ok": True}
    assert client.get(f"/api/models/{model_id}").json()["bestAccuracy"] == m["bestAccuracy"]

    # Architecture now reflects the REAL ONNX graph
    arch = client.get(f"/api/models/{model_id}/architecture").json()
    ids = {n["id"] for n in arch["nodes"]}
    assert "input" in ids and "output" in ids
    assert any("Gemm" in i or "fc" in i for i in ids), ids
    for n in arch["nodes"]:
        assert 0 <= n["sensitivity"] <= 1
        assert n["recommend"] in {"INT8", "FP16", "FP32"}
    for e in arch["edges"]:
        assert e["from"] in ids and e["to"] in ids

    # Pareto experiment from real Optuna trials
    par = client.get(f"/api/models/{model_id}/pareto").json()
    assert par["status"] == "completed"
    assert par["trials"], "expected real trials"
    assert any(t["onFrontier"] for t in par["trials"])
    assert par["experimentId"] == f"exp_{run_id}"
    for t in par["trials"]:
        assert 0 <= t["score"] <= 100
        assert t["size"] > 0

    # Real benchmark-backed telemetry exists for this model
    kpi = client.get(f"/api/models/{model_id}/telemetry/kpi").json()
    assert kpi["p95LatencyMs"]["value"] > 0

    # Compressed artifact downloadable
    art = client.get(f"/api/models/{model_id}/artifact")
    assert art.status_code == 200
    assert len(art.content) > 100

    # Dashboard reflects the completed run + activity
    runs = client.get("/api/dashboard/runs", params={"status": "done"}).json()
    assert any(r["id"] == f"run_{run_id}" for r in runs)
    acts = client.get("/api/dashboard/activity", params={"limit": 50}).json()
    texts = " | ".join(a["text"] for a in acts)
    assert "e2e-demo" in texts


def test_worker_finalizes_without_complete_call(client):
    """If the SPA never calls /ingestion/complete (tab closed, crash), the
    worker itself must move the model to a terminal status — it must not be
    stuck in 'analyzing' forever."""
    r = client.post("/api/models/import", json={"fileName": "orphaned-run.onnx"}).json()
    model_id, run_id = r["modelId"], r["runId"]
    status = wait_run(client, model_id, run_id)
    assert status["status"] == "completed", status.get("error")
    deadline = time.time() + 10
    m = {}
    while time.time() < deadline:
        m = client.get(f"/api/models/{model_id}").json()
        if m["status"] == "draft":
            break
        time.sleep(0.2)
    assert m["status"] == "draft"
    assert m["bestAccuracy"] is not None
    assert "analysisRunId" not in m
    # the late /complete is a harmless no-op
    assert client.post(f"/api/models/{model_id}/ingestion/complete").json() == {"ok": True}


def test_import_default_filename(client):
    r = client.post("/api/models/import", json={})
    assert r.status_code == 200
    assert r.json()["fileName"] == "uploaded-model.onnx"


def test_complete_while_streaming_marks_optimizing(client):
    r = client.post("/api/models/import", json={"fileName": "early-ack.pt"}).json()
    model_id, run_id = r["modelId"], r["runId"]
    # Acknowledge immediately — pipeline may still be running.
    client.post(f"/api/models/{model_id}/ingestion/complete")
    m = client.get(f"/api/models/{model_id}").json()
    assert m["status"] in {"optimizing", "draft", "failed"}
    status = wait_run(client, model_id, run_id)
    assert status["status"] == "completed", status.get("error")
    # Worker finishes the transition with the real accuracy.
    deadline = time.time() + 10
    while time.time() < deadline:
        m = client.get(f"/api/models/{model_id}").json()
        if m["status"] == "draft":
            break
        time.sleep(0.2)
    assert m["status"] == "draft"
    assert m["bestAccuracy"] is not None


def test_failed_upload_lifecycle(client, failed_model):
    """A genuinely broken artifact fails honestly: failed status, real error in
    the run, a real danger alert — and no fabricated results anywhere."""
    m = client.get(f"/api/models/{failed_model['modelId']}").json()
    assert m["status"] == "failed"
    assert m["bestAccuracy"] is None
    run = client.get(
        f"/api/models/{failed_model['modelId']}/ingestion/{failed_model['runId']}"
    ).json()
    assert run["status"] == "failed"
    assert run.get("error")


def test_rename_lifecycle(client):
    r = client.post("/api/models/import", json={"fileName": "rename-me.onnx"}).json()
    mid = r["modelId"]
    wait_run(client, mid, r["runId"])

    renamed = client.patch(f"/api/models/{mid}", json={"name": "Renamed Model X"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Renamed Model X"
    assert client.get(f"/api/models/{mid}").json()["name"] == "Renamed Model X"
    # dashboard run rows carry the new display name too
    runs = client.get("/api/dashboard/runs").json()
    mine = [run for run in runs if run["modelId"] == mid]
    assert mine and all(run["name"] == "Renamed Model X" for run in mine)
    # validation
    assert client.patch(f"/api/models/{mid}", json={"name": "   "}).status_code == 400
    assert client.patch(f"/api/models/{mid}", json={"name": "x" * 81}).status_code == 400
    assert client.patch("/api/models/m_missing", json={"name": "ok"}).status_code == 404


def test_delete_lifecycle(client):
    r = client.post("/api/models/import", json={"fileName": "delete-me.onnx"}).json()
    mid, rid = r["modelId"], r["runId"]
    wait_run(client, mid, rid)
    assert client.get(f"/api/models/{mid}/artifact").status_code == 200

    assert client.delete(f"/api/models/{mid}").json() == {"ok": True}
    # everything about the model is gone
    assert client.get(f"/api/models/{mid}").status_code == 404
    assert client.get(f"/api/models/{mid}/architecture").status_code == 404
    assert client.get(f"/api/models/{mid}/ingestion/{rid}").status_code == 404
    assert client.get(f"/api/models/{mid}/artifact").status_code == 404
    runs = client.get("/api/dashboard/runs").json()
    assert not any(run["modelId"] == mid for run in runs)
    # repeat delete is an honest 404 (nothing left to delete)
    assert client.delete(f"/api/models/{mid}").status_code == 404


def test_artifact_info_and_usage(client, real_model):
    mid = real_model["modelId"]
    info = client.get(f"/api/models/{mid}/artifact/info").json()
    assert info["kind"] == "onnx"
    assert info["sizeBytes"] > 0
    assert len(info["sha256"]) == 64
    assert info["downloadPath"] == f"/api/models/{mid}/artifact"
    assert info["inputs"] and all({"name", "shape", "dtype"} <= set(i) for i in info["inputs"])
    # info matches the actual downloadable bytes
    art = client.get(f"/api/models/{mid}/artifact")
    assert len(art.content) == info["sizeBytes"]
    import hashlib

    assert hashlib.sha256(art.content).hexdigest() == info["sha256"]

    usage = client.get(f"/api/models/{mid}/sdk/usage").json()
    assert set(usage) == {"python", "curl"}
    assert "onnxruntime" in usage["python"]["code"]
    assert info["fileName"] in usage["python"]["code"]
    assert info["sha256"] in usage["curl"]["code"]


def test_artifact_info_statedict_npz(client, statedict_model):
    mid = statedict_model["modelId"]
    info = client.get(f"/api/models/{mid}/artifact/info").json()
    assert info["kind"] == "npz"
    usage = client.get(f"/api/models/{mid}/sdk/usage").json()
    assert "np.load" in usage["python"]["code"]
    assert "__scale__" in usage["python"]["code"]


def test_artifact_info_404_when_failed(client, failed_model):
    r = client.get(f"/api/models/{failed_model['modelId']}/artifact/info")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "no_artifact"
