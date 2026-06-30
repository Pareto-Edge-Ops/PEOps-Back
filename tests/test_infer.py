"""Public inference endpoint — API-key auth + real event recording."""

from __future__ import annotations


def test_infer_serves_and_records(make_live_model, deploy_model, client):
    mid = make_live_model("infer-a.onnx")["modelId"]
    dep_id, key = deploy_model(mid)

    # Before any traffic, telemetry is empty (deployed but not yet serving).
    assert client.get(f"/api/models/{mid}/telemetry/meta").json()["source"] == "none"

    r = client.post(
        f"/api/v1/infer/{dep_id}",
        json={"inputs": None},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deploymentId"] == dep_id
    assert body["latencyMs"] >= 0
    assert isinstance(body["outputs"], list) and body["outputs"]
    assert "shape" in body["outputs"][0]

    # The model is now LIVE and the call shows up as real telemetry.
    assert client.get(f"/api/models/{mid}/telemetry/meta").json()["source"] == "live"
    kpi = client.get(f"/api/models/{mid}/telemetry/kpi").json()
    assert kpi["p95LatencyMs"]["value"] > 0  # a real measured latency


def test_infer_auth_failures(make_live_model, deploy_model, client):
    mid = make_live_model("infer-b.onnx")["modelId"]
    dep_id, key = deploy_model(mid)

    # No key / bad key → 401.
    assert client.post(f"/api/v1/infer/{dep_id}", json={"inputs": None}).status_code == 401
    bad = client.post(
        f"/api/v1/infer/{dep_id}", json={"inputs": None},
        headers={"Authorization": "Bearer astra_sk_live_deadbeef"},
    )
    assert bad.status_code == 401

    # Valid key but wrong deployment id → 404 (never leaks existence).
    assert client.post(
        "/api/v1/infer/dep_other", json={"inputs": None},
        headers={"Authorization": f"Bearer {key}"},
    ).status_code == 404


def test_infer_bad_input_records_failure(make_live_model, deploy_model, client):
    mid = make_live_model("infer-c.onnx")["modelId"]
    dep_id, key = deploy_model(mid)
    r = client.post(
        f"/api/v1/infer/{dep_id}",
        json={"inputs": {"__not_a_real_input__": [[1.0]]}},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "bad_input"
    # The failed call is still real telemetry → model is now "live".
    assert client.get(f"/api/models/{mid}/telemetry/meta").json()["source"] == "live"


def test_infer_paused_deployment_409(make_live_model, deploy_model, client):
    mid = make_live_model("infer-d.onnx")["modelId"]
    dep_id, key = deploy_model(mid)
    client.post(f"/api/deployments/{dep_id}/pause")
    r = client.post(
        f"/api/v1/infer/{dep_id}", json={"inputs": None},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "deployment_paused"
