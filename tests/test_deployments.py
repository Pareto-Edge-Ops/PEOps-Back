"""Deployment lifecycle — create / list / pause / resume / rotate / delete."""

from __future__ import annotations


def test_deploy_mints_endpoint_and_key(make_live_model, deploy_model, client):
    model = make_live_model("deploy-a.onnx")
    mid = model["modelId"]

    r = client.post(f"/api/models/{mid}/deployments", json={"region": "ap-northeast-2"})
    assert r.status_code == 200, r.text
    data = r.json()
    dep, key = data["deployment"], data["apiKey"]
    assert dep["id"].startswith("dep_")
    assert dep["endpoint"].endswith(f"/api/v1/infer/{dep['id']}")
    assert dep["status"] == "live"
    # Plaintext key shown once; only a masked prefix is persisted/listed.
    assert key.startswith("peops_sk_live_")
    assert "…" in dep["keyPrefix"] and key not in dep["keyPrefix"]

    # The model flips to deployed.
    m = client.get(f"/api/models/{mid}").json()
    assert m["isDeployed"] is True

    # It shows up in the management list and the telemetry deployments widget.
    listed = client.get(f"/api/models/{mid}/deployments").json()
    assert any(d["id"] == dep["id"] for d in listed)
    tele = client.get(f"/api/models/{mid}/telemetry/deployments").json()
    assert any(d["endpoint"] == dep["endpoint"] for d in tele)


def test_pause_resume_and_rotate(make_live_model, deploy_model, client):
    mid = make_live_model("deploy-b.onnx")["modelId"]
    dep_id, key = deploy_model(mid)

    paused = client.post(f"/api/deployments/{dep_id}/pause").json()
    assert paused["status"] == "paused"
    resumed = client.post(f"/api/deployments/{dep_id}/resume").json()
    assert resumed["status"] == "live"

    rot = client.post(f"/api/deployments/{dep_id}/rotate-key").json()
    assert rot["apiKey"].startswith("peops_sk_live_")
    assert rot["apiKey"] != key  # a genuinely new secret


def test_delete_clears_deployed_flag(make_live_model, deploy_model, client):
    mid = make_live_model("deploy-c.onnx")["modelId"]
    dep_id, _ = deploy_model(mid)

    assert client.delete(f"/api/deployments/{dep_id}").status_code == 200
    assert client.get(f"/api/models/{mid}/deployments").json() == []
    m = client.get(f"/api/models/{mid}").json()
    assert m["isDeployed"] is False


def test_deploy_unknown_model_404(client):
    assert client.post("/api/models/m_missing/deployments", json={}).status_code == 404


def test_manage_unknown_deployment_404(client):
    assert client.post("/api/deployments/dep_missing/pause").status_code == 404
    assert client.delete("/api/deployments/dep_missing").status_code == 404
