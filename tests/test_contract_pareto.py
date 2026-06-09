"""Pareto endpoint — REAL Optuna trials only; honest structured 404s otherwise."""

from __future__ import annotations


def test_real_pareto_shape(client, real_model):
    exp = client.get(f"/api/models/{real_model['modelId']}/pareto").json()
    assert set(exp) == {
        "modelId", "modelName", "experimentId", "status", "iterCurrent",
        "iterTotal", "budget", "baseAccuracy", "trials",
    }
    assert exp["status"] == "completed"
    assert set(exp["budget"]) == {"maxLatency", "maxAccuracyDrop", "maxSize"}
    assert exp["iterCurrent"] <= exp["iterTotal"]
    trials = exp["trials"]
    assert trials, "expected real Optuna trials"
    assert any(t["onFrontier"] for t in trials)
    for t in trials:
        assert 0 <= t["score"] <= 100
        assert t["size"] > 0
        assert t["quant"]


def test_statedict_pareto_is_structured_404(client, statedict_model):
    r = client.get(f"/api/models/{statedict_model['modelId']}/pareto")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "weights_only_checkpoint"
    assert "state_dict" in detail["message"]
    # scene must report the same structured 404
    scene = client.get(f"/api/models/{statedict_model['modelId']}/pareto/scene")
    assert scene.status_code == 404
    assert scene.json()["detail"]["code"] == "weights_only_checkpoint"


def test_failed_model_pareto_is_not_analyzed(client, failed_model):
    r = client.get(f"/api/models/{failed_model['modelId']}/pareto")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "not_analyzed"


def test_pareto_stable_across_reads(client, real_model):
    a = client.get(f"/api/models/{real_model['modelId']}/pareto").text
    b = client.get(f"/api/models/{real_model['modelId']}/pareto").text
    assert a == b


def test_pareto_404_missing_model(client):
    assert client.get("/api/models/m_missing/pareto").status_code == 404
