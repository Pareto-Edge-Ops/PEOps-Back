"""Pareto endpoint — REAL Optuna trials only; honest structured 404s otherwise."""

from __future__ import annotations


def test_real_pareto_shape(client, real_model):
    exp = client.get(f"/api/models/{real_model['modelId']}/pareto").json()
    assert set(exp) == {
        "modelId", "modelName", "experimentId", "status", "iterCurrent",
        "iterTotal", "budget", "baseAccuracy", "trials", "servedTrialNumber",
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
    # The default-download trial (when the served artifact IS a Pareto trial)
    # must reference a real trial so the Studio badge points somewhere valid.
    served = exp["servedTrialNumber"]
    if served is not None:
        assert any(t["trialNumber"] == served for t in trials)


def test_served_trial_size_matches_artifact(client, real_model):
    """The Pareto Studio Size and the SDK Hub artifact Size must reconcile:
    same model + same decimal-MB unit. When the served artifact is a Pareto
    trial, its plotted size must equal the real artifact bytes (÷1e6)."""
    model_id = real_model["modelId"]
    exp = client.get(f"/api/models/{model_id}/pareto").json()
    info = client.get(f"/api/models/{model_id}/artifact/info").json()
    prov = info.get("provenance")
    assert prov, "artifact info must carry served-artifact provenance"
    served = exp["servedTrialNumber"]
    artifact_mb = info["sizeBytes"] / 1e6  # decimal MB — same unit as trial.size
    if prov["source"] == "pareto":
        assert served is not None and prov["trialNumber"] == served
        trial = next(t for t in exp["trials"] if t["trialNumber"] == served)
        # Plotted size (ByteSize/1e6) vs real saved file (len/1e6): same model,
        # so they agree within protobuf serialization rounding.
        assert abs(trial["size"] - artifact_mb) <= max(0.01, artifact_mb * 0.02)
    else:
        # Ladder / fallback candidate is NOT a trial — honestly unmarked.
        assert prov["source"] in {"ladder", "fallback"}
        assert served is None


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
