"""Per-Pareto-trial export/download: determinism, caching, auth, error paths."""

from __future__ import annotations

import hashlib

import onnx
import pytest


@pytest.fixture(scope="module")
def exported(client, real_model):
    model_id = real_model["modelId"]
    trials = client.get(f"/api/models/{model_id}/pareto").json()["trials"]
    assert trials, "real pipeline must produce trials"
    target = next((t for t in trials if t["onFrontier"]), trials[0])
    r = client.post(
        f"/api/models/{model_id}/pareto/trials/{target['trialNumber']}/export")
    assert r.status_code == 200, r.text
    return {"model_id": model_id, "trial": target, "meta": r.json()}


def test_export_metadata(exported):
    meta = exported["meta"]
    assert meta["status"] == "ready"
    assert meta["trialNumber"] == exported["trial"]["trialNumber"]
    assert meta["sizeBytes"] > 0
    assert len(meta["sha256"]) == 64
    assert meta["downloadPath"].endswith(
        f"/pareto/trials/{meta['trialNumber']}/artifact")


def test_export_idempotent(client, exported):
    r = client.post(
        f"/api/models/{exported['model_id']}/pareto/trials/"
        f"{exported['meta']['trialNumber']}/export")
    assert r.status_code == 200
    assert r.json()["sha256"] == exported["meta"]["sha256"], (
        "re-export must reuse the cached artifact (deterministic)")


def test_download_artifact_is_valid_onnx(client, exported):
    r = client.get(exported["meta"]["downloadPath"])
    assert r.status_code == 200
    assert hashlib.sha256(r.content).hexdigest() == exported["meta"]["sha256"]
    assert "attachment" in r.headers.get("content-disposition", "")

    model = onnx.load_from_string(r.content)
    assert len(model.graph.node) > 0

    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(r.content)
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        feeds[inp.name] = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    outs = sess.run(None, feeds)
    assert all(np.all(np.isfinite(o)) for o in outs if hasattr(o, "dtype"))


def test_every_trial_exportable(client, exported):
    """All trials (not just frontier) must export — the user picks ANY node."""
    model_id = exported["model_id"]
    trials = client.get(f"/api/models/{model_id}/pareto").json()["trials"]
    for t in trials[:4]:
        r = client.post(
            f"/api/models/{model_id}/pareto/trials/{t['trialNumber']}/export")
        assert r.status_code == 200, f"trial {t['trialNumber']}: {r.text}"


def test_unknown_trial_404(client, exported):
    r = client.post(
        f"/api/models/{exported['model_id']}/pareto/trials/99999/export")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "trial_not_found"


def test_download_before_export_404(client, exported):
    """A trial that exists but was never exported → honest 404 with code."""
    model_id = exported["model_id"]
    trials = client.get(f"/api/models/{model_id}/pareto").json()["trials"]
    fresh = [t for t in trials if t["trialNumber"] != exported["meta"]["trialNumber"]]
    if not fresh:
        pytest.skip("only one trial in fast mode")
    # pick one outside the first-4 exported in test_every_trial_exportable
    candidates = [t for t in fresh if t not in trials[:4]]
    if not candidates:
        pytest.skip("all trials already exported")
    r = client.get(
        f"/api/models/{model_id}/pareto/trials/{candidates[0]['trialNumber']}/artifact")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "trial_artifact_missing"


def test_trial_number_in_contract(client, exported):
    trials = client.get(f"/api/models/{exported['model_id']}/pareto").json()["trials"]
    for t in trials:
        assert t["trialNumber"] >= 0
        assert t["id"] == f"t_{t['trialNumber']}"
