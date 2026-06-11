"""The served artifact must be REALLY compressed and certified.

Contract: after the (fast) pipeline completes, the exported artifact's bytes
are genuinely smaller than the ingested original whenever any non-FP32
precision was selected, the ingestion log carries the guarantee certificate,
and the per-trial reconstruction prerequisites (ingested ONNX + trial config
cache) exist.
"""

from __future__ import annotations

import onnx
import pytest


def _log_text(client, model_id: str, run_id: str) -> str:
    r = client.get(f"/api/models/{model_id}/ingestion/{run_id}/logs")
    if r.status_code == 404:  # poll endpoint variant
        r = client.get(f"/api/models/{model_id}/ingestion/{run_id}")
    body = r.json()
    logs = body.get("logs") or body.get("lines") or []
    return "\n".join(
        entry.get("message", "") if isinstance(entry, dict) else str(entry)
        for entry in logs
    )


@pytest.fixture(scope="module")
def pipeline_ctx(client, real_model):
    import json

    from sqlmodel import select

    from app.db import open_session
    from app.dbmodels import ResultCacheRow
    from app.services.storage import get_storage, ingested_key

    def cached(s, model_id: str, kind: str):
        row = s.exec(
            select(ResultCacheRow).where(
                ResultCacheRow.model_id == model_id, ResultCacheRow.kind == kind)
        ).first()
        return json.loads(row.payload) if row else None

    model_id = real_model["modelId"]
    with open_session() as s:
        configs = cached(s, model_id, "pareto_configs")
        arch = cached(s, model_id, "architecture")
    return {
        "model_id": model_id,
        "run_id": real_model["runId"],
        "configs": configs,
        "architecture": arch,
        "storage": get_storage(),
        "ingested_key": ingested_key(model_id),
    }


def test_artifact_really_smaller(client, pipeline_ctx, tmp_path):
    """Download both the ingested original and the served artifact; whenever a
    quantized precision was selected the artifact must be at least 10% smaller."""
    storage = pipeline_ctx["storage"]
    model_id = pipeline_ctx["model_id"]

    assert storage.exists(pipeline_ctx["ingested_key"]), "ingested ONNX must be persisted"
    ingested_path = tmp_path / "ingested.onnx"
    storage.download_to(pipeline_ctx["ingested_key"], str(ingested_path))

    r = client.get(f"/api/models/{model_id}/artifact")
    assert r.status_code == 200
    artifact_bytes = r.content

    original = onnx.load(str(ingested_path))
    compressed = onnx.load_from_string(artifact_bytes)

    arch = pipeline_ctx["architecture"]
    selected = {
        n.get("recommend") for n in arch.get("nodes", [])
        if isinstance(n, dict)
    }
    if selected & {"FP16", "INT8", "INT4"}:
        assert compressed.ByteSize() < 0.9 * original.ByteSize(), (
            f"quantized precisions selected {selected} but artifact is "
            f"{compressed.ByteSize()}B vs original {original.ByteSize()}B"
        )


def test_artifact_runs_in_ort(client, pipeline_ctx):
    import numpy as np
    import onnxruntime as ort

    r = client.get(f"/api/models/{pipeline_ctx['model_id']}/artifact")
    sess = ort.InferenceSession(r.content)
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        feeds[inp.name] = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    outs = sess.run(None, feeds)
    assert all(np.all(np.isfinite(o)) for o in outs if hasattr(o, "dtype"))


def test_guarantee_certificate_in_log(client, pipeline_ctx):
    text = _log_text(client, pipeline_ctx["model_id"], pipeline_ctx["run_id"])
    assert "Guarantee Certificate" in text
    assert "Fidelity floor (tau)" in text
    assert "Weights-only:" in text


def test_trial_configs_cached(pipeline_ctx):
    configs = pipeline_ctx["configs"]
    assert configs is not None, "pareto_configs cache must exist after the pipeline"
    assert configs.get("schema") == 1
    trials = configs.get("trials") or {}
    assert len(trials) >= 1
    sample = next(iter(trials.values()))
    assert isinstance(sample, dict)
    for op_cfg in sample.values():
        assert "precision" in op_cfg


def test_quality_not_regressed(client, pipeline_ctx):
    m = client.get(f"/api/models/{pipeline_ctx['model_id']}").json()
    assert m.get("bestAccuracy", 0) >= 85.0, (
        "fast-pipeline fixture model should retain a high DFCV quality score"
    )
