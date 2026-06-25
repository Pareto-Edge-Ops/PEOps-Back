"""Contract for the public format-capability matrix (/api/meta/format-capabilities).

The matrix is the single source of truth the upload UI and docs read to label,
per format, what the pipeline delivers. It must mirror the worker's real
dispatch (weights-only formats get no Pareto/certificate) and agree with
infer_format for every extension, so the UI can never overclaim.
"""

from __future__ import annotations

import typing

from app.schemas.common import ModelFormat
from app.services.capabilities import FORMAT_CAPABILITIES
from app.services.formats import infer_format

_REQUIRED_KEYS = {
    "format", "extensions", "tier", "pareto", "realLatency",
    "certificate", "taskValidation", "llmCaveat", "noteKey",
}


def test_endpoint_is_public_and_well_shaped(client):
    r = client.get("/api/meta/format-capabilities")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and data
    for row in data:
        assert _REQUIRED_KEYS <= set(row), row


def test_one_entry_per_model_format():
    formats = {c.format for c in FORMAT_CAPABILITIES}
    assert formats == set(typing.get_args(ModelFormat))


def test_weights_only_formats_get_no_guarantee():
    by = {c.format: c for c in FORMAT_CAPABILITIES}
    for name in ("GGUF", "SafeTensors", "CoreML"):
        c = by[name]
        assert c.tier == "weights_only", name
        assert not c.pareto and not c.certificate and not c.realLatency, name


def test_no_format_claims_task_validation():
    # Fidelity is output similarity on synthetic probes — never a task metric
    # (perplexity/accuracy). The matrix must not imply otherwise for ANY format.
    assert all(not c.taskValidation for c in FORMAT_CAPABILITIES)


def test_full_tier_has_pareto_and_certificate():
    by = {c.format: c for c in FORMAT_CAPABILITIES}
    onnx = by["ONNX"]
    assert onnx.tier == "full" and onnx.pareto and onnx.certificate


def test_transformer_capable_formats_flag_the_llm_caveat():
    by = {c.format: c for c in FORMAT_CAPABILITIES}
    for name in ("ONNX", "PyTorch", "SafeTensors", "GGUF"):
        assert by[name].llmCaveat is True, name


def test_infer_format_agrees_with_matrix_for_every_extension():
    for c in FORMAT_CAPABILITIES:
        for ext in c.extensions:
            assert infer_format(f"model{ext}")[0] == c.format, ext
