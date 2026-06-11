"""Architecture endpoint — REAL pipeline + weights-only checkpoints only."""

from __future__ import annotations

LAYER_KINDS = {
    "input", "conv", "bn", "relu", "pool", "dense", "attn", "ffn",
    "norm", "output", "embed", "lstm", "softmax", "upsample",
}
NODE_REQUIRED = {
    "id", "name", "kind", "depth", "col", "sensitivity", "params", "recommend",
}
NODE_OPTIONAL = (
    "zCol", "width", "latencyMs",
    "opType", "category", "inputShape", "outputShape",
    "flops", "units", "precisionSource", "description",
)


def _validate_arch_shape(arch: dict) -> None:
    kinds = {n["kind"] for n in arch["nodes"]}
    assert kinds <= LAYER_KINDS
    ids = {n["id"] for n in arch["nodes"]}
    assert len(ids) == len(arch["nodes"])  # unique ids
    for n in arch["nodes"]:
        assert NODE_REQUIRED <= set(n)
        assert 0 <= n["sensitivity"] <= 1
        assert n["recommend"] in {"INT8", "FP16", "FP32"}
        # zod .optional() — keys absent rather than null
        for opt in NODE_OPTIONAL:
            if opt in n:
                assert n[opt] is not None
    for e in arch["edges"]:
        assert set(e) == {"from", "to"}
        assert e["from"] in ids and e["to"] in ids


def test_real_model_architecture(client, real_model):
    arch = client.get(f"/api/models/{real_model['modelId']}/architecture").json()
    assert arch["modelId"] == real_model["modelId"]
    _validate_arch_shape(arch)
    # Real ONNX graphs carry measured latency on their nodes.
    op_nodes = [n for n in arch["nodes"] if n["kind"] not in ("input", "output")]
    assert op_nodes
    assert all("latencyMs" in n for n in op_nodes)


def test_real_model_per_op_metadata(client, real_model):
    """Every op node must carry REAL ONNX-derived facts: op_type, category,
    flops, a bilingual description, and (when shapes are static) tensor shapes
    + the raw units count the rendered width stylizes."""
    arch = client.get(f"/api/models/{real_model['modelId']}/architecture").json()
    op_nodes = [n for n in arch["nodes"] if n["kind"] not in ("input", "output")]
    assert op_nodes
    for n in op_nodes:
        assert n["opType"], n["id"]                      # real op_type, nonempty
        assert isinstance(n["category"], str) and n["category"]
        assert isinstance(n["flops"], int) and n["flops"] >= 0
        assert n["precisionSource"] in ("pareto", "recommended")
        desc = n["description"]
        assert desc["title"], n["id"]
        assert desc["summary"]["en"] and desc["summary"]["ko"], n["id"]
        assert "formula" in desc  # str | null, always present inside description
        for key in ("inputShape", "outputShape"):
            if key in n:  # absent when dynamic/unknown — never invented
                assert isinstance(n[key], list) and n[key]
                assert all(isinstance(d, int) and d > 0 for d in n[key]), n["id"]
        if "units" in n:
            assert isinstance(n["units"], int) and n["units"] > 0
        assert isinstance(n["params"], (int, float)) and n["params"] >= 0
    # The mapped op_types must be real ONNX operator names, not kind enums.
    assert any(n["opType"] not in LAYER_KINDS for n in op_nodes)
    # Synthetic input/output endpoints carry no fabricated op metadata.
    for n in arch["nodes"]:
        if n["kind"] in ("input", "output"):
            for key in ("opType", "category", "flops", "precisionSource", "description"):
                assert key not in n, (n["id"], key)


def test_statedict_architecture_is_real_inventory(client, statedict_model):
    arch = client.get(f"/api/models/{statedict_model['modelId']}/architecture").json()
    _validate_arch_shape(arch)
    by_name = {n["name"]: n for n in arch["nodes"]}
    # Real layers recovered from the checkpoint's tensors (module. stripped).
    assert by_name["backbone.0"]["kind"] == "conv"
    assert by_name["backbone.1"]["kind"] == "bn"
    assert by_name["embed"]["kind"] == "embed"
    assert by_name["head"]["kind"] == "dense"
    # Real param counts from the actual shapes: conv 8*3*3*3 + 8 bias = 224.
    assert by_name["backbone.0"]["params"] == 8 * 3 * 3 * 3 + 8
    # latencyMs is unmeasurable for a state_dict — key must be ABSENT, not 0.
    assert all("latencyMs" not in n for n in arch["nodes"])
    # No executable graph → no ONNX ops: per-op metadata must be ABSENT, never
    # fabricated from tensor names.
    for n in arch["nodes"]:
        for key in ("opType", "category", "flops", "inputShape", "outputShape",
                    "units", "precisionSource", "description"):
            assert key not in n, (n["name"], key)
    # Linear chain in registration order (no topology exists in a state_dict).
    ids = [n["id"] for n in arch["nodes"]]
    assert arch["edges"] == [{"from": a, "to": b} for a, b in zip(ids, ids[1:])]


def test_statedict_scene_renders(client, statedict_model):
    scene = client.get(
        f"/api/models/{statedict_model['modelId']}/architecture/scene"
    ).json()
    assert scene["counts"]["neurons"] > 0
    assert scene["camera"]["position"]


def test_architecture_404_missing_model(client):
    assert client.get("/api/models/m_missing/architecture").status_code == 404


def test_architecture_404_not_analyzed(client, failed_model):
    r = client.get(f"/api/models/{failed_model['modelId']}/architecture")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "not_analyzed"


def test_architecture_stable_across_reads(client, real_model):
    a = client.get(f"/api/models/{real_model['modelId']}/architecture").text
    b = client.get(f"/api/models/{real_model['modelId']}/architecture").text
    assert a == b  # served from the cached real result
