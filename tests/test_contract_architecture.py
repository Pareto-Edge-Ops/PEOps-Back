"""Architecture endpoint — REAL pipeline + weights-only checkpoints only."""

from __future__ import annotations

LAYER_KINDS = {
    "input", "conv", "bn", "relu", "pool", "dense", "attn", "ffn",
    "norm", "output", "embed", "lstm", "softmax", "upsample",
}
NODE_REQUIRED = {
    "id", "name", "kind", "depth", "col", "sensitivity", "params", "recommend",
}


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
        for opt in ("zCol", "width", "latencyMs"):
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
