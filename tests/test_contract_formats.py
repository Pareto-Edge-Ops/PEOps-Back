"""Per-format upload contracts — every container either runs the FULL real
pipeline (convertible to ONNX) or the honest weight-only pipeline. Uses
locally-BUILT real models so the suite stays hermetic."""

from __future__ import annotations

import io

import pytest

from conftest import wait_model_terminal, wait_run


def _upload(client, name: str, data: bytes) -> dict:
    r = client.post(
        "/api/models/upload",
        files={"file": (name, io.BytesIO(data), "application/octet-stream")},
    )
    assert r.status_code == 200
    body = r.json()
    status = wait_run(client, body["modelId"], body["runId"], timeout=300)
    body["status"] = status["status"]
    body["error"] = status.get("error")
    wait_model_terminal(client, body["modelId"])
    return body


@pytest.fixture(scope="module")
def keras_h5(tmp_path_factory) -> bytes:
    """A REAL modern Keras model saved as legacy .h5 (with model_config)."""
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    keras = pytest.importorskip("keras", reason="formats extra (tensorflow/keras) not installed")

    m = keras.Sequential([
        keras.layers.Input((12,)),
        keras.layers.Dense(8, activation="relu", name="hidden"),
        keras.layers.Dense(3, name="logits"),
    ])
    path = tmp_path_factory.mktemp("fmt") / "modern.h5"
    m.save(str(path))
    return path.read_bytes()


def test_keras_h5_runs_full_pipeline(client, keras_h5):
    """Modern .h5 → Keras 3 loads it → tf.function-traced ONNX → FULL pipeline
    with real Optuna trials and a real benchmark."""
    body = _upload(client, "modern-mlp.h5", keras_h5)
    assert body["status"] == "completed", body["error"]
    mid = body["modelId"]

    m = client.get(f"/api/models/{mid}").json()
    assert m["format"] == "TensorFlow"
    assert m["bestAccuracy"] is not None and m["bestAccuracy"] > 0  # measured!
    assert m["weightsOnly"] is False              # executable graph → full guarantee

    par = client.get(f"/api/models/{mid}/pareto")
    assert par.status_code == 200 and par.json()["trials"]
    kpi = client.get(f"/api/models/{mid}/telemetry/kpi")
    assert kpi.status_code == 200 and kpi.json()["p95LatencyMs"]["value"] > 0

    logs = client.get(
        f"/api/models/{mid}/ingestion/{body['runId']}/logs"
    ).json()["logs"]
    joined = "\n".join(e["message"] for e in logs)
    assert "tf.function tracing" in joined        # honest conversion provenance
    assert "Exported real ONNX" in joined


def test_safetensors_weight_only(client):
    import numpy as np

    pytest.importorskip("safetensors", reason="formats extra (safetensors) not installed")
    from safetensors.numpy import save

    data = save({
        "encoder.layer0.weight": np.random.randn(16, 8).astype(np.float32),
        "encoder.layer0.bias": np.random.randn(16).astype(np.float32),
        "head.weight": np.random.randn(4, 16).astype(np.float32),
    })
    body = _upload(client, "tiny.safetensors", data)
    assert body["status"] == "completed", body["error"]
    mid = body["modelId"]

    m = client.get(f"/api/models/{mid}").json()
    assert m["format"] == "SafeTensors"
    assert m["bestAccuracy"] is None              # unmeasurable — never invented
    assert m["weightsOnly"] is True               # no graph → no guarantee chip

    arch = client.get(f"/api/models/{mid}/architecture").json()
    names = {n["name"] for n in arch["nodes"]}
    assert "encoder.layer0" in names and "head" in names   # REAL tensor names
    pr = client.get(f"/api/models/{mid}/pareto")
    assert pr.status_code == 404
    assert pr.json()["detail"]["code"] == "weights_only_checkpoint"

    art = client.get(f"/api/models/{mid}/artifact")
    assert art.status_code == 200                 # real quantized .npz
    assert art.headers["content-disposition"].endswith('_compressed.npz"')


def test_trainer_ckpt_unwraps_state_dict(client, tmp_path):
    """Lightning-style .ckpt {state_dict: {...}} must unwrap and analyze."""
    import torch

    ckpt = {
        "epoch": 3,
        "state_dict": {
            "net.fc1.weight": torch.randn(8, 4),
            "net.fc1.bias": torch.randn(8),
        },
    }
    path = tmp_path / "trainer.ckpt"
    torch.save(ckpt, str(path))
    body = _upload(client, "trainer.ckpt", path.read_bytes())
    assert body["status"] == "completed", body["error"]
    logs = client.get(
        f"/api/models/{body['modelId']}/ingestion/{body['runId']}/logs"
    ).json()["logs"]
    joined = "\n".join(e["message"] for e in logs)
    assert "unwrapped the inner `state_dict`" in joined
    arch = client.get(f"/api/models/{body['modelId']}/architecture").json()
    assert any(n["name"] == "net.fc1" for n in arch["nodes"])


def test_tflite_runs_full_pipeline(client, keras_h5, tmp_path):
    """.tflite converts through tf2onnx's flatbuffer frontend → FULL pipeline."""
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    keras = pytest.importorskip("keras", reason="formats extra (tensorflow/keras) not installed")
    tf = pytest.importorskip("tensorflow", reason="formats extra (tensorflow) not installed")

    h5_path = tmp_path / "m.h5"
    h5_path.write_bytes(keras_h5)
    m = keras.models.load_model(str(h5_path), compile=False)
    tfl = tf.lite.TFLiteConverter.from_keras_model(m).convert()

    body = _upload(client, "converted.tflite", tfl)
    assert body["status"] == "completed", body["error"]
    mid = body["modelId"]
    assert client.get(f"/api/models/{mid}").json()["format"] == "TFLite"
    par = client.get(f"/api/models/{mid}/pareto")
    assert par.status_code == 200 and par.json()["trials"]


def test_weights_only_h5_falls_back_honestly(client, tmp_path):
    """An .h5 with no model_config (pure weight store) → weight-only analysis."""
    h5py = pytest.importorskip("h5py", reason="formats extra (h5py) not installed")
    import numpy as np

    path = tmp_path / "weights-only.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("block1_conv1")
        g.create_dataset("kernel:0", data=np.random.randn(3, 3, 3, 8).astype(np.float32))
        g.create_dataset("bias:0", data=np.random.randn(8).astype(np.float32))
    body = _upload(client, "weights-only.h5", path.read_bytes())
    assert body["status"] == "completed", body["error"]
    mid = body["modelId"]
    assert client.get(f"/api/models/{mid}").json()["bestAccuracy"] is None
    arch = client.get(f"/api/models/{mid}/architecture").json()
    conv = next(n for n in arch["nodes"] if n["name"] == "block1_conv1")
    assert conv["kind"] == "conv"
    assert conv["params"] == 3 * 3 * 3 * 8 + 8     # real shapes
    pr = client.get(f"/api/models/{mid}/pareto")
    assert pr.json()["detail"]["code"] == "weights_only_checkpoint"


def test_transformer_onnx_warns_about_fidelity_scope(client):
    """A graph with an attention pattern (MatMul→Softmax→MatMul) is detected as a
    Transformer; even though it runs the FULL pipeline, the run log must warn
    that the guarantee is probe-fidelity — NOT task accuracy/perplexity."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    wq = numpy_helper.from_array(np.random.randn(8, 8).astype(np.float32), "Wq")
    wv = numpy_helper.from_array(np.random.randn(8, 4).astype(np.float32), "Wv")
    nodes = [
        helper.make_node("MatMul", ["X", "Wq"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
        helper.make_node("MatMul", ["probs", "Wv"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "mini-attn",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
        [wq, wv],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    body = _upload(client, "mini-attn.onnx", model.SerializeToString())
    assert body["status"] == "completed", body["error"]
    mid = body["modelId"]
    assert client.get(f"/api/models/{mid}").json()["weightsOnly"] is False

    logs = client.get(
        f"/api/models/{mid}/ingestion/{body['runId']}/logs"
    ).json()["logs"]
    joined = "\n".join(e["message"] for e in logs)
    assert "Detected architecture: Transformer" in joined
    assert "NOT task accuracy/perplexity" in joined   # the honest LLM caveat
