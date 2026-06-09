"""Weight extractors for checkpoint formats that cannot be executed.

Each loader returns a `WeightBundle`: REAL tensors (numpy) + optionally the
REAL layer topology when the container actually stores one (Keras h5
`model_config`, CoreML NN spec). Nothing is guessed — when a container has no
graph, `layers` stays None and the pipeline says so.

All imports are lazy; a missing optional reader raises `WeightLoadError` with
an honest, actionable message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class WeightLoadError(Exception):
    """User-facing reason why this file's weights cannot be read."""


@dataclass
class LayerSpec:
    """One REAL layer recovered from the container's own metadata."""

    name: str
    kind: str                       # frontend LayerKind
    params: int
    inputs: list[str] = field(default_factory=list)   # real upstream layer names
    tensor_keys: list[str] = field(default_factory=list)


@dataclass
class WeightBundle:
    tensors: dict          # name -> np.ndarray (REAL weights; may be empty)
    framework: str         # honest provenance label for modelType
    layers: list[LayerSpec] | None = None   # REAL topology when the file has one
    notes: list[str] = field(default_factory=list)
    quantized_already: bool = False         # e.g. GGUF Q4 — re-quantizing is meaningless


# ── torch checkpoints (.pt/.pth/.bin/.ckpt) ──────────────────────────────────

def load_torch_state_dict(source_path: str) -> WeightBundle | None:
    """Return a bundle when the file is a raw state_dict; None when it's a
    full pickled module (those take the executable pipeline)."""
    import torch

    try:
        obj = torch.load(source_path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    notes: list[str] = []
    # Lightning/trainer checkpoints wrap the weights under "state_dict".
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        notes.append("Trainer checkpoint detected — unwrapped the inner `state_dict`.")
        obj = obj["state_dict"]
    if not (isinstance(obj, dict) and obj
            and all(isinstance(v, torch.Tensor) for v in obj.values())):
        return None
    tensors = {k: v.detach().numpy() for k, v in obj.items()}
    return WeightBundle(tensors=tensors, framework="PyTorch state_dict (weights-only)",
                        notes=notes)


# ── safetensors ──────────────────────────────────────────────────────────────

def load_safetensors(source_path: str) -> WeightBundle:
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise WeightLoadError(f"`safetensors` reader not installed: {exc}") from exc
    try:
        tensors = load_file(source_path)
    except Exception as exc:
        raise WeightLoadError(
            f"Could not parse safetensors file ({type(exc).__name__}: {str(exc)[:120]})"
        ) from exc
    return WeightBundle(tensors=dict(tensors), framework="SafeTensors (weights-only)")


# ── Keras HDF5 (.h5) — weights + (when present) the REAL architecture ───────

_KERAS_KIND = {
    "conv1d": "conv", "conv2d": "conv", "conv3d": "conv",
    "separableconv1d": "conv", "separableconv2d": "conv",
    "depthwiseconv2d": "conv", "conv2dtranspose": "upsample",
    "dense": "dense", "embedding": "embed",
    "batchnormalization": "bn", "layernormalization": "norm",
    "groupnormalization": "norm",
    "lstm": "lstm", "gru": "lstm", "simplernn": "lstm", "bidirectional": "lstm",
    "multiheadattention": "attn", "attention": "attn",
    "maxpooling1d": "pool", "maxpooling2d": "pool", "maxpooling3d": "pool",
    "averagepooling1d": "pool", "averagepooling2d": "pool",
    "globalaveragepooling1d": "pool", "globalaveragepooling2d": "pool",
    "globalmaxpooling1d": "pool", "globalmaxpooling2d": "pool",
    "activation": "relu", "relu": "relu", "leakyrelu": "relu", "prelu": "relu",
    "softmax": "softmax", "upsampling1d": "upsample", "upsampling2d": "upsample",
    "inputlayer": "input", "dropout": "norm", "flatten": "pool",
}


def _h5_weight_tensors(f) -> dict:
    """Collect every weight dataset under the (legacy) `model_weights` group
    or the file root — names become `layer/param` keys."""
    import numpy as np  # noqa: F401

    root = f["model_weights"] if "model_weights" in f else f
    tensors: dict = {}

    def visit(name, obj):
        import h5py

        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in ("f", "i", "u"):
            tensors[name] = obj[()]

    root.visititems(visit)
    return tensors


def load_keras_h5(source_path: str) -> WeightBundle:
    try:
        import h5py
    except ImportError as exc:
        raise WeightLoadError(f"`h5py` not installed: {exc}") from exc

    try:
        f = h5py.File(source_path, "r")
    except Exception as exc:
        raise WeightLoadError(
            f"Not a readable HDF5 file ({type(exc).__name__}: {str(exc)[:120]})"
        ) from exc

    with f:
        tensors = _h5_weight_tensors(f)
        notes: list[str] = []
        layers: list[LayerSpec] | None = None

        raw_config = f.attrs.get("model_config")
        if raw_config is not None:
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            try:
                config = json.loads(raw_config)
                layers = _layers_from_keras_config(config, tensors)
                notes.append(
                    "Architecture recovered from the file's own `model_config` — "
                    "layer graph is the model's REAL topology."
                )
            except Exception as exc:  # noqa: BLE001 — fall back to flat inventory
                notes.append(f"model_config present but unparsable "
                             f"({type(exc).__name__}) — flat weight inventory used.")
        else:
            notes.append("No `model_config` in this HDF5 — it is a weights-only "
                         "Keras file; layer ordering follows the stored groups.")

    return WeightBundle(
        tensors=tensors,
        framework="Keras HDF5 (weights-only analysis)",
        layers=layers,
        notes=notes,
    )


def _layers_from_keras_config(config: dict, tensors: dict) -> list[LayerSpec]:
    """Build REAL LayerSpecs from a Keras model_config (Sequential or Functional)."""
    model_cfg = config.get("config", config)
    raw_layers = model_cfg.get("layers", [])
    specs: list[LayerSpec] = []
    prev_name: str | None = None
    for entry in raw_layers:
        cls = str(entry.get("class_name", "")).lower()
        lcfg = entry.get("config", {}) or {}
        name = lcfg.get("name") or entry.get("name") or f"layer_{len(specs)}"
        kind = _KERAS_KIND.get(cls, "dense")

        # REAL inbound edges (Functional API); Sequential chains implicitly.
        inputs: list[str] = []
        for node in entry.get("inbound_nodes", []) or []:
            if isinstance(node, list):
                for ib in node:
                    if isinstance(ib, list) and ib and isinstance(ib[0], str):
                        inputs.append(ib[0])
            elif isinstance(node, dict):  # Keras 3 style {"args": [...]}
                blob = json.dumps(node)
                for piece in blob.split('"keras_history": ["')[1:]:
                    inputs.append(piece.split('"')[0])
        if not inputs and prev_name is not None:
            inputs = [prev_name]

        keys = [k for k in tensors if k.split("/")[0] == name or k.startswith(name + "/")]
        params = int(sum(tensors[k].size for k in keys))
        specs.append(LayerSpec(name=name, kind=kind, params=params,
                               inputs=inputs, tensor_keys=keys))
        prev_name = name
    return specs


# ── CoreML (.mlmodel) — the spec protobuf IS the real graph ─────────────────

_COREML_KIND = {
    "convolution": "conv", "innerProduct": "dense", "batchnorm": "bn",
    "activation": "relu", "pooling": "pool", "softmax": "softmax",
    "embedding": "embed", "simpleRecurrent": "lstm", "uniDirectionalLSTM": "lstm",
    "biDirectionalLSTM": "lstm", "gru": "lstm", "upsample": "upsample",
    "loadConstant": "norm", "scale": "norm", "add": "norm", "multiply": "norm",
    "reshape": "pool", "flatten": "pool", "padding": "pool", "concat": "pool",
}


def load_coreml(source_path: str) -> WeightBundle:
    try:
        from coremltools.proto import Model_pb2
    except ImportError as exc:
        raise WeightLoadError(
            f"`coremltools` not installed — cannot read CoreML specs: {exc}"
        ) from exc
    import numpy as np

    spec = Model_pb2.Model()
    try:
        spec.ParseFromString(Path(source_path).read_bytes())
    except Exception as exc:
        raise WeightLoadError(
            f"Not a CoreML model protobuf ({type(exc).__name__})"
        ) from exc

    which = spec.WhichOneof("Type")
    nn = getattr(spec, which, None) if which else None
    if which not in ("neuralNetwork", "neuralNetworkClassifier", "neuralNetworkRegressor"):
        raise WeightLoadError(
            f"CoreML model type `{which}` is not a classic NeuralNetwork spec — "
            "ML Program (.mlpackage) weight extraction is not supported yet."
        )

    tensors: dict = {}
    specs: list[LayerSpec] = []
    produced_by: dict[str, str] = {}
    for layer in nn.layers:
        ltype = layer.WhichOneof("layer") or "unknown"
        kind = _COREML_KIND.get(ltype, "dense")
        keys: list[str] = []
        for wname in ("weights", "bias"):
            holder = getattr(getattr(layer, ltype, None), wname, None)
            fv = getattr(holder, "floatValue", None) if holder is not None else None
            if fv:
                arr = np.asarray(list(fv), dtype=np.float32)
                key = f"{layer.name}/{wname}"
                tensors[key] = arr
                keys.append(key)
        inputs = [produced_by[i] for i in layer.input if i in produced_by]
        for out in layer.output:
            produced_by[out] = layer.name
        params = int(sum(tensors[k].size for k in keys))
        specs.append(LayerSpec(name=layer.name, kind=kind, params=params,
                               inputs=inputs, tensor_keys=keys))

    return WeightBundle(
        tensors=tensors,
        framework=f"CoreML {which} (weights-only analysis)",
        layers=specs,
        notes=["Layer graph read from the CoreML spec protobuf — REAL topology."],
    )


# ── GGUF — inventory of already-quantized LLM tensors ────────────────────────

def load_gguf(source_path: str) -> WeightBundle:
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise WeightLoadError(f"`gguf` reader not installed: {exc}") from exc
    import numpy as np

    try:
        reader = GGUFReader(source_path)
    except Exception as exc:
        raise WeightLoadError(
            f"Not a readable GGUF file ({type(exc).__name__}: {str(exc)[:120]})"
        ) from exc

    def _kind(name: str) -> str:
        n = name.lower()
        if "embd" in n or "embed" in n or "token" in n:
            return "embed"
        if "attn" in n or "attention" in n:
            return "attn"
        if "ffn" in n or "mlp" in n:
            return "ffn"
        if "norm" in n:
            return "norm"
        if "output" in n or "head" in n:
            return "dense"
        return "dense"

    float_types = {"F32", "F16", "BF16"}
    tensors: dict = {}
    specs: list[LayerSpec] = []
    already_quant = False
    for t in reader.tensors:
        tname = str(getattr(t.tensor_type, "name", t.tensor_type))
        n_el = 1
        for d in t.shape:
            n_el *= int(d)
        keys: list[str] = []
        if tname in float_types:
            tensors[t.name] = np.asarray(t.data, dtype=np.float32).reshape(-1)
            keys = [t.name]
        else:
            already_quant = True
        # REAL inventory: names/shapes from the GGUF header; quantized tensors
        # have no float weights to measure, so they carry no tensor_keys —
        # the pipeline reports their sensitivity as not-measurable.
        display = t.name if tname in float_types else f"{t.name} [{tname}]"
        specs.append(LayerSpec(name=display, kind=_kind(t.name),
                               params=n_el, tensor_keys=keys))
    for a, b in zip(specs, specs[1:], strict=False):
        b.inputs = [a.name]  # storage order — GGUF stores no compute graph

    notes = ["Tensor inventory read from the GGUF header — names/shapes/quant types are real."]
    if already_quant:
        notes.append("Tensors are ALREADY quantized (Q4/Q8 …) — quantization "
                     "sensitivity is not applicable and is not invented; "
                     "ordering follows storage order (GGUF stores no compute graph).")
    return WeightBundle(
        tensors=tensors,
        framework="GGUF (weights-only analysis)",
        layers=specs,
        notes=notes,
        quantized_already=already_quant,
    )
