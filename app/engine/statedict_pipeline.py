"""Weight-only pipeline for checkpoints that cannot be executed.

Handles every weights-only container (torch state_dict, safetensors, Keras
HDF5, CoreML specs, GGUF) through one honest path:

- layer inventory recovered from the container's own metadata — REAL names,
  REAL shapes, REAL param counts, and REAL topology whenever the container
  stores one (Keras `model_config`, CoreML spec); plain tensor stores fall
  back to registration/storage order, stated in the log
- per-layer INT8 quantization sensitivity measured as SQNR on the real weights
- a real INT8-quantized compressed artifact with the actual byte savings
  (skipped, loudly, for already-quantized containers like GGUF)

Latency, task accuracy, Pareto search and benchmarks are *physically
unmeasurable* without an executable graph — they are skipped, loudly and
honestly, never estimated.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.engine.weight_loaders import LayerSpec, WeightBundle

Emit = Callable[[str, str], None]          # (level, message)
Progress = Callable[[int], None]           # 0..100


@dataclass
class StatedictArtifacts:
    architecture: dict          # frontend Architecture JSON (response-ready)
    best_accuracy: None         # unmeasurable without a forward graph — never faked
    risk_level: str             # derived from worst real SQNR band
    artifact_path: str | None   # quantized artifact (.pth / .npz) or None
    elapsed_sec: float
    max_sensitivity: float      # highest real per-layer sensitivity ∈ [0,1]


# Parameter-name suffixes that belong to one logical layer (torch style).
_PARAM_SUFFIXES = (
    "weight", "bias", "running_mean", "running_var", "num_batches_tracked",
    "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
    "gamma", "beta", "moving_mean", "moving_variance", "kernel",
)


def is_state_dict(obj: object) -> bool:
    """True when torch.load() returned a plain mapping of tensors (state_dict)."""
    import torch

    if not isinstance(obj, dict) or not obj:
        return False
    return all(isinstance(v, torch.Tensor) for v in obj.values())


def _strip_prefix(key: str) -> str:
    # nn.DataParallel saves everything under `module.` — strip it.
    return key[len("module."):] if key.startswith("module.") else key


def _layer_path(key: str) -> str:
    """Group `basenet.slice1.0.weight` → layer `basenet.slice1.0`; h5-style
    `block1_conv1/block1_conv1_W:0` groups by its first path segment."""
    if "/" in key:
        return key.split("/")[0]
    parts = key.rsplit(".", 1)
    if len(parts) != 2:
        return key
    path, leaf = parts
    return path if leaf.split(":")[0] in _PARAM_SUFFIXES else key


def _classify(path: str, tensors: dict) -> str:
    """Infer the layer kind from REAL tensor shapes (+ name hints)."""
    name = path.lower()
    weight = None
    has_running = False
    for k, v in tensors.items():
        leaf = k.rsplit(".", 1)[-1].rsplit("/", 1)[-1].split(":")[0].lower()
        if "running_mean" in leaf or "moving_mean" in leaf:
            has_running = True
        if leaf in ("weight", "kernel") or leaf.endswith("_w") or weight is None:
            if hasattr(v, "ndim") and (weight is None or v.ndim > weight.ndim):
                weight = v
    if has_running or "batchnorm" in name or name.endswith("bn"):
        return "bn"
    if weight is not None:
        if weight.ndim == 4:
            return "conv"
        if weight.ndim == 3:
            return "conv"
        if weight.ndim == 2:
            if "embed" in name:
                return "embed"
            return "dense"
        if weight.ndim == 1:
            return "norm"
    if any("weight_ih" in k for k in tensors):
        return "lstm"
    if "attn" in name or "attention" in name:
        return "attn"
    if "lstm" in name or "gru" in name:
        return "lstm"
    return "dense"


def _sqnr_db_int8(arr) -> float:
    """Real INT8 quantization SQNR (dB) of one weight tensor.

    Same symmetric per-tensor scheme as peops' OnnxTransformer weight path.
    """
    import numpy as np

    a = arr.astype(np.float32)
    scale = float(np.abs(a).max()) / 127.0 + 1e-12
    q = np.round(a / scale).clip(-127, 127) * scale
    signal = float(np.mean(a ** 2))
    noise = float(np.mean((a - q) ** 2))
    if noise <= 0:
        return 80.0  # numerically lossless at fp32 resolution
    return 10.0 * math.log10(signal / noise + 1e-30)


def _recommend(sqnr_db: float) -> str:
    if sqnr_db >= 40.0:
        return "INT8"
    if sqnr_db >= 20.0:
        return "FP16"
    return "FP32"


def _bundle_from_flat_tensors(tensors: dict) -> list[LayerSpec]:
    """Group a flat tensor mapping (state_dict / safetensors / weights-only h5)
    into logical layers by their REAL parameter names."""
    layers: dict[str, dict] = {}
    order: list[str] = []
    for key, tensor in tensors.items():
        clean = _strip_prefix(key)
        path = _layer_path(clean)
        if path not in layers:
            layers[path] = {}
            order.append(path)
        layers[path][clean] = tensor
    specs: list[LayerSpec] = []
    prev: str | None = None
    for path in order:
        group = layers[path]
        spec = LayerSpec(
            name=path,
            kind=_classify(path, group),
            params=int(sum(getattr(t, "size", 0) for t in group.values())),
            inputs=[prev] if prev is not None else [],
            tensor_keys=list(group.keys()),
        )
        specs.append(spec)
        prev = path
    return specs


def _topo_depths(specs: list[LayerSpec]) -> dict[str, int]:
    """Longest-path depth over the REAL edges (falls back to list order)."""
    by_name = {s.name: s for s in specs}
    depth: dict[str, int] = {}

    def resolve(name: str, seen: frozenset) -> int:
        if name in depth:
            return depth[name]
        spec = by_name.get(name)
        if spec is None or name in seen:
            return 0
        parents = [p for p in spec.inputs if p in by_name]
        d = 1 + max((resolve(p, seen | {name}) for p in parents), default=0)
        depth[name] = d
        return d

    for s in specs:
        resolve(s.name, frozenset())
    return depth


def run_weight_only_pipeline(
    *,
    model_id: str,
    model_name: str,
    file_name: str,
    source_path: str,
    bundle: WeightBundle,
    run_id: str,
    emit: Emit,
    progress: Progress,
    storage_dir: str,
    should_cancel: Callable[[], bool],
) -> StatedictArtifacts:
    import numpy as np

    t0 = time.time()

    def check_cancel() -> None:
        if should_cancel():
            from app.engine.adapter import PipelineCancelled

            raise PipelineCancelled(run_id)

    emit("INFO", "═══ Phase 1/3 · Checkpoint Ingestion (weights-only) ═══")
    size_mb = Path(source_path).stat().st_size / 1e6
    emit("INFO", f"Reading {file_name} ({size_mb:.2f} MB) · {bundle.framework}")
    emit("WARN", "This file carries weights only — it has no executable forward "
                 "graph, so latency, task accuracy and Pareto search are NOT "
                 "possible and will be skipped. Nothing is estimated or invented.")
    emit("INFO", "To unlock full optimization: export an executable model "
                 "(ONNX, full torch.save(model), Keras .h5 with config, or TFLite) "
                 "and re-upload.")
    for note in bundle.notes:
        emit("INFO", note)

    if bundle.layers is not None:
        specs = bundle.layers
        has_real_topology = any(s.inputs for s in specs)
        if has_real_topology:
            emit("INFO", f"Recovered {len(specs)} layers WITH their real connections "
                         f"from the container's own metadata")
    else:
        specs = _bundle_from_flat_tensors(bundle.tensors)
        has_real_topology = False
        emit("INFO", f"Recovered {len(specs)} layers from "
                     f"{len(bundle.tensors)} tensors (registration order preserved)")
        if any(k.startswith("module.") for k in bundle.tensors):
            emit("INFO", "Detected nn.DataParallel checkpoint — stripped `module.` prefix")
    total_params = sum(s.params for s in specs)
    emit("INFO", f"Total parameters: {total_params:,} (measured from tensor shapes)")
    progress(20)
    check_cancel()

    # ── Phase 2 · real per-layer INT8 sensitivity (SQNR on actual weights) ──
    emit("INFO", "═══ Phase 2/3 · Weight Quantization Sensitivity (real SQNR) ═══")
    sqnr_by_layer: dict[str, float] = {}
    for s in specs:
        weight = None
        for k in s.tensor_keys:
            t = bundle.tensors.get(k)
            if t is not None and getattr(t, "ndim", 0) >= 2 and t.size > 0:
                if weight is None or t.size > weight.size:
                    weight = t
        if weight is None:  # fall back to the largest 1-D tensor
            for k in s.tensor_keys:
                t = bundle.tensors.get(k)
                if t is not None and t.size > 0 and (weight is None or t.size > weight.size):
                    weight = t
        if weight is not None and weight.size > 0:
            sqnr_by_layer[s.name] = _sqnr_db_int8(np.asarray(weight))

    measured = list(sqnr_by_layer.values())
    lo, hi = (min(measured), max(measured)) if measured else (0.0, 1.0)
    span = (hi - lo) or 1.0

    depths = _topo_depths(specs) if has_real_topology else {
        s.name: i + 1 for i, s in enumerate(specs)
    }

    nodes: list[dict] = []
    name_to_id: dict[str, str] = {}
    for i, s in enumerate(specs):
        sqnr = sqnr_by_layer.get(s.name)
        if sqnr is None:
            sensitivity, recommend = 0.0, "FP32"
        else:
            # lower SQNR = more quantization damage = higher sensitivity
            sensitivity = round(float((hi - sqnr) / span), 4)
            recommend = _recommend(sqnr)
        nid = f"l{i}"
        name_to_id[s.name] = nid
        nodes.append({
            "id": nid,
            "name": s.name,
            "kind": s.kind,
            "depth": float(depths.get(s.name, i + 1)),
            "col": 0.0,
            "sensitivity": sensitivity,
            "params": float(s.params),
            # latencyMs intentionally OMITTED — unmeasurable without a graph
            "recommend": recommend,
        })
    if measured:
        ranked = sorted(sqnr_by_layer.items(), key=lambda x: x[1])
        for path, sqnr in ranked[:5]:
            emit("INFO", f"  SQNR({path}) = {sqnr:.1f} dB → {_recommend(sqnr)}")
        worst, worst_sqnr = ranked[0]
        emit("WARN", f"Layer {worst} is most quantization-sensitive "
                     f"({worst_sqnr:.1f} dB) — keep at {_recommend(worst_sqnr)}")
    else:
        emit("INFO", "No float weights available to measure (already-quantized "
                     "container) — sensitivity not computed.")
    progress(60)
    check_cancel()

    # spread parallel branches across columns when the real topology has them
    if has_real_topology:
        by_depth: dict[float, list[dict]] = {}
        for n in nodes:
            by_depth.setdefault(n["depth"], []).append(n)
        for siblings in by_depth.values():
            k = len(siblings)
            for j, n in enumerate(siblings):
                n["col"] = round((j - (k - 1) / 2) * 1.2, 2)
        emit("INFO", "Edges reflect the container's REAL layer connections.")
    else:
        emit("INFO", "Edges reflect tensor registration order — this container "
                     "stores no graph topology, so true branching/skip "
                     "connections are unknown.")

    arch_nodes = (
        [{"id": "in", "name": "input", "kind": "input", "depth": 0.0, "col": 0.0,
          "sensitivity": 0.0, "params": 0.0, "recommend": "FP32"}]
        + nodes
        + [{"id": "out", "name": "output", "kind": "output",
            "depth": float(max((n["depth"] for n in nodes), default=0) + 1), "col": 0.0,
            "sensitivity": 0.0, "params": 0.0, "recommend": "FP32"}]
    )
    edges: list[dict] = []
    if has_real_topology:
        consumed: set[str] = set()
        for s in specs:
            for parent in s.inputs:
                if parent in name_to_id:
                    edges.append({"from": name_to_id[parent], "to": name_to_id[s.name]})
                    consumed.add(parent)
        roots = [s.name for s in specs if not any(p in name_to_id for p in s.inputs)]
        for r in roots:
            edges.insert(0, {"from": "in", "to": name_to_id[r]})
        for s in specs:
            if s.name not in consumed:
                edges.append({"from": name_to_id[s.name], "to": "out"})
    else:
        ids = [n["id"] for n in arch_nodes]
        edges = [{"from": a, "to": b} for a, b in zip(ids, ids[1:], strict=False)]

    architecture = {
        "modelId": model_id,
        "modelType": bundle.framework,
        "nodes": arch_nodes,
        "edges": edges,
    }

    # ── Phase 3 · real INT8 weight-only compression artifact ────────────────
    emit("INFO", "═══ Phase 3/3 · Weight-Only INT8 Compression ═══")
    artifact_path: str | None = None
    if bundle.quantized_already:
        emit("INFO", "Skipped — tensors are already quantized in this container; "
                     "re-quantizing would be meaningless.")
    elif not bundle.tensors:
        emit("INFO", "Skipped — no float weights available to compress.")
    else:
        out_dir = Path(storage_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        quantized: dict = {}
        for key, tensor in bundle.tensors.items():
            arr = np.asarray(tensor)
            if arr.dtype == np.float32 and arr.ndim >= 2 and arr.size > 0:
                scale = float(np.abs(arr).max()) / 127.0 + 1e-12
                quantized[key] = np.round(arr / scale).clip(-127, 127).astype(np.int8)
                quantized[key + ".__scale__"] = np.float32(scale)
            else:
                quantized[key] = arr
        path = out_dir / f"{model_id}_compressed.npz"
        np.savez_compressed(str(path), **{k: v for k, v in quantized.items()})
        new_mb = path.stat().st_size / 1e6
        artifact_path = str(path)
        emit("INFO", f"Compressed artifact · {path.name} "
                     f"({new_mb:.2f} MB, was {size_mb:.2f} MB — "
                     f"real {100 * (1 - new_mb / size_mb):.1f}% smaller)")
    progress(95)
    check_cancel()

    max_sensitivity = max((n["sensitivity"] for n in nodes), default=0.0)
    risk = "review" if (measured and min(measured) < 20.0) else "safe"
    if measured:
        emit("INFO", f"Weight-only analysis complete · risk={risk} "
                     f"(worst SQNR {min(measured):.1f} dB)")
    else:
        emit("INFO", "Weight-only analysis complete")
    emit("INFO", "Pareto search skipped — no executable graph. Accuracy left unset "
                 "(never estimated).")
    elapsed = time.time() - t0
    emit("INFO", f"Pipeline finished in {elapsed:.1f}s")
    emit("INFO", "Sensitivity analysis ready")
    progress(100)

    return StatedictArtifacts(
        architecture=architecture,
        best_accuracy=None,
        risk_level=risk,
        artifact_path=artifact_path,
        elapsed_sec=elapsed,
        max_sensitivity=max_sensitivity,
    )
