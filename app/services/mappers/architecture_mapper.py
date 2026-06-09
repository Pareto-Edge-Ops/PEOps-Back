"""GraphInfo + UOSA SensitivityProfile → frontend Architecture.

Inputs are duck-typed (peops dataclasses or test doubles) so this module has
no peops import; the engine adapter supplies the real `recommend_for` callable
built on peops' `get_action_space`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

from app.schemas.architecture import Architecture, LayerEdge, LayerNode
from app.services.mappers.layout import compute_layout
from app.services.mappers.op_kind import find_attention_ops, kind_for

ARCH_DISPLAY = {
    "MLP": "Multi-Layer Perceptron",
    "CNN": "Convolutional Neural Networks",
    "RNN": "Recurrent Neural Network",
    "Transformer": "Transformer",
    "TreeEnsemble": "Gradient Boosted Trees",
    "LinearModel": "Linear Model",
    "SVM": "Support Vector Machine",
    "Hybrid": "Hybrid Network",
    "Unknown": "Neural Network",
}


class OpLike(Protocol):  # subset of peops OperatorInfo
    name: str
    op_type: str
    param_count: int
    flops_estimate: int
    output_shapes: list


class GraphLike(Protocol):  # subset of peops GraphInfo
    operators: list
    adjacency: dict[str, list[str]]
    topo_order: list[str]


def _width_for(op: OpLike, kind: str) -> float | None:
    if not op.output_shapes or not op.output_shapes[0]:
        return None
    shape = [d for d in op.output_shapes[0] if isinstance(d, int) and d > 0]
    if not shape:
        return None
    units = shape[1] if kind in ("conv", "bn", "pool", "upsample") and len(shape) > 1 else shape[-1]
    return float(max(3, min(22, round(math.sqrt(units)))))


def map_architecture(
    model_id: str,
    graph: GraphLike,
    sensitivity: dict[str, float],
    *,
    architecture_name: str,
    total_latency_ms: float,
    recommend_for: Callable[[object, float], str],
    selected_precisions: dict[str, str] | None = None,
) -> Architecture:
    """`sensitivity` = normalized [0,1] scores keyed by op name.
    `selected_precisions` = per-op precision actually chosen by the Pareto
    point (preferred over the action-space recommendation when present)."""
    ops = {op.name: op for op in graph.operators}
    topo = [n for n in graph.topo_order if n in ops]
    ordered_types = [(n, ops[n].op_type) for n in topo]
    attn_ops = find_attention_ops(ordered_types)

    kept: dict[str, str] = {}  # name -> kind
    for name in topo:
        kind = kind_for(name, ops[name].op_type, in_attention_window=name in attn_ops)
        if kind is not None:
            kept[name] = kind

    if not kept:  # degenerate graph — keep everything as dense
        kept = {name: "dense" for name in topo}

    # Bridge edges through dropped nodes.
    memo: dict[str, set[str]] = {}

    def kept_successors(name: str, _seen: frozenset = frozenset()) -> set[str]:
        if name in memo:
            return memo[name]
        out: set[str] = set()
        for succ in graph.adjacency.get(name, []):
            if succ in _seen:
                continue
            if succ in kept:
                out.add(succ)
            else:
                out |= kept_successors(succ, _seen | {name})
        memo[name] = out
        return out

    edge_pairs: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for name in topo:
        if name not in kept:
            continue
        for succ in sorted(kept_successors(name), key=lambda s: topo.index(s)):
            if (name, succ) not in seen_edges:
                seen_edges.add((name, succ))
                edge_pairs.append((name, succ))

    kept_order = [n for n in topo if n in kept]
    order_index = {n: i for i, n in enumerate(kept_order)}
    pos = compute_layout(kept_order, edge_pairs, order_index)

    has_pred = {to for _, to in edge_pairs}
    has_succ = {frm for frm, _ in edge_pairs}
    sources = [n for n in kept_order if n not in has_pred]
    sinks = [n for n in kept_order if n not in has_succ]

    flops_total = sum(max(0, ops[n].flops_estimate) for n in kept_order) or 1
    latency_floor = round(total_latency_ms * 0.002, 4)

    nodes: list[LayerNode] = [
        LayerNode(
            id="input", name="input", kind="input", depth=0, col=0.0,
            sensitivity=0.05, params=0, latencyMs=0.0, recommend="FP32",
        )
    ]
    for name in kept_order:
        op = ops[name]
        kind = kept[name]
        p = pos[name]
        sens = round(min(1.0, max(0.0, sensitivity.get(name, 0.0))), 4)
        share = max(0, op.flops_estimate) / flops_total
        latency = round(total_latency_ms * share, 4)
        if latency == 0 and kind in ("pool", "relu", "softmax", "norm", "bn"):
            latency = latency_floor

        if selected_precisions and name in selected_precisions:
            rec = selected_precisions[name]
            if rec not in ("INT8", "FP16", "FP32"):
                rec = "FP16"  # INT4 etc. — clamp into the frontend enum
        else:
            rec = recommend_for(op, sens)

        nodes.append(LayerNode(
            id=name,
            name=name,
            kind=kind,  # type: ignore[arg-type]
            depth=p.depth + 1,  # shift: synthetic input occupies depth 0
            col=p.col,
            zCol=p.z_col,
            sensitivity=sens,
            params=op.param_count,
            latencyMs=latency,
            recommend=rec,  # type: ignore[arg-type]
            width=_width_for(op, kind),
        ))

    max_depth = max((n.depth for n in nodes), default=0)
    nodes.append(LayerNode(
        id="output", name="output", kind="output", depth=max_depth + 1, col=0.0,
        sensitivity=0.05, params=0, latencyMs=0.0, recommend="FP32",
    ))

    edges = [LayerEdge(from_="input", to=s) for s in sources]
    edges += [LayerEdge(from_=f, to=t) for f, t in edge_pairs]
    edges += [LayerEdge(from_=s, to="output") for s in sinks]

    return Architecture(
        modelId=model_id,
        modelType=ARCH_DISPLAY.get(architecture_name, architecture_name),
        nodes=nodes,
        edges=edges,
    )
