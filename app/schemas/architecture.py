"""Mirrors PEOps-Front/src/features/architecture/types.ts.

`LayerEdge.from` is a Python keyword — aliased; every dump must use
`by_alias=True` (routers serialize via `to_response()` helpers).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import LayerKind, Recommend


class LayerSummary(BaseModel):
    en: str
    ko: str


class LayerDescription(BaseModel):
    title: str
    summary: LayerSummary
    formula: str | None = None


# Per-node keys with zod `.optional()` on the frontend — omitted when unset.
_OPTIONAL_NODE_KEYS = (
    "zCol", "width", "latencyMs",
    "opType", "category", "inputShape", "outputShape",
    "flops", "units", "precisionSource", "description",
)


class LayerNode(BaseModel):
    id: str
    name: str
    kind: LayerKind
    depth: float
    col: float
    zCol: float | None = None
    sensitivity: float
    params: float
    # Absent for weights-only checkpoints — latency is unmeasurable without
    # an executable graph and is never estimated.
    latencyMs: float | None = None
    recommend: Recommend
    width: float | None = None
    # ── real per-op metadata from the ONNX graph (absent when the node has
    #    no executable op: synthetic input/output, weights-only layers) ──
    opType: str | None = None              # real ONNX op_type (e.g. "Gelu")
    category: str | None = None            # OperatorCategory value
    inputShape: list[int] | None = None    # first input shape; None when dynamic/unknown
    outputShape: list[int] | None = None   # first output shape; None when dynamic/unknown
    flops: int | None = None               # analyzer flops_estimate
    units: int | None = None               # real channel/feature count the width stylizes
    # "pareto" when `recommend` is the served artifact's per-op choice from the
    # selected Pareto point; "recommended" when it is the action-space advice.
    precisionSource: Literal["pareto", "recommended"] | None = None
    description: LayerDescription | None = None


class LayerEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str


class Architecture(BaseModel):
    modelId: str
    modelType: str
    nodes: list[LayerNode]
    edges: list[LayerEdge]

    def to_response(self) -> dict:
        data = self.model_dump(by_alias=True)
        # zod `.optional()` — omit when unset
        for node in data["nodes"]:
            for key in _OPTIONAL_NODE_KEYS:
                if node.get(key) is None:
                    node.pop(key, None)
        return data
