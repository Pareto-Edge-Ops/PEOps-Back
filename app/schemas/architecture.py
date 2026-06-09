"""Mirrors PEOps-Front/src/features/architecture/types.ts.

`LayerEdge.from` is a Python keyword — aliased; every dump must use
`by_alias=True` (routers serialize via `to_response()` helpers).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import LayerKind, Recommend


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
            for key in ("zCol", "width", "latencyMs"):
                if node.get(key) is None:
                    node.pop(key, None)
        return data
