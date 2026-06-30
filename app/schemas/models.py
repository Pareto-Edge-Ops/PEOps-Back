"""Mirrors Astra-Front/src/features/models/types.ts.

Optionality rules (zod):
  - lastOptimizedAt / bestAccuracy: `.nullable()` — key always present, may be null
  - description / analysisRunId:    `.optional()` — key may be absent entirely
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.schemas.common import IngestionLogLevel, ModelFormat, ModelStatus


class ModelListItem(BaseModel):
    id: str
    name: str
    typeFull: str
    typeShort: str
    format: ModelFormat
    lastLearnedAt: str
    lastOptimizedAt: str | None
    status: ModelStatus
    bestAccuracy: float | None
    isDeployed: bool
    # True only when the model has ≥1 deployment actively routing traffic
    # (live/canary). A model whose deployments are ALL paused stays
    # `isDeployed=True` / `status="deployed"` (it still HAS a deployment) but
    # `isServing=False` — the list badge uses this to render the honest
    # "Deployed · paused" state instead of the green serving badge.
    isServing: bool = False
    # True for raw weights-only checkpoints (no executable graph): they skip
    # Pareto/accuracy/latency/certificate. Drives the "no guarantee" label.
    weightsOnly: bool = False
    description: str | None = None
    analysisRunId: str | None = None

    def to_response(self) -> dict:
        """zod `.optional()` fields are dropped when None (vs nullable ones)."""
        data = self.model_dump()
        if data.get("description") is None:
            data.pop("description", None)
        if data.get("analysisRunId") is None:
            data.pop("analysisRunId", None)
        return data


class IngestionLog(BaseModel):
    ts: str
    level: IngestionLogLevel
    message: str


class IngestionRun(BaseModel):
    id: str
    modelId: str
    fileName: str
    startedAt: str
    status: Literal["streaming", "completed", "failed"]


class ImportRequest(BaseModel):
    fileName: str = "uploaded-model.onnx"


class RenameRequest(BaseModel):
    name: str


class ImportResponse(BaseModel):
    runId: str
    modelId: str
    fileName: str
