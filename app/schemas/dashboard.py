"""Mirrors PEOps-Front/src/features/dashboard/types.ts."""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.common import ActivityKind, RunStatus, Spark


class KpiBlock(BaseModel):
    value: float
    deltaText: str
    spark: list[Spark]


class ComputeUsed(BaseModel):
    used: float
    quota: float
    label: str
    progressNote: str


class KpiSummary(BaseModel):
    activeRuns: KpiBlock
    completedThisWeek: KpiBlock
    liveDeployments: KpiBlock
    computeUsed: ComputeUsed


class DashboardRun(BaseModel):
    id: str
    modelId: str
    name: str
    status: RunStatus
    progressPct: float
    iter: str
    bestAcc: float
    deltaAcc: float


class ParetoSnapshotPoint(BaseModel):
    id: str
    accuracy: float
    latency: float
    size: float
    onFrontier: bool


class ParetoSnapshot(BaseModel):
    modelId: str
    modelName: str
    subtitle: str
    points: list[ParetoSnapshotPoint]
    # Real best trial accuracy of the snapshot model; absent when unknown.
    bestAccuracy: float | None = None


class TopModel(BaseModel):
    rank: int
    modelId: str
    name: str
    bestAccuracy: float
    paretoCoverage: float
    spark: list[float]


class CostSegment(BaseModel):
    label: str
    value: float
    color: str


class ComputeCost(BaseModel):
    usedGpuHours: float
    quotaGpuHours: float
    # No real cloud billing exists for local compute — absent, never invented.
    costUsd: float | None = None
    region: str | None = None
    resetDateText: str | None = None
    noteText: str | None = None
    segments: list[CostSegment]


class ActivityEvent(BaseModel):
    id: str
    kind: ActivityKind
    text: str
    timestamp: str
