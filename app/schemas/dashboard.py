"""Mirrors PEOps-Front/src/features/dashboard/types.ts."""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.common import ActivityKind, RunStatus, Spark


class KpiBlock(BaseModel):
    value: float
    deltaText: str
    spark: list[Spark]


class SizeReduced(BaseModel):
    """Headline value of PEOps: how much smaller the portfolio's optimized
    artifacts are than their originals. Replaces the borrowed "compute used"
    quota metric (PEOps sells compression, not GPU spend)."""

    bytesSaved: float          # Σ (baseline − compressed) over models with a ratio
    avgReductionX: float       # mean × smaller (1 / sizeRatio)
    modelCount: int            # models contributing a real savings figure
    deltaText: str             # window-over-window, same grammar as other KPIs
    spark: list[Spark]         # cumulative bytes saved over the trailing window


class KpiSummary(BaseModel):
    activeRuns: KpiBlock
    completedThisWeek: KpiBlock
    liveDeployments: KpiBlock
    sizeReduced: SizeReduced
    # Which window deltas/counts were computed over ("7d", "30d", ...) — the SPA
    # renders this instead of hardcoding "this week"/"last 30 days".
    periodLabel: str = "7d"


class DashboardRun(BaseModel):
    id: str
    modelId: str
    name: str
    status: RunStatus
    progressPct: float
    iter: str
    bestAcc: float
    deltaAcc: float


class CompressionPoint(BaseModel):
    """One model's served/best pick on the portfolio compression map —
    size reduction (×) against accuracy retained (%)."""

    modelId: str
    name: str
    reductionX: float          # × smaller = 1 / sizeRatio
    sizeRatio: float           # compressed / baseline
    accuracyRetained: float    # measured accuracy of the served artifact (%)
    accuracyDrop: float        # baseAccuracy − accuracyRetained (pts)
    withinTolerance: bool      # accuracyDrop ≤ the model's own budget
    certified: bool            # cleared a guarantee gate (source ≠ fallback)
    rung: str | None = None    # guarantee rung code (PARETO_CERTIFIED, FP16, …)
    latencyMs: float | None = None


class CompressionBest(BaseModel):
    modelId: str
    reductionX: float
    accuracyRetained: float


class CompressionMap(BaseModel):
    points: list[CompressionPoint]
    modelCount: int            # optimized models considered (any provenance)
    certifiedCount: int        # of those, models that cleared a guarantee gate
    best: CompressionBest | None = None   # most reduction within tolerance


class TopModel(BaseModel):
    rank: int
    modelId: str
    name: str
    bestAccuracy: float
    paretoCoverage: float
    spark: list[float]


class GuaranteeSegment(BaseModel):
    label: str                 # raw rung code; the SPA localizes it for display
    value: float               # model count in this rung
    color: str


class GuaranteeCoverage(BaseModel):
    """PEOps's unique promise made legible: how many optimized models carry a
    fidelity guarantee, and where on the ladder they landed."""

    certifiedCount: int        # models that cleared a guarantee gate
    totalModels: int           # optimized models with recorded provenance
    avgFidelity: float | None = None   # mean output-fidelity (ladder OFS), if any
    segments: list[GuaranteeSegment]   # rung distribution (SplitBar)


class ActivityEvent(BaseModel):
    id: str
    kind: ActivityKind
    text: str
    timestamp: str
