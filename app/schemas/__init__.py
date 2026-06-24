"""Pydantic response models mirroring the frontend zod schemas 1:1.

Field names are camelCase on purpose — every response is zod-`.parse()`d by the
SPA, so names/optionality/enums must match `PEOps-Front/src/features/*/types.ts`
exactly.
"""

from app.schemas.architecture import Architecture, LayerEdge, LayerNode
from app.schemas.common import OkResponse
from app.schemas.dashboard import (
    ActivityEvent,
    CompressionMap,
    CompressionPoint,
    DashboardRun,
    GuaranteeCoverage,
    KpiSummary,
    SizeReduced,
    TopModel,
)
from app.schemas.models import (
    ImportRequest,
    ImportResponse,
    IngestionLog,
    IngestionRun,
    ModelListItem,
)
from app.schemas.pareto import ParetoBudget, ParetoExperiment, Trial
from app.schemas.sdk import Recipe, SdkSnippet
from app.schemas.telemetry import (
    Alert,
    Deployment,
    Percentiles,
    TelemetryKpi,
    TelemetryPoint,
)

__all__ = [
    "Architecture", "LayerEdge", "LayerNode",
    "OkResponse",
    "ActivityEvent", "CompressionMap", "CompressionPoint", "DashboardRun",
    "GuaranteeCoverage", "KpiSummary", "SizeReduced", "TopModel",
    "ImportRequest", "ImportResponse", "IngestionLog", "IngestionRun", "ModelListItem",
    "ParetoBudget", "ParetoExperiment", "Trial",
    "Recipe", "SdkSnippet",
    "Alert", "Deployment", "Percentiles", "TelemetryKpi", "TelemetryPoint",
]
