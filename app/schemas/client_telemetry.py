"""Request schemas for the client-telemetry batch endpoint (peops-sdk)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClientEvent(BaseModel):
    """One locally-served inference, measured by the SDK."""
    ts: str
    latencyMs: float = Field(ge=0)
    preMs: float | None = Field(default=None, ge=0)
    postMs: float | None = Field(default=None, ge=0)
    success: bool = True
    errorCode: str | None = None
    batchSize: int = Field(default=1, ge=1)
    region: str = "local"
    inputSig: str | None = None     # e.g. "input:1x3x224x224:float32"


class ClientSnapshot(BaseModel):
    """Periodic host/system snapshot (~30s cadence)."""
    ts: str
    cpuPct: float = 0.0
    rssMb: float = 0.0
    throughputRpm: float = 0.0
    droppedEvents: int = 0
    sdkVersion: str = ""
    pythonVersion: str = ""
    ortVersion: str = ""
    os: str = ""
    arch: str = ""
    provider: str = ""
    host: str = ""
    # Hardware identity (static, new in the hardware-aware SDK).
    cpuModel: str = ""
    cpuCores: int = 0
    ramTotalMb: float = 0.0
    availableProviders: str = ""     # comma-joined ORT providers
    activeProvider: str = ""         # provider the serving session actually bound
    gpuName: str = ""
    gpuCount: int = 0
    gpuMemTotalMb: float = 0.0
    cudaVersion: str = ""
    # Dynamic accelerator sample (None when the host has no NVIDIA GPU).
    gpuUtilPct: float | None = None
    gpuMemUsedMb: float | None = None
    gpuTempC: float | None = None


class InputStat(BaseModel):
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    nanPct: float = 0.0


class ClientWindowStats(BaseModel):
    """Windowed input/output distribution stats (~60s cadence) — the raw
    signal the drift monitor uses for prediction/input drift."""
    windowStart: str
    windowEnd: str
    n: int = Field(ge=0)
    inputs: dict[str, InputStat] = Field(default_factory=dict)
    # Free-form output summary: {"classDist": {"3": 0.42, ...}, "hist": [...],
    # "entropyMean": x, "top1ConfMean": y} — classifier-shaped outputs only.
    output: dict = Field(default_factory=dict)


class TelemetryBatch(BaseModel):
    clientId: str = Field(min_length=1, max_length=64)
    events: list[ClientEvent] = Field(default_factory=list)
    snapshots: list[ClientSnapshot] = Field(default_factory=list)
    windows: list[ClientWindowStats] = Field(default_factory=list)


class BatchAccepted(BaseModel):
    accepted: dict[str, int]
    dropped: int
