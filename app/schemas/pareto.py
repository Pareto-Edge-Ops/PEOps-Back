"""Mirrors Astra-Front/src/features/pareto/types.ts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Trial(BaseModel):
    id: str
    name: str
    accuracy: float
    latency: float
    size: float
    score: float
    quant: str
    onFrontier: bool
    trialNumber: int = -1   # Optuna trial number — the per-trial export handle


class ParetoBudget(BaseModel):
    maxLatency: float
    maxAccuracyDrop: float
    maxSize: float


class ParetoExperiment(BaseModel):
    modelId: str
    modelName: str
    experimentId: str
    status: Literal["running", "completed", "paused"]
    iterCurrent: int
    iterTotal: int
    budget: ParetoBudget
    baseAccuracy: float
    trials: list[Trial]
    # Optuna trial number whose artifact is the DEFAULT SDK Hub download, or
    # null when the served artifact is a ladder/fallback candidate (not a trial).
    servedTrialNumber: int | None = None
