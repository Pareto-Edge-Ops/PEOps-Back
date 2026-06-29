"""Deployment management DTOs (mirrors PEOps-Front/src/features/deployments)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CreateDeploymentRequest(BaseModel):
    region: str | None = None
    name: str | None = None
    status: Literal["live", "canary"] = "live"


class UpdateDeploymentRequest(BaseModel):
    """Partial update — an omitted (None) field is left untouched."""

    name: str | None = None
    description: str | None = None


class DeploymentItem(BaseModel):
    id: str
    name: str
    description: str = ""
    endpoint: str
    region: str
    status: Literal["live", "canary", "paused"]
    qps: float
    p95: float
    errorsPct: float
    accuracyDrift: float
    keyPrefix: str | None = None
    createdAt: str
    lastEventAt: str | None = None


class CreatedDeployment(BaseModel):
    deployment: DeploymentItem
    # The plaintext API key — shown EXACTLY once, never returned again.
    apiKey: str


class RotatedKey(BaseModel):
    apiKey: str
    keyPrefix: str
