"""Shared literals and primitives."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

RunStatus = Literal["running", "queued", "done", "failed"]
ModelStatus = Literal["deployed", "training", "optimizing", "draft", "failed", "analyzing"]
ModelFormat = Literal[
    "ONNX", "PyTorch", "TensorFlow", "TFLite", "CoreML", "Scikit-learn",
    "SafeTensors", "GGUF",
]
IngestionLogLevel = Literal["INFO", "WARN", "ERROR", "DEBUG"]
LayerKind = Literal[
    "input", "conv", "bn", "relu", "pool", "dense", "attn",
    "ffn", "norm", "output", "embed", "lstm", "softmax", "upsample",
]
Recommend = Literal["INT8", "FP16", "FP32"]
TimeRange = Literal["1h", "6h", "24h", "7d", "30d"]
ActivityKind = Literal[
    "run_started", "run_completed", "deploy_promoted", "accuracy_drift", "model_uploaded",
]
SdkLanguage = Literal["python", "node", "cli", "curl"]


class Spark(BaseModel):
    t: str
    value: float


class OkResponse(BaseModel):
    ok: Literal[True] = True
