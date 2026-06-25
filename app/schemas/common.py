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


CapabilityTier = Literal["full", "convertible", "weights_only"]


class FormatCapability(BaseModel):
    """What an uploaded format actually receives from the pipeline.

    Single source of truth for the three-tier truth-in-labeling: `full` (executable
    graph → Pareto + UOSA + fidelity certificate + real latency), `convertible`
    (converted to ONNX first, then behaves as full; falls back to weight-only if
    conversion fails), and `weights_only` (no graph → per-layer SQNR + INT8 artifact
    only). `task_validation` is always False — there is no language/task metric
    (perplexity/BLEU) anywhere; fidelity is output similarity on synthetic probes.
    For `.pt/.pth/.bin` and `.h5` the extension alone cannot decide the tier (a full
    module vs a bare state_dict / a Keras file with vs without its config), so the
    matrix states the *best case* and the runtime `weights_only` flag on the model is
    the definitive answer once uploaded.
    """

    format: str            # ModelFormat literal value, e.g. "GGUF"
    extensions: list[str]  # e.g. [".gguf"]
    tier: CapabilityTier
    pareto: bool           # sensitivity-guided Pareto search runs
    realLatency: bool      # latency measured on onnxruntime
    certificate: bool      # output-fidelity certificate produced
    taskValidation: bool   # always False — no perplexity/BLEU/token-accuracy exists
    llmCaveat: bool        # True for formats a Transformer/LLM is plausibly shipped in
    noteKey: str           # i18n key the frontend resolves for the human one-liner


class Spark(BaseModel):
    t: str
    value: float


class OkResponse(BaseModel):
    ok: Literal[True] = True
