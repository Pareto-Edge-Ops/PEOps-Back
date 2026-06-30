"""Single source of truth for per-format compression capabilities.

The worker dispatches by file extension (`app/services/jobs.py`): executable
graphs take the full Pareto/UOSA/certificate pipeline, while weights-only
containers (no forward graph) take the honest SQNR-only pipeline. This module
encodes that same routing as data so the UI, the docs, and `infer_format` can
never disagree with what the backend actually does.

Honest scope encoded here:
  - `taskValidation` is False for EVERY format: there is no perplexity/BLEU/
    token-accuracy metric anywhere — fidelity is output similarity on synthetic
    probes (`astra/core/guarantee.py`), not downstream task accuracy.
  - `llmCaveat` flags the containers a Transformer/LLM is plausibly shipped in.
    Even on the full tier those models' "guarantee" is probe-fidelity, and the
    calibration probes are float tensors, not token IDs — so it does not mean
    preserved LLM task performance.
  - `.pt/.pth/.bin/.ckpt` (full module vs bare state_dict) and `.h5/.keras`
    (with vs without config) cannot be tiered from the extension alone; the
    entries below state the BEST case and `ModelRow.weights_only` (set at run
    time, surfaced on the model detail) is the definitive post-upload answer.
"""

from __future__ import annotations

from app.schemas.common import FormatCapability

_NOTE = "common.capability.note"

FORMAT_CAPABILITIES: list[FormatCapability] = [
    FormatCapability(
        format="ONNX", extensions=[".onnx"], tier="full",
        pareto=True, realLatency=True, certificate=True,
        taskValidation=False, llmCaveat=True, noteKey=f"{_NOTE}.full",
    ),
    FormatCapability(
        format="PyTorch", extensions=[".pt", ".pth", ".bin", ".ckpt"], tier="full",
        pareto=True, realLatency=True, certificate=True,
        taskValidation=False, llmCaveat=True, noteKey=f"{_NOTE}.pytorch",
    ),
    FormatCapability(
        format="TensorFlow", extensions=[".pb", ".h5", ".keras"], tier="convertible",
        pareto=True, realLatency=True, certificate=True,
        taskValidation=False, llmCaveat=False, noteKey=f"{_NOTE}.convertible",
    ),
    FormatCapability(
        format="TFLite", extensions=[".tflite"], tier="convertible",
        pareto=True, realLatency=True, certificate=True,
        taskValidation=False, llmCaveat=False, noteKey=f"{_NOTE}.convertible",
    ),
    FormatCapability(
        format="CoreML", extensions=[".mlmodel"], tier="weights_only",
        pareto=False, realLatency=False, certificate=False,
        taskValidation=False, llmCaveat=False, noteKey=f"{_NOTE}.weightsOnly",
    ),
    FormatCapability(
        format="Scikit-learn", extensions=[".pkl", ".joblib"], tier="full",
        pareto=True, realLatency=True, certificate=True,
        taskValidation=False, llmCaveat=False, noteKey=f"{_NOTE}.full",
    ),
    FormatCapability(
        format="SafeTensors", extensions=[".safetensors"], tier="weights_only",
        pareto=False, realLatency=False, certificate=False,
        taskValidation=False, llmCaveat=True, noteKey=f"{_NOTE}.weightsOnly",
    ),
    FormatCapability(
        format="GGUF", extensions=[".gguf"], tier="weights_only",
        pareto=False, realLatency=False, certificate=False,
        taskValidation=False, llmCaveat=True, noteKey=f"{_NOTE}.gguf",
    ),
]

# Default applied to unknown extensions (mirrors infer_format's ONNX fallback).
_DEFAULT_FORMAT = "ONNX"


def capability_for_filename(file_name: str) -> FormatCapability | None:
    """The capability entry whose extension matches `file_name`, else None."""
    lower = file_name.lower()
    for cap in FORMAT_CAPABILITIES:
        if any(lower.endswith(ext) for ext in cap.extensions):
            return cap
    return None


def format_for_filename(file_name: str) -> str:
    """The ModelFormat string for a file (the SSOT that infer_format mirrors)."""
    cap = capability_for_filename(file_name)
    return cap.format if cap is not None else _DEFAULT_FORMAT
