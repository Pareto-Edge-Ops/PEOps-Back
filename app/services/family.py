"""Model family inference — single source of truth shared by the architecture,
pareto and ingestion-log generators. Port of the front's `inferFamily`
(architecture/api/mockData.ts + pareto/api/mockData.ts)."""

from __future__ import annotations

FAMILIES = ("han", "lstm", "diffusion-t", "cnn", "tree")


def infer_family(model_id: str, type_full: str | None = None, fmt: str | None = None) -> str:
    if model_id.startswith("m_ggee_han"):
        return "han"
    if model_id.startswith("m_naratmalsame"):
        return "lstm"
    if model_id.startswith("m_ptkorea_qwen"):
        return "diffusion-t"

    t = (type_full or "").lower()
    if "attention" in t or "graph" in t:
        return "han"
    if "lstm" in t or "recurrent" in t or "gru" in t:
        return "lstm"
    if "decomposition" in t or "diffusion" in t or "transformer" in t:
        return "diffusion-t"
    if "tree" in t or "boost" in t or "forest" in t:
        return "tree"
    if fmt == "Scikit-learn":
        return "tree"
    return "cnn"


def family_from_file_name(file_name: str, fmt: str) -> str:
    """Family hint for imported models — filename tokens beat the format."""
    lower = file_name.lower()
    if "lstm" in lower or "gru" in lower or "rnn" in lower:
        return "lstm"
    if "attn" in lower or "attention" in lower or "han" in lower or "transformer" in lower:
        return "han"
    if "diffusion" in lower or "dit" in lower:
        return "diffusion-t"
    if "tree" in lower or "boost" in lower or "forest" in lower or fmt == "Scikit-learn":
        return "tree"
    if "mlp" in lower or "dense" in lower:
        return "cnn"
    return "cnn"
