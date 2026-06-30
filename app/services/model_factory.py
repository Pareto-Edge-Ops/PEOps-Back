"""Synthesize a REAL small model from an import fileName.

`POST /models/import` carries only a fileName (no bytes), so to run the genuine
astra pipeline we build an actual model whose framework matches the extension
and whose architecture matches filename hints (lstm/attn/tree/...). All heavy
imports (torch/sklearn) are lazy — the API boots without them.

`.pb`/`.tflite`/`.mlmodel` converters are not installed; those formats get an
ONNX-equivalent model and keep only the declared format label.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from app.services.family import family_from_file_name
from app.services.formats import infer_format


@dataclass
class SynthModel:
    path: str                    # file the pipeline ingests
    input_shape: list[int] | None
    declared_format: str         # frontend ModelFormat
    family: str                  # han|lstm|diffusion-t|cnn|tree
    note: str                    # one-line description for the ingestion log


_FAMILY_TORCH = {
    "cnn": ("TinyCNN", [1, 1, 16, 16]),
    "han": ("TinyAttention", [1, 6, 16]),
    "diffusion-t": ("TinyAttention", [1, 6, 16]),
    "lstm": ("TinyLSTM", [1, 8, 12]),
    "mlp": ("TinyMLP", [1, 8]),
}


def _build_torch(family: str, seed: int):
    import torch

    from app.services import torch_models

    torch.manual_seed(seed)
    cls_name, shape = _FAMILY_TORCH.get(family, _FAMILY_TORCH["mlp"])
    model = getattr(torch_models, cls_name)()
    model.eval()
    return model, list(shape)


def _export_onnx(model, shape: list[int], out_path: Path) -> None:
    import torch

    dummy = torch.randn(*shape)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["X"], output_names=["Y"], opset_version=17,
        dynamo=False,
    )


def _build_sklearn_pickle(out_path: Path, seed: int) -> None:
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 6)).astype("float64")
    y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
    clf = GradientBoostingClassifier(n_estimators=8, max_depth=2, random_state=seed)
    clf.fit(X, y)
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)


def synthesize(file_name: str, *, out_dir: str, fast: bool, seed: int = 42) -> SynthModel:
    declared_format, _, _ = infer_format(file_name)
    family = family_from_file_name(file_name, declared_format)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(file_name).stem or "model"
    lower = file_name.lower()

    if fast:
        # Tiniest pipeline for tests/CI — 2-Gemm MLP regardless of family.
        path = out / f"{stem}_synth.onnx"
        model, shape = _build_torch("mlp", seed)
        _export_onnx(model, shape, path)
        return SynthModel(str(path), shape, declared_format, family,
                          "fast-mode TinyMLP (2×Gemm) synthesized")

    if lower.endswith((".pkl", ".joblib")) or family == "tree":
        path = out / f"{stem}_synth.pkl"
        _build_sklearn_pickle(path, seed)
        return SynthModel(str(path), None, declared_format, "tree",
                          "GradientBoosting(n_estimators=8) trained on synthetic data")

    if lower.endswith((".pt", ".pth")):
        import torch

        model, shape = _build_torch(family, seed)
        path = out / f"{stem}_synth.pt"
        torch.save(model, str(path))
        return SynthModel(str(path), shape, declared_format, family,
                          f"torch {type(model).__name__} saved as full module")

    # .onnx and converter-less formats (.pb/.tflite/.mlmodel) → ONNX-equivalent
    model, shape = _build_torch(family, seed)
    path = out / f"{stem}_synth.onnx"
    _export_onnx(model, shape, path)
    note = f"torch {type(model).__name__} exported to ONNX (opset 17)"
    if not lower.endswith(".onnx"):
        note += f" — {declared_format} converter unavailable, ONNX-equivalent used"
    return SynthModel(str(path), shape, declared_format, family, note)
