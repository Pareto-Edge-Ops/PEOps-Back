"""Universal Model Ingestion: auto-detect and convert any ML model format to ONNX.

Supports: .pt/.pth (PyTorch), .h5/.pb/.keras (TensorFlow), .pkl/.joblib (scikit-learn),
.onnx (pass-through). Detects format from file content, not just extension.
"""

from __future__ import annotations

import enum
import io
import pickle
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx


class ModelFormat(enum.Enum):
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW_SAVED_MODEL = "tf_saved_model"
    TENSORFLOW_H5 = "tf_h5"
    SKLEARN_PICKLE = "sklearn_pickle"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    UNKNOWN = "unknown"


@dataclass
class IngestionResult:
    """Result of model ingestion."""
    original_path: str
    detected_format: ModelFormat
    onnx_model: onnx.ModelProto
    input_spec: dict[str, list[int]]
    metadata: dict[str, Any]

    @property
    def is_successful(self) -> bool:
        return self.onnx_model is not None


def detect_format(path: str) -> ModelFormat:
    """Detect model format from file content and extension.

    Uses magic bytes and structural analysis, not just file extension,
    to correctly identify the model format.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".onnx":
        return ModelFormat.ONNX

    if p.is_dir():
        if (p / "saved_model.pb").exists():
            return ModelFormat.TENSORFLOW_SAVED_MODEL
        return ModelFormat.UNKNOWN

    try:
        with open(path, "rb") as f:
            header = f.read(16)
    except Exception:
        return ModelFormat.UNKNOWN

    # PyTorch: ZIP archive (torch.save uses zipfile)
    if header[:4] == b'PK\x03\x04':
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                names = zf.namelist()
                if any("data.pkl" in n or "archive/data" in n for n in names):
                    return ModelFormat.PYTORCH
        except zipfile.BadZipFile:
            pass

    # PyTorch legacy format (pickle with magic number)
    if header[:2] == b'\x80\x02' or (suffix in (".pt", ".pth") and header[:4] != b'PK\x03\x04'):
        if suffix in (".pt", ".pth", ".bin"):
            return ModelFormat.PYTORCH

    # HDF5 format (TensorFlow/Keras .h5)
    if header[:8] == b'\x89HDF\r\n\x1a\n':
        return ModelFormat.TENSORFLOW_H5

    # TensorFlow SavedModel (protobuf)
    if suffix == ".pb":
        return ModelFormat.TENSORFLOW_SAVED_MODEL

    # Pickle-based models (sklearn, xgboost, lightgbm)
    if suffix in (".pkl", ".joblib", ".pickle"):
        return _detect_pickle_type(path)

    # Pickle files may start with protocol bytes (0x80 + version)
    if header[0:1] == b'\x80' and suffix not in (".pt", ".pth"):
        return _detect_pickle_type(path)

    if suffix in (".h5", ".hdf5"):
        return ModelFormat.TENSORFLOW_H5

    if suffix == ".keras":
        return ModelFormat.TENSORFLOW_H5

    return ModelFormat.UNKNOWN


def _detect_pickle_type(path: str) -> ModelFormat:
    """Detect the type of a pickle-serialized model."""
    try:
        import joblib
        obj = joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            return ModelFormat.UNKNOWN

    type_name = type(obj).__module__ + "." + type(obj).__name__

    if "sklearn" in type_name or "sklearn" in str(type(obj).__mro__):
        return ModelFormat.SKLEARN_PICKLE
    if "xgboost" in type_name:
        return ModelFormat.XGBOOST
    if "lightgbm" in type_name:
        return ModelFormat.LIGHTGBM

    return ModelFormat.SKLEARN_PICKLE


def ingest(
    path: str,
    input_shape: list[int] | None = None,
    input_dtype: str = "float32",
) -> IngestionResult:
    """Ingest a model from any supported format and convert to ONNX.

    Args:
        path: Path to model file or directory.
        input_shape: Optional input shape hint (e.g., [1, 3, 224, 224]).
                     Required for PyTorch models, auto-inferred for sklearn.
        input_dtype: Input data type (default: float32).

    Returns:
        IngestionResult with ONNX model and metadata.
    """
    fmt = detect_format(path)

    converters = {
        ModelFormat.ONNX: _ingest_onnx,
        ModelFormat.PYTORCH: _ingest_pytorch,
        ModelFormat.TENSORFLOW_H5: _ingest_tensorflow,
        ModelFormat.TENSORFLOW_SAVED_MODEL: _ingest_tensorflow,
        ModelFormat.SKLEARN_PICKLE: _ingest_sklearn,
        ModelFormat.XGBOOST: _ingest_sklearn,
        ModelFormat.LIGHTGBM: _ingest_sklearn,
    }

    converter = converters.get(fmt)
    if converter is None:
        raise ValueError(f"Unsupported model format: {fmt.value} at {path}")

    return converter(path, fmt, input_shape, input_dtype)


def _ingest_onnx(
    path: str, fmt: ModelFormat,
    input_shape: list[int] | None, input_dtype: str,
) -> IngestionResult:
    model = onnx.load(path)
    input_spec = {}
    for inp in model.graph.input:
        if inp.name not in {i.name for i in model.graph.initializer}:
            shape = []
            for d in inp.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.dim_value > 0 else -1)
            input_spec[inp.name] = shape

    return IngestionResult(
        original_path=path,
        detected_format=fmt,
        onnx_model=model,
        input_spec=input_spec,
        metadata={"ir_version": model.ir_version, "producer": model.producer_name},
    )


def _ingest_pytorch(
    path: str, fmt: ModelFormat,
    input_shape: list[int] | None, input_dtype: str,
) -> IngestionResult:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict):
        raise ValueError(
            "PyTorch file contains a state_dict, not a full model. "
            "Provide a model with architecture (torch.save(model, path), not torch.save(model.state_dict(), path))."
        )
    else:
        raise ValueError(f"Unexpected PyTorch object type: {type(obj)}")

    model.eval()

    if input_shape is None:
        raise ValueError("input_shape is required for PyTorch models (e.g., [1, 3, 224, 224])")

    dtype_map = {"float32": torch.float32, "float64": torch.float64, "int32": torch.int32}
    dummy_input = torch.randn(*input_shape, dtype=dtype_map.get(input_dtype, torch.float32))

    buf = io.BytesIO()
    torch.onnx.export(
        model, dummy_input, buf,
        input_names=["X"],
        output_names=["Y"],
        dynamic_axes={"X": {0: "batch"}, "Y": {0: "batch"}},
        opset_version=17,
    )
    buf.seek(0)
    onnx_model = onnx.load_model_from_string(buf.read())

    return IngestionResult(
        original_path=path,
        detected_format=fmt,
        onnx_model=onnx_model,
        input_spec={"X": input_shape},
        metadata={"pytorch_class": type(model).__name__},
    )


def _ingest_tensorflow(
    path: str, fmt: ModelFormat,
    input_shape: list[int] | None, input_dtype: str,
) -> IngestionResult:
    try:
        import tf2onnx
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow and tf2onnx are required for .h5/.pb/.keras conversion. "
            "Install with: pip install tensorflow tf2onnx"
        )

    if fmt == ModelFormat.TENSORFLOW_H5:
        model = tf.keras.models.load_model(path)
    else:
        model = tf.saved_model.load(path)

    import tf2onnx
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=17)

    input_spec = {}
    for inp in onnx_model.graph.input:
        if inp.name not in {i.name for i in onnx_model.graph.initializer}:
            shape = []
            for d in inp.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.dim_value > 0 else -1)
            input_spec[inp.name] = shape

    return IngestionResult(
        original_path=path,
        detected_format=fmt,
        onnx_model=onnx_model,
        input_spec=input_spec,
        metadata={"source": "tensorflow"},
    )


def _ingest_sklearn(
    path: str, fmt: ModelFormat,
    input_shape: list[int] | None, input_dtype: str,
) -> IngestionResult:
    try:
        import joblib
        obj = joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            obj = pickle.load(f)

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

    n_features = _infer_n_features(obj)
    if input_shape is not None and len(input_shape) == 2:
        n_features = input_shape[1]

    if n_features is None:
        raise ValueError("Cannot infer input feature count. Provide input_shape=[1, n_features].")

    dtype_type = DoubleTensorType if input_dtype == "float64" else FloatTensorType
    initial_types = [("X", dtype_type([None, n_features]))]

    onnx_model = convert_sklearn(obj, initial_types=initial_types)

    return IngestionResult(
        original_path=path,
        detected_format=fmt,
        onnx_model=onnx_model,
        input_spec={"X": [-1, n_features]},
        metadata={
            "sklearn_class": type(obj).__name__,
            "n_features": n_features,
        },
    )


def _infer_n_features(model: Any) -> int | None:
    """Infer the number of input features from a sklearn-like model."""
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "feature_importances_"):
        return len(model.feature_importances_)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        return coef.shape[-1] if coef.ndim >= 2 else coef.shape[0]
    return None
