"""File-name → frontend ModelFormat mapping (mirrors models/types.ts enum)."""

from __future__ import annotations


def infer_format(file_name: str) -> tuple[str, str, str]:
    """Returns (format, typeFull, typeShort)."""
    lower = file_name.lower()
    if lower.endswith(".onnx"):
        return "ONNX", "ONNX Imported Model", "ONNX Imported Model"
    if lower.endswith((".pt", ".pth", ".bin", ".ckpt")):
        return "PyTorch", "PyTorch Imported Model", "PyTorch Imported Model"
    if lower.endswith((".pb", ".h5", ".keras")):
        return "TensorFlow", "TensorFlow Imported Model", "TensorFlow Imported Model"
    if lower.endswith(".tflite"):
        return "TFLite", "TFLite Imported Model", "TFLite Imported Model"
    if lower.endswith(".mlmodel"):
        return "CoreML", "CoreML Imported Model", "CoreML Imported Model"
    if lower.endswith((".pkl", ".joblib")):
        return "Scikit-learn", "Scikit-learn Imported Model", "Scikit-learn Imported Model"
    if lower.endswith(".safetensors"):
        return "SafeTensors", "SafeTensors Imported Model", "SafeTensors Imported Model"
    if lower.endswith(".gguf"):
        return "GGUF", "GGUF Imported Model", "GGUF Imported Model"
    return "ONNX", "Imported Model", "Imported Model"


def display_name(file_name: str) -> str:
    dot = file_name.rfind(".")
    base = file_name if dot == -1 else file_name[:dot]
    return base or "Uploaded model"
