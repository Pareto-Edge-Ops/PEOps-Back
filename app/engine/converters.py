"""Format → ONNX converters that feed the FULL real pipeline.

tf2onnx's `from_keras` is broken with Keras 3 (KeyError keras_tensor_*), so
Keras models are converted by tracing a `tf.function` wrapper instead —
`from_function` bypasses the broken Keras-internal name mapping. TFLite uses
tf2onnx's own flatbuffer frontend; frozen GraphDefs go through
`from_graph_def` with discovered placeholder/output names.

Every converter returns the path of a REAL exported ONNX file (or raises with
an honest, actionable error message). All imports are lazy so the API boots
without TensorFlow.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

Emit = Callable[[str, str], None]


class ConversionError(Exception):
    """Raised when a real conversion is impossible — message is user-facing."""


def _silence_tf() -> None:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def convert_keras_to_onnx(source_path: str, out_dir: str, emit: Emit) -> tuple[str, list[int] | None]:
    """Load a Keras .h5/.keras model and export real ONNX via tf.function tracing.

    Returns (onnx_path, input_shape). Raises ConversionError when the model
    can't be loaded (e.g. ancient Keras 2 layer configs) — callers fall back
    to the weight-only pipeline.
    """
    _silence_tf()
    try:
        import keras
        import tensorflow as tf
        import tf2onnx
    except ImportError as exc:
        raise ConversionError(f"TensorFlow/Keras not available: {exc}") from exc

    try:
        model = keras.models.load_model(source_path, compile=False)
    except Exception as exc:
        # Keras 3 can't parse legacy (Keras 2.x) .h5 configs — e.g. an ancient
        # Sequential whose `config` is a flat list of layers, which trips
        # "TypeError: pop expected at most 1 argument, got 2". Before giving up
        # and falling back to weight-only analysis, retry with tf_keras (the
        # Keras 2 compatibility runtime), which loads these full models cleanly.
        emit("INFO", f"Keras 3 load failed ({type(exc).__name__}) — "
                     f"retrying with tf_keras (Keras 2 compatibility runtime)")
        try:
            import tf_keras

            model = tf_keras.models.load_model(source_path, compile=False)
        except Exception as exc2:
            raise ConversionError(
                f"Keras could not load this file "
                f"(keras3: {type(exc).__name__}; tf_keras: {type(exc2).__name__}: "
                f"{str(exc2)[:80]}) — it may be a weights-only file or saved by an "
                "incompatible Keras version."
            ) from exc2

    emit("INFO", f"Keras model loaded: {type(model).__name__} · "
                 f"{len(getattr(model, 'layers', []))} layers")

    try:
        inputs = model.inputs
        if not inputs:
            raise ConversionError("Keras model exposes no symbolic inputs to trace.")
        specs = tuple(
            tf.TensorSpec(
                [1 if d is None else int(d) for d in inp.shape],
                tf.as_dtype(inp.dtype) if not isinstance(inp.dtype, str) else inp.dtype,
                name=f"input_{i}",
            )
            for i, inp in enumerate(inputs)
        )

        @tf.autograph.experimental.do_not_convert
        def call(*args):
            out = model(args[0] if len(args) == 1 else list(args))
            return out

        fn = tf.function(call)
        emit("INFO", "tf2onnx.from_keras is incompatible with Keras 3 — converting via "
                     "tf.function tracing instead (from_function)")
        onnx_model, _ = tf2onnx.convert.from_function(fn, input_signature=specs, opset=17)
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(
            f"Keras→ONNX tracing failed ({type(exc).__name__}: {str(exc)[:120]})"
        ) from exc

    out_path = Path(out_dir) / (Path(source_path).stem + "_converted.onnx")
    out_path.write_bytes(onnx_model.SerializeToString())
    shape = [1 if d is None else int(d) for d in inputs[0].shape]
    emit("INFO", f"Exported real ONNX: {out_path.name} "
                 f"({len(onnx_model.graph.node)} ops, input {shape})")
    return str(out_path), shape


def convert_tflite_to_onnx(source_path: str, out_dir: str, emit: Emit) -> tuple[str, list[int] | None]:
    """Convert a .tflite flatbuffer to real ONNX via tf2onnx's tflite frontend."""
    _silence_tf()
    try:
        import tf2onnx
    except ImportError as exc:
        raise ConversionError(f"tf2onnx not available: {exc}") from exc

    try:
        onnx_model, _ = tf2onnx.convert.from_tflite(source_path, opset=17)
    except Exception as exc:
        raise ConversionError(
            f"TFLite→ONNX conversion failed ({type(exc).__name__}: {str(exc)[:120]})"
        ) from exc

    out_path = Path(out_dir) / (Path(source_path).stem + "_converted.onnx")
    out_path.write_bytes(onnx_model.SerializeToString())
    shape = _onnx_first_input_shape(onnx_model)
    emit("INFO", f"Converted TFLite → real ONNX: {out_path.name} "
                 f"({len(onnx_model.graph.node)} ops)")
    return str(out_path), shape


def convert_frozen_pb_to_onnx(source_path: str, out_dir: str, emit: Emit) -> tuple[str, list[int] | None]:
    """Convert a frozen TF GraphDef (.pb) — discovers placeholders/outputs."""
    _silence_tf()
    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError as exc:
        raise ConversionError(f"TensorFlow not available: {exc}") from exc

    gd = tf.compat.v1.GraphDef()
    try:
        gd.ParseFromString(Path(source_path).read_bytes())
    except Exception as exc:
        raise ConversionError(
            f"Not a frozen GraphDef protobuf ({type(exc).__name__}) — SavedModel "
            "directories must be zipped or exported differently."
        ) from exc

    placeholders = [n.name for n in gd.node if n.op == "Placeholder"]
    consumed: set[str] = set()
    for n in gd.node:
        for i in n.input:
            consumed.add(i.split(":")[0].lstrip("^"))
    outputs = [n.name for n in gd.node
               if n.name not in consumed and n.op not in ("Const", "Placeholder", "NoOp", "Assert")]
    if not placeholders or not outputs:
        raise ConversionError(
            f"Could not identify graph I/O (inputs={placeholders[:3]}, outputs={outputs[:3]})."
        )
    emit("INFO", f"Frozen graph I/O discovered — inputs: {placeholders}, outputs: {outputs[:3]}")

    try:
        onnx_model, _ = tf2onnx.convert.from_graph_def(
            gd,
            input_names=[f"{p}:0" for p in placeholders],
            output_names=[f"{o}:0" for o in outputs[:1]],
            opset=17,
        )
    except Exception as exc:
        raise ConversionError(
            f"GraphDef→ONNX conversion failed ({type(exc).__name__}: {str(exc)[:120]})"
        ) from exc

    out_path = Path(out_dir) / (Path(source_path).stem + "_converted.onnx")
    out_path.write_bytes(onnx_model.SerializeToString())
    shape = _onnx_first_input_shape(onnx_model)
    emit("INFO", f"Converted frozen .pb → real ONNX: {out_path.name} "
                 f"({len(onnx_model.graph.node)} ops)")
    return str(out_path), shape


def _onnx_first_input_shape(onnx_model) -> list[int] | None:
    try:
        inp = onnx_model.graph.input[0]
        dims = [d.dim_value if d.dim_value > 0 else 1
                for d in inp.type.tensor_type.shape.dim]
        return dims or None
    except Exception:
        return None
