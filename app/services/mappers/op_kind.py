"""ONNX op_type → frontend LayerKind mapping.

The 14 LayerKind values come from Astra-Front/src/features/architecture/types.ts;
op categories from astra/graph/onnx_analyzer.py `_CATEGORY_MAP`.
Data-movement / preprocessing ops return None → dropped (edges bridged).
"""

from __future__ import annotations

_DIRECT: dict[str, str] = {
    "Conv": "conv",
    "ConvInteger": "conv",
    "ConvTranspose": "upsample",
    "Resize": "upsample",
    "Upsample": "upsample",
    "BatchNormalization": "bn",
    "InstanceNormalization": "bn",
    "LayerNormalization": "norm",
    "GroupNormalization": "norm",
    "Relu": "relu",
    "LeakyRelu": "relu",
    "Sigmoid": "relu",
    "Tanh": "relu",
    "Gelu": "relu",
    "Selu": "relu",
    "Elu": "relu",
    "Clip": "relu",
    "Softmax": "softmax",
    "LogSoftmax": "softmax",
    "AveragePool": "pool",
    "MaxPool": "pool",
    "GlobalAveragePool": "pool",
    "GlobalMaxPool": "pool",
    "MatMul": "dense",
    "MatMulInteger": "dense",
    "Gemm": "dense",
    "Gather": "embed",
    "GatherElements": "embed",
    "LSTM": "lstm",
    "GRU": "lstm",
    "RNN": "lstm",
    # tf2onnx converts Keras-3 recurrent layers via tf.function tracing, which
    # emits a generic ONNX `Loop`/`Scan` instead of the fused `LSTM`/`GRU` op
    # (the fused op only comes from the from_keras path, broken on Keras 3). The
    # recurrent weights ride on the Loop node as initializers, so dropping it
    # erases the whole RNN — every layer AND its params — from the viz and the
    # param total. Surface it as a recurrent block instead.
    "Loop": "lstm",
    "Scan": "lstm",
    # Classical-ML operators — no dedicated LayerKind, rendered as dense.
    "TreeEnsembleClassifier": "dense",
    "TreeEnsembleRegressor": "dense",
    "TreeEnsemble": "dense",
    "LinearClassifier": "dense",
    "LinearRegressor": "dense",
    "SVMClassifier": "dense",
    "SVMRegressor": "dense",
}

# torch exporters emit scope-style names like "/q/MatMul", "/attn/qkv/MatMul".
_ATTN_NAME_TOKENS = ("attn", "attention", "query", "/q/", "/k/", "/v/", "qkv")

ATTN_WINDOW = 5  # mirrors astra/graph/model_detector.py:_detect_attention_pattern


def kind_for(op_name: str, op_type: str, *, in_attention_window: bool = False) -> str | None:
    """Returns a LayerKind or None when the op should be dropped from the viz."""
    kind = _DIRECT.get(op_type)
    if kind is None:
        return None
    if kind == "dense":
        lower = op_name.lower()
        if in_attention_window or any(tok in lower for tok in _ATTN_NAME_TOKENS):
            return "attn"
    return kind


def find_attention_ops(ordered_ops: list[tuple[str, str]]) -> set[str]:
    """Detect attention score blocks with the PoC's window semantics
    (astra/graph/model_detector.py:184-191): a MatMul whose next ≤5 ops contain
    a Softmax and at least one more MatMul. Real exports insert a scale `Mul`
    between QK^T and Softmax, so a strict [MatMul,Softmax,MatMul] match misses
    every actual attention block."""
    attn: set[str] = set()
    types = [t for _, t in ordered_ops]
    for i in range(len(types)):
        if types[i] != "MatMul":
            continue
        window = types[i:i + ATTN_WINDOW]
        if "Softmax" in window and window.count("MatMul") >= 2:
            soft = i + window.index("Softmax")
            for j in range(i, min(i + ATTN_WINDOW, len(types))):
                # tag the score MatMul(s) before Softmax and the context
                # MatMul right after it
                if types[j] == "MatMul" and (j <= soft or j == soft + 1):
                    attn.add(ordered_ops[j][0])
    return attn
