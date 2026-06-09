"""Auto Model Detector: infer architecture type and extract per-layer
mathematical formulas from an ONNX graph without user-provided metadata.

Given only an ONNX model (from any source), this module:
1. Classifies the overall model architecture (CNN, RNN, Transformer, MLP, Tree, Linear, SVM, ...)
2. Extracts per-operator mathematical formulas (e.g., Y = X·W + b, Y = ReLU(X))
3. Reports a human-readable architecture summary
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from peops.graph.onnx_analyzer import OnnxAnalyzer, GraphInfo, OperatorInfo, OperatorCategory


class ArchitectureType(enum.Enum):
    MLP = "MLP"
    CNN = "CNN"
    RNN = "RNN"
    TRANSFORMER = "Transformer"
    TREE_ENSEMBLE = "TreeEnsemble"
    LINEAR_MODEL = "LinearModel"
    SVM = "SVM"
    HYBRID = "Hybrid"
    UNKNOWN = "Unknown"


@dataclass
class LayerFormula:
    """Mathematical formula for a single operator/layer."""
    operator_name: str
    op_type: str
    formula: str
    dimensions: str
    param_summary: str


@dataclass
class ModelReport:
    """Complete auto-detection report for an ONNX model."""
    architecture: ArchitectureType
    confidence: float
    layer_count: int
    total_params: int
    total_flops: int
    formulas: list[LayerFormula]
    architecture_pattern: str
    summary: str


_OP_FORMULAS: dict[str, str] = {
    "MatMul": "Y = X · W",
    "Gemm": "Y = α·(X · W) + β·B",
    "Conv": "Y = Conv(X, W, b; stride, padding, dilation)",
    "ConvTranspose": "Y = ConvTranspose(X, W, b; stride, padding)",
    "Add": "Y = A + B",
    "Sub": "Y = A - B",
    "Mul": "Y = A ⊙ B",
    "Div": "Y = A / B",
    "Relu": "Y = max(0, X)",
    "LeakyRelu": "Y = X if X > 0 else α·X",
    "Sigmoid": "Y = 1 / (1 + exp(-X))",
    "Tanh": "Y = tanh(X)",
    "Gelu": "Y = X · Φ(X)  where Φ is CDF of N(0,1)",
    "Selu": "Y = λ·(max(0,X) + min(0, α·(exp(X)-1)))",
    "Elu": "Y = X if X > 0 else α·(exp(X)-1)",
    "Softmax": "Y_i = exp(X_i) / Σ_j exp(X_j)",
    "LogSoftmax": "Y_i = X_i - log(Σ_j exp(X_j))",
    "BatchNormalization": "Y = γ · (X - μ) / √(σ² + ε) + β",
    "LayerNormalization": "Y = γ · (X - μ) / √(σ² + ε) + β",
    "InstanceNormalization": "Y = γ · (X - μ_inst) / √(σ²_inst + ε) + β",
    "GroupNormalization": "Y = γ · (X - μ_grp) / √(σ²_grp + ε) + β",
    "AveragePool": "Y[i,j] = mean(X[i·s:i·s+k, j·s:j·s+k])",
    "MaxPool": "Y[i,j] = max(X[i·s:i·s+k, j·s:j·s+k])",
    "GlobalAveragePool": "Y = mean(X, axis=[H, W])",
    "GlobalMaxPool": "Y = max(X, axis=[H, W])",
    "Flatten": "Y = reshape(X, [batch, -1])",
    "Reshape": "Y = reshape(X, target_shape)",
    "Transpose": "Y = permute(X, perm)",
    "Concat": "Y = concat([X₁, X₂, ...], axis)",
    "Gather": "Y = X[indices]  (embedding lookup)",
    "Dropout": "Y = X · mask / (1-p)  (training only)",
    "Clip": "Y = clip(X, min, max)",
    "Cast": "Y = cast(X, target_dtype)",
    "ReduceMean": "Y = mean(X, axes)",
    "ReduceSum": "Y = sum(X, axes)",
    "Sqrt": "Y = √X",
    "Pow": "Y = X^p",
    "Exp": "Y = exp(X)",
    "Log": "Y = log(X)",
    "Abs": "Y = |X|",
    "Neg": "Y = -X",
    "Reciprocal": "Y = 1/X",
    "Where": "Y = cond ? A : B",
    "TreeEnsembleClassifier": "Y = Σ_t Tree_t(X)  (weighted vote of T decision trees)",
    "TreeEnsembleRegressor": "Y = (1/T) · Σ_t Tree_t(X)  (mean of T regression trees)",
    "LinearClassifier": "Y = softmax(X · W + b)",
    "LinearRegressor": "Y = X · W + b",
    "SVMClassifier": "Y = sign(Σ_i α_i · K(x_i, X) + b)",
    "SVMRegressor": "Y = Σ_i α_i · K(x_i, X) + b",
    "Normalizer": "Y = X / ||X||_p",
    "Scaler": "Y = (X - offset) · scale",
    "ZipMap": "Y = {class_i: prob_i for i in classes}",
    "ArgMax": "Y = argmax(X, axis)",
}


class ModelDetector:
    """Detects model architecture and extracts per-layer mathematical formulas."""

    def __init__(self) -> None:
        self._analyzer = OnnxAnalyzer()

    def detect(self, model: onnx.ModelProto | str) -> ModelReport:
        if isinstance(model, str):
            model = onnx.load(model)

        graph_info = self._analyzer.analyze(model)
        arch = self._classify_architecture(graph_info)
        formulas = self._extract_formulas(model, graph_info)
        pattern = self._describe_pattern(graph_info)
        summary = self._generate_summary(graph_info, arch, formulas)

        return ModelReport(
            architecture=arch[0],
            confidence=arch[1],
            layer_count=len(graph_info.operators),
            total_params=graph_info.total_params,
            total_flops=graph_info.total_flops,
            formulas=formulas,
            architecture_pattern=pattern,
            summary=summary,
        )

    def _classify_architecture(self, info: GraphInfo) -> tuple[ArchitectureType, float]:
        """Classify architecture by analyzing operator composition."""
        op_types = [op.op_type for op in info.operators]
        categories = [op.category for op in info.operators]

        has_conv = any(t in ("Conv", "ConvTranspose") for t in op_types)
        has_matmul = any(t in ("MatMul", "Gemm") for t in op_types)
        has_rnn = any(t in ("LSTM", "GRU", "RNN") for t in op_types)
        has_attention = self._detect_attention_pattern(op_types)
        has_tree = OperatorCategory.TREE_ENSEMBLE in categories
        has_linear_ml = OperatorCategory.LINEAR_MODEL in categories
        has_svm = OperatorCategory.SVM in categories

        # Count model-type indicators
        neural_ops = sum(1 for c in categories if c in (
            OperatorCategory.DENSE_COMPUTE, OperatorCategory.NORMALIZATION,
            OperatorCategory.ACTIVATION))
        ml_ops = sum(1 for c in categories if c in (
            OperatorCategory.TREE_ENSEMBLE, OperatorCategory.LINEAR_MODEL,
            OperatorCategory.SVM))

        if ml_ops > 0 and neural_ops > 0:
            return (ArchitectureType.HYBRID, 0.8)

        if has_tree:
            return (ArchitectureType.TREE_ENSEMBLE, 0.95)
        if has_svm:
            return (ArchitectureType.SVM, 0.95)
        if has_linear_ml:
            return (ArchitectureType.LINEAR_MODEL, 0.95)

        if has_rnn:
            return (ArchitectureType.RNN, 0.9)
        if has_attention:
            return (ArchitectureType.TRANSFORMER, 0.85)
        if has_conv:
            return (ArchitectureType.CNN, 0.9)
        if has_matmul and not has_conv:
            return (ArchitectureType.MLP, 0.85)

        return (ArchitectureType.UNKNOWN, 0.3)

    def _detect_attention_pattern(self, op_types: list[str]) -> bool:
        """Detect Q·K^T·V attention pattern (MatMul → Softmax → MatMul)."""
        for i in range(len(op_types) - 2):
            if (op_types[i] == "MatMul" and
                "Softmax" in op_types[i:i+5] and
                op_types[i:i+5].count("MatMul") >= 2):
                return True
        return False

    def _extract_formulas(
        self, model: onnx.ModelProto, info: GraphInfo,
    ) -> list[LayerFormula]:
        formulas = []
        initializer_map = {init.name: numpy_helper.to_array(init)
                           for init in model.graph.initializer}

        for op in info.operators:
            formula_template = _OP_FORMULAS.get(op.op_type, f"Y = {op.op_type}(X)")
            dims = self._describe_dimensions(op)
            params = self._describe_params(op, initializer_map)

            formulas.append(LayerFormula(
                operator_name=op.name,
                op_type=op.op_type,
                formula=formula_template,
                dimensions=dims,
                param_summary=params,
            ))

        return formulas

    def _describe_dimensions(self, op: OperatorInfo) -> str:
        parts = []
        for i, shape in enumerate(op.input_shapes):
            if shape:
                dim_str = "×".join(str(d) if d > 0 else "N" for d in shape)
                parts.append(f"in{i}=[{dim_str}]")
        for i, shape in enumerate(op.output_shapes):
            if shape:
                dim_str = "×".join(str(d) if d > 0 else "N" for d in shape)
                parts.append(f"out{i}=[{dim_str}]")
        return ", ".join(parts) if parts else "shapes unknown"

    def _describe_params(
        self, op: OperatorInfo, initializer_map: dict[str, np.ndarray],
    ) -> str:
        if op.param_count == 0 and not op.attributes:
            return "no parameters"

        parts = []
        if op.param_count > 0:
            parts.append(f"{op.param_count:,} params")

        for inp in op.input_names:
            if inp in initializer_map:
                arr = initializer_map[inp]
                parts.append(f"{inp}: shape={list(arr.shape)}, dtype={arr.dtype}")

        if op.category == OperatorCategory.TREE_ENSEMBLE:
            if "nodes_treeids" in op.attributes:
                tree_ids = op.attributes["nodes_treeids"]
                n_trees = len(set(tree_ids)) if tree_ids else 0
                n_nodes = len(tree_ids) if tree_ids else 0
                parts.append(f"{n_trees} trees, {n_nodes} nodes")

        return "; ".join(parts) if parts else "no parameters"

    def _describe_pattern(self, info: GraphInfo) -> str:
        """Describe the architectural pattern as a sequence."""
        pattern_parts = []
        for op in info.operators:
            if op.category in (OperatorCategory.DATA_MOVEMENT, OperatorCategory.PREPROCESSING,
                               OperatorCategory.OTHER):
                continue
            if op.op_type in ("Add", "Sub", "Cast", "Reshape", "Flatten", "Squeeze", "Unsqueeze"):
                continue
            pattern_parts.append(op.op_type)
        return " → ".join(pattern_parts) if pattern_parts else "empty"

    def _generate_summary(
        self,
        info: GraphInfo,
        arch: tuple[ArchitectureType, float],
        formulas: list[LayerFormula],
    ) -> str:
        lines = [
            f"Architecture: {arch[0].value} (confidence: {arch[1]:.0%})",
            f"Operators: {len(info.operators)} total, {len(info.compressible_operators)} compressible",
            f"Parameters: {info.total_params:,}",
            f"Estimated FLOPs: {info.total_flops:,}",
            "",
            "Layer-by-layer formulas:",
        ]
        for f in formulas:
            lines.append(f"  [{f.operator_name}] {f.formula}")
            lines.append(f"    dims: {f.dimensions}")
            if f.param_summary != "no parameters":
                lines.append(f"    params: {f.param_summary}")
        return "\n".join(lines)
