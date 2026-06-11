"""ONNX Graph Analyzer: parses ONNX models into operator-level representations
for UOSA sensitivity analysis and compression action assignment."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


class OperatorCategory(enum.Enum):
    DENSE_COMPUTE = "dense_compute"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    POOLING = "pooling"
    EMBEDDING = "embedding"
    TREE_ENSEMBLE = "tree_ensemble"
    LINEAR_MODEL = "linear_model"
    SVM = "svm"
    PREPROCESSING = "preprocessing"
    DATA_MOVEMENT = "data_movement"
    OTHER = "other"


_CATEGORY_MAP: dict[str, OperatorCategory] = {
    "Conv": OperatorCategory.DENSE_COMPUTE,
    "ConvTranspose": OperatorCategory.DENSE_COMPUTE,
    "MatMul": OperatorCategory.DENSE_COMPUTE,
    "Gemm": OperatorCategory.DENSE_COMPUTE,
    "MatMulInteger": OperatorCategory.DENSE_COMPUTE,
    "ConvInteger": OperatorCategory.DENSE_COMPUTE,
    "BatchNormalization": OperatorCategory.NORMALIZATION,
    "LayerNormalization": OperatorCategory.NORMALIZATION,
    "InstanceNormalization": OperatorCategory.NORMALIZATION,
    "GroupNormalization": OperatorCategory.NORMALIZATION,
    "Relu": OperatorCategory.ACTIVATION,
    "LeakyRelu": OperatorCategory.ACTIVATION,
    "Sigmoid": OperatorCategory.ACTIVATION,
    "Tanh": OperatorCategory.ACTIVATION,
    "Gelu": OperatorCategory.ACTIVATION,
    "Selu": OperatorCategory.ACTIVATION,
    "Softmax": OperatorCategory.ACTIVATION,
    "Elu": OperatorCategory.ACTIVATION,
    "AveragePool": OperatorCategory.POOLING,
    "MaxPool": OperatorCategory.POOLING,
    "GlobalAveragePool": OperatorCategory.POOLING,
    "GlobalMaxPool": OperatorCategory.POOLING,
    "Gather": OperatorCategory.EMBEDDING,
    "GatherElements": OperatorCategory.EMBEDDING,
    "Reshape": OperatorCategory.DATA_MOVEMENT,
    "Transpose": OperatorCategory.DATA_MOVEMENT,
    "Concat": OperatorCategory.DATA_MOVEMENT,
    "Split": OperatorCategory.DATA_MOVEMENT,
    "Slice": OperatorCategory.DATA_MOVEMENT,
    "Pad": OperatorCategory.DATA_MOVEMENT,
    "Flatten": OperatorCategory.DATA_MOVEMENT,
    "Squeeze": OperatorCategory.DATA_MOVEMENT,
    "Unsqueeze": OperatorCategory.DATA_MOVEMENT,
    "TreeEnsembleClassifier": OperatorCategory.TREE_ENSEMBLE,
    "TreeEnsembleRegressor": OperatorCategory.TREE_ENSEMBLE,
    "TreeEnsemble": OperatorCategory.TREE_ENSEMBLE,
    "LinearClassifier": OperatorCategory.LINEAR_MODEL,
    "LinearRegressor": OperatorCategory.LINEAR_MODEL,
    "SVMClassifier": OperatorCategory.SVM,
    "SVMRegressor": OperatorCategory.SVM,
    "Normalizer": OperatorCategory.PREPROCESSING,
    "Scaler": OperatorCategory.PREPROCESSING,
    "OneHotEncoder": OperatorCategory.PREPROCESSING,
    "LabelEncoder": OperatorCategory.PREPROCESSING,
    "Binarizer": OperatorCategory.PREPROCESSING,
    "Imputer": OperatorCategory.PREPROCESSING,
    "ZipMap": OperatorCategory.PREPROCESSING,
}

COMPRESSIBLE_CATEGORIES = frozenset({
    OperatorCategory.DENSE_COMPUTE,
    OperatorCategory.NORMALIZATION,
    OperatorCategory.ACTIVATION,
    OperatorCategory.EMBEDDING,
    OperatorCategory.TREE_ENSEMBLE,
    OperatorCategory.LINEAR_MODEL,
    OperatorCategory.SVM,
})


@dataclass
class OperatorInfo:
    """Metadata for a single ONNX operator node."""
    name: str
    op_type: str
    domain: str
    category: OperatorCategory
    input_names: list[str]
    output_names: list[str]
    input_shapes: list[list[int] | None]
    output_shapes: list[list[int] | None]
    param_count: int
    flops_estimate: int
    attributes: dict[str, Any] = field(default_factory=dict)
    is_compressible: bool = False
    topo_index: int = -1

    @property
    def is_ml_operator(self) -> bool:
        return self.domain == "ai.onnx.ml"


@dataclass
class GraphInfo:
    """Analyzed ONNX graph representation."""
    model_path: str | None
    operators: list[OperatorInfo]
    adjacency: dict[str, list[str]]  # node_name -> [successor node_names]
    topo_order: list[str]
    input_names: list[str]
    output_names: list[str]
    total_params: int
    total_flops: int

    @property
    def compressible_operators(self) -> list[OperatorInfo]:
        return [op for op in self.operators if op.is_compressible]

    def get_operator(self, name: str) -> OperatorInfo | None:
        for op in self.operators:
            if op.name == name:
                return op
        return None


class OnnxAnalyzer:
    """Parses ONNX models and extracts operator-level information for UOSA."""

    def __init__(self) -> None:
        self._initializer_map: dict[str, np.ndarray] = {}
        self._shape_map: dict[str, list[int] | None] = {}
        self._producer_map: dict[str, str] = {}

    def analyze(self, model_or_path: onnx.ModelProto | str) -> GraphInfo:
        if isinstance(model_or_path, str):
            model = onnx.load(model_or_path)
            model_path = model_or_path
        else:
            model = model_or_path
            model_path = None

        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass

        graph = model.graph
        self._build_initializer_map(graph)
        self._build_shape_map(graph)
        self._build_producer_map(graph)

        operators = []
        for i, node in enumerate(graph.node):
            op_info = self._analyze_node(node, i)
            operators.append(op_info)

        adjacency = self._build_adjacency(graph)
        topo_order = self._topological_sort(operators, adjacency)
        for idx, name in enumerate(topo_order):
            for op in operators:
                if op.name == name:
                    op.topo_index = idx
                    break

        input_names = [inp.name for inp in graph.input
                       if inp.name not in self._initializer_map]
        output_names = [out.name for out in graph.output]
        total_params = sum(op.param_count for op in operators)
        total_flops = sum(op.flops_estimate for op in operators)

        return GraphInfo(
            model_path=model_path,
            operators=operators,
            adjacency=adjacency,
            topo_order=topo_order,
            input_names=input_names,
            output_names=output_names,
            total_params=total_params,
            total_flops=total_flops,
        )

    def _build_initializer_map(self, graph: onnx.GraphProto) -> None:
        self._initializer_map = {}
        for init in graph.initializer:
            self._initializer_map[init.name] = numpy_helper.to_array(init)

    def _build_shape_map(self, graph: onnx.GraphProto) -> None:
        self._shape_map = {}
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            shape = self._extract_shape(vi)
            if shape is not None:
                self._shape_map[vi.name] = shape

    def _build_producer_map(self, graph: onnx.GraphProto) -> None:
        self._producer_map = {}
        for node in graph.node:
            node_name = node.name or f"{node.op_type}_{id(node)}"
            for output in node.output:
                if output:
                    self._producer_map[output] = node_name

    def _extract_shape(self, value_info: onnx.ValueInfoProto) -> list[int] | None:
        try:
            tensor_type = value_info.type.tensor_type
            if not tensor_type.HasField("shape"):
                return None
            dims = []
            for dim in tensor_type.shape.dim:
                if dim.dim_param:
                    dims.append(-1)
                else:
                    dims.append(dim.dim_value)
            return dims
        except Exception:
            return None

    def _analyze_node(self, node: onnx.NodeProto, index: int) -> OperatorInfo:
        name = node.name or f"{node.op_type}_{index}"
        domain = node.domain or "ai.onnx"
        if domain == "":
            domain = "ai.onnx"

        category = self._classify_operator(node.op_type, domain)

        input_shapes = [self._shape_map.get(inp) for inp in node.input]
        output_shapes = [self._shape_map.get(out) for out in node.output]

        param_count = self._count_params(node)
        flops_estimate = self._estimate_flops(node, input_shapes, output_shapes)

        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = self._extract_attribute_value(attr)

        return OperatorInfo(
            name=name,
            op_type=node.op_type,
            domain=domain,
            category=category,
            input_names=list(node.input),
            output_names=list(node.output),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            param_count=param_count,
            flops_estimate=flops_estimate,
            attributes=attrs,
            is_compressible=category in COMPRESSIBLE_CATEGORIES,
        )

    def _classify_operator(self, op_type: str, domain: str) -> OperatorCategory:
        if op_type in _CATEGORY_MAP:
            return _CATEGORY_MAP[op_type]
        if domain == "ai.onnx.ml":
            if "Tree" in op_type:
                return OperatorCategory.TREE_ENSEMBLE
            if "Linear" in op_type:
                return OperatorCategory.LINEAR_MODEL
            if "SVM" in op_type:
                return OperatorCategory.SVM
            return OperatorCategory.PREPROCESSING
        return OperatorCategory.OTHER

    def _count_params(self, node: onnx.NodeProto) -> int:
        total = 0
        for inp in node.input:
            if inp in self._initializer_map:
                total += self._initializer_map[inp].size
        return total

    def _estimate_flops(
        self,
        node: onnx.NodeProto,
        input_shapes: list[list[int] | None],
        output_shapes: list[list[int] | None],
    ) -> int:
        op = node.op_type
        if op in ("Conv", "ConvTranspose"):
            return self._conv_flops(node, input_shapes, output_shapes)
        if op in ("MatMul", "Gemm"):
            return self._matmul_flops(node, input_shapes)
        if op in ("BatchNormalization", "LayerNormalization"):
            if output_shapes and output_shapes[0]:
                return int(np.prod([max(d, 1) for d in output_shapes[0]])) * 4
        return 0

    def _conv_flops(
        self,
        node: onnx.NodeProto,
        input_shapes: list[list[int] | None],
        output_shapes: list[list[int] | None],
    ) -> int:
        if not output_shapes or not output_shapes[0]:
            return 0
        weight_name = node.input[1] if len(node.input) > 1 else None
        if not weight_name or weight_name not in self._initializer_map:
            return 0
        w = self._initializer_map[weight_name]
        # FLOPs ≈ 2 * output_spatial * kernel_size * C_in * C_out
        output_shape = output_shapes[0]
        output_spatial = int(np.prod([max(d, 1) for d in output_shape[2:]]))
        kernel_size = int(np.prod(w.shape[2:]))
        c_in = w.shape[1]
        c_out = w.shape[0]
        group = 1
        for attr in node.attribute:
            if attr.name == "group":
                group = attr.i
        return 2 * output_spatial * kernel_size * (c_in // group) * c_out

    def _matmul_flops(
        self,
        node: onnx.NodeProto,
        input_shapes: list[list[int] | None],
    ) -> int:
        if len(input_shapes) < 2:
            return 0
        a_shape = input_shapes[0]
        b_shape = input_shapes[1]
        if not a_shape or not b_shape:
            return 0
        if len(a_shape) < 2 or len(b_shape) < 2:
            return 0
        m = max(a_shape[-2], 1)
        k = max(a_shape[-1], 1)
        n = max(b_shape[-1], 1)
        batch = int(np.prod([max(d, 1) for d in a_shape[:-2]])) if len(a_shape) > 2 else 1
        return 2 * batch * m * k * n

    def _extract_attribute_value(self, attr: onnx.AttributeProto) -> Any:
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        if attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
        if attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        if attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        if attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode("utf-8") if isinstance(s, bytes) else s for s in attr.strings]
        return None

    def _build_adjacency(self, graph: onnx.GraphProto) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = {}
        output_to_node: dict[str, str] = {}
        node_names: list[str] = []

        for i, node in enumerate(graph.node):
            name = node.name or f"{node.op_type}_{i}"
            node_names.append(name)
            adjacency[name] = []
            for out in node.output:
                if out:
                    output_to_node[out] = name

        for i, node in enumerate(graph.node):
            name = node.name or f"{node.op_type}_{i}"
            for inp in node.input:
                if inp in output_to_node:
                    producer = output_to_node[inp]
                    if name not in adjacency[producer]:
                        adjacency[producer].append(name)

        return adjacency

    def _topological_sort(
        self,
        operators: list[OperatorInfo],
        adjacency: dict[str, list[str]],
    ) -> list[str]:
        in_degree: dict[str, int] = {op.name: 0 for op in operators}
        for node, successors in adjacency.items():
            for succ in successors:
                if succ in in_degree:
                    in_degree[succ] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for succ in adjacency.get(node, []):
                if succ in in_degree:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)

        return result


def initializer_bytes(model: onnx.ModelProto) -> int:
    """Total stored bytes of all weight initializers (weights-only size).

    Unlike ``ModelProto.ByteSize()`` this excludes graph structure and
    ai.onnx.ml attribute payloads, so it isolates the part of the file that
    weight quantization can actually shrink.
    """
    total = 0
    for init in model.graph.initializer:
        if init.raw_data:
            total += len(init.raw_data)
        else:
            try:
                total += numpy_helper.to_array(init).nbytes
            except Exception:
                continue
    return total
