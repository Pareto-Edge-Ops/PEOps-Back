"""ONNX Graph Transformer: applies compression actions at the ONNX graph level.

Handles graph-level operations: node insertion/deletion, initializer modification,
shape propagation. Works as the execution layer for ConcreteActions from the
ActionTranslator.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from astra.core.compression_actions import ConcreteAction
from astra.graph.onnx_analyzer import OperatorCategory


# Execution order for actions within a single apply() pass. Quantization must
# run last: earlier phases (fusion/pruning/SVD) look up weights by their
# original names and need full-precision values to operate on.
_ACTION_PHASE: dict[str, int] = {
    "bn_fusion": 0,
    "channel_pruning": 1,
    "tree_pruning": 1,
    "feature_pruning": 1,
    "support_vector_pruning": 1,
    "depth_limiting": 1,
    "low_rank_svd": 2,
    "leaf_merging": 2,
    "quantization": 3,
    "param_quantization": 3,
    "embedding_quantization": 3,
    "coefficient_quantization": 3,
    "leaf_quantization": 3,
}

# Initializers smaller than this are left untouched by weight quantization:
# the byte payoff is negligible and the graph noise (extra Cast/DQ nodes) isn't.
_MIN_QUANT_BYTES = 1024


class OnnxTransformer:
    """Applies compression transformations to ONNX models.

    Weight quantization stores REAL reduced-precision bytes: FP16 weights are
    kept as float16 initializers behind a Cast node, INT8/INT4 weights as int8
    initializers behind a DequantizeLinear node. File size genuinely shrinks
    (~2x / ~4x for the converted tensors) while every consumer node keeps its
    original input name, so the graph stays runnable by onnxruntime (which
    constant-folds the Cast/DequantizeLinear at session load).
    """

    def apply(self, model: onnx.ModelProto, actions: list[ConcreteAction]) -> onnx.ModelProto:
        result = onnx.ModelProto()
        result.CopyFrom(model)

        # Shared initializers must be converted at most once per pass.
        self._converted_inits: set[str] = set()
        self._int8_unavailable = False

        # DequantizeLinear needs opset >= 10 (>= 13 for per-axis). Old models
        # are upgraded once; if the version converter can't handle the graph,
        # int8 actions degrade to fp16 storage (still real 2x bytes).
        wants_int8 = any(
            a.parameters.get("target_dtype") in ("INT8", "INT4")
            for a in actions
            if a.action_type in ("quantization", "param_quantization", "embedding_quantization")
        )
        if wants_int8 and self._model_opset(result) < 10:
            try:
                result = onnx.version_converter.convert_version(result, 13)
            except Exception:
                self._int8_unavailable = True

        ordered = sorted(actions, key=lambda a: _ACTION_PHASE.get(a.action_type, 2))
        for action in ordered:
            handler = self._get_handler(action)
            if handler is not None:
                result = handler(result, action)

        try:
            result = onnx.shape_inference.infer_shapes(result)
        except Exception:
            pass

        return result

    def _get_handler(self, action: ConcreteAction):
        handlers = {
            "quantization": self._apply_weight_quantization,
            "channel_pruning": self._apply_channel_pruning,
            "low_rank_svd": self._apply_low_rank_svd,
            "bn_fusion": self._apply_bn_fusion,
            "param_quantization": self._apply_weight_quantization,
            "embedding_quantization": self._apply_weight_quantization,
            "coefficient_quantization": self._apply_coefficient_quantization,
            "leaf_quantization": self._apply_leaf_quantization,
            "tree_pruning": self._apply_tree_pruning,
            "leaf_merging": self._apply_leaf_merging,
            "depth_limiting": self._apply_depth_limiting,
            "feature_pruning": self._apply_feature_pruning,
            "support_vector_pruning": self._apply_sv_pruning,
        }
        return handlers.get(action.action_type)

    # --- Neural Network Compression ---

    def _apply_weight_quantization(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Quantize weight initializers with REAL reduced-precision storage.

        FP16: float32 initializer -> float16 initializer `{name}_fp16` + a
        Cast(to=FLOAT) node that re-exposes the original tensor name.
        INT8:  -> int8 initializer `{name}_q` + scale/zero-point + a
        DequantizeLinear node re-exposing the original name. Per-channel
        (axis=0) for 4-D Conv weights when the opset allows, per-tensor
        otherwise.
        INT4: stored as int8 restricted to 16 levels (sub-byte packing needs
        opset >= 21, which the supported model zoo predates) — an honest 4x,
        never advertised as 8x.

        Biases and other 1-D tensors are excluded from integer quantization
        (disproportionate accuracy damage for negligible bytes); FP16 applies
        to any float32 tensor above the size floor.
        """
        target_dtype = action.parameters.get("target_dtype", "FP16")
        if target_dtype not in ("FP16", "INT8", "INT4"):
            return model
        node = self._find_node(model, action.operator_name)
        if node is None:
            return model

        for inp_name in list(node.input):
            if inp_name in self._converted_inits:
                continue
            init = self._find_initializer(model, inp_name)
            if init is None:
                continue
            if init.data_type != TensorProto.FLOAT:
                continue  # already reduced or non-float (e.g. int indices)
            arr = numpy_helper.to_array(init)
            if arr.nbytes < _MIN_QUANT_BYTES:
                continue
            if target_dtype in ("INT8", "INT4") and arr.ndim < 2:
                continue

            if target_dtype == "FP16":
                self._store_fp16(model, init, arr)
            else:
                self._store_int8(model, init, arr, target_dtype, node)
            self._converted_inits.add(inp_name)

        return model

    def _store_fp16(
        self, model: onnx.ModelProto, init: onnx.TensorProto, arr: np.ndarray,
    ) -> None:
        """Replace a float32 initializer with float16 storage + Cast node."""
        fp16_name = self._unique_name(model, f"{init.name}_fp16")
        fp16_init = numpy_helper.from_array(arr.astype(np.float16), name=fp16_name)

        cast = onnx.helper.make_node(
            "Cast",
            inputs=[fp16_name],
            outputs=[init.name],
            name=self._unique_name(model, f"{init.name}_fp16_cast"),
            to=TensorProto.FLOAT,
        )
        self._remove_initializer(model, init.name)
        model.graph.initializer.append(fp16_init)
        model.graph.node.insert(0, cast)

    def _store_int8(
        self,
        model: onnx.ModelProto,
        init: onnx.TensorProto,
        arr: np.ndarray,
        target_dtype: str,
        consumer: onnx.NodeProto,
    ) -> None:
        """Replace a float32 initializer with int8 storage + DequantizeLinear."""
        if self._int8_unavailable or self._model_opset(model) < 10:
            # DequantizeLinear unavailable at this opset: fp16 storage is the
            # best real-bytes reduction that stays valid (still 2x).
            self._store_fp16(model, init, arr)
            return

        qmax = 127.0 if target_dtype == "INT8" else 7.0

        per_channel = (
            consumer.op_type == "Conv"
            and arr.ndim == 4
            and self._model_opset(model) >= 13
            and arr.shape[0] > 1
        )
        if per_channel:
            axes = tuple(range(1, arr.ndim))
            scale = np.abs(arr).max(axis=axes) / qmax + 1e-12  # [C_out]
            q = np.round(arr / scale.reshape(-1, *([1] * (arr.ndim - 1))))
            scale_arr = scale.astype(np.float32)
            zp_arr = np.zeros(arr.shape[0], dtype=np.int8)
        else:
            scale = np.abs(arr).max() / qmax + 1e-12
            q = np.round(arr / scale)
            scale_arr = np.float32(scale)
            zp_arr = np.int8(0)

        q = q.clip(-qmax, qmax).astype(np.int8)

        q_name = self._unique_name(model, f"{init.name}_q")
        scale_name = self._unique_name(model, f"{init.name}_scale")
        zp_name = self._unique_name(model, f"{init.name}_zp")

        dq_kwargs: dict[str, Any] = {}
        if per_channel:
            dq_kwargs["axis"] = 0
        dq = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=[q_name, scale_name, zp_name],
            outputs=[init.name],
            name=self._unique_name(model, f"{init.name}_dq"),
            **dq_kwargs,
        )

        self._remove_initializer(model, init.name)
        model.graph.initializer.append(numpy_helper.from_array(q, name=q_name))
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(scale_arr, dtype=np.float32), name=scale_name))
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(zp_arr, dtype=np.int8), name=zp_name))
        model.graph.node.insert(0, dq)

    def _apply_channel_pruning(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Structured channel pruning: remove output channels with smallest L1 norm.

        For Conv [C_out, C_in, H, W]: prunes along axis=0 (output channels).
        For MatMul [M, N]: prunes along axis=1 (output dimension).
        """
        ratio = action.parameters.get("ratio", 0.0)
        if ratio <= 0:
            return model

        node = self._find_node(model, action.operator_name)
        if node is None or len(node.input) < 2:
            return model

        weight_init = self._find_initializer(model, node.input[1])
        if weight_init is None:
            return model

        weight = numpy_helper.to_array(weight_init)

        if node.op_type in ("MatMul",) and weight.ndim == 2:
            # MatMul: weight is [input_dim, output_dim], prune output dim (axis=1)
            n_channels = weight.shape[1]
            n_keep = max(1, int(n_channels * (1 - ratio)))
            channel_importance = np.abs(weight).sum(axis=0)
            keep_indices = np.argsort(channel_importance)[-n_keep:]
            keep_indices.sort()
            pruned_weight = weight[:, keep_indices]
        else:
            # Conv/Gemm: weight is [C_out, ...], prune output channels (axis=0)
            n_channels = weight.shape[0]
            n_keep = max(1, int(n_channels * (1 - ratio)))
            channel_importance = np.abs(weight.reshape(n_channels, -1)).sum(axis=1)
            keep_indices = np.argsort(channel_importance)[-n_keep:]
            keep_indices.sort()
            pruned_weight = weight[keep_indices]

        new_init = numpy_helper.from_array(pruned_weight, name=weight_init.name)
        self._replace_initializer(model, weight_init.name, new_init)

        if len(node.input) > 2:
            bias_init = self._find_initializer(model, node.input[2])
            if bias_init is not None:
                bias = numpy_helper.to_array(bias_init)
                if len(bias) == n_channels:
                    pruned_bias = bias[keep_indices]
                    new_bias = numpy_helper.from_array(pruned_bias, name=bias_init.name)
                    self._replace_initializer(model, bias_init.name, new_bias)

        return model

    def _apply_low_rank_svd(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Low-rank SVD decomposition of weight matrices."""
        rank = action.parameters.get("rank", 8)
        node = self._find_node(model, action.operator_name)
        if node is None or len(node.input) < 2:
            return model

        weight_init = self._find_initializer(model, node.input[1])
        if weight_init is None:
            return model

        weight = numpy_helper.to_array(weight_init)
        if weight.ndim != 2:
            return model

        rank = min(rank, min(weight.shape))
        U, S, Vt = np.linalg.svd(weight, full_matrices=False)
        U_r = (U[:, :rank] * S[:rank]).astype(np.float32)
        V_r = Vt[:rank, :].astype(np.float32)

        new_init = numpy_helper.from_array(U_r, name=weight_init.name)
        self._replace_initializer(model, weight_init.name, new_init)

        v_name = f"{weight_init.name}_svd_v"
        v_init = numpy_helper.from_array(V_r, name=v_name)
        model.graph.initializer.append(v_init)

        original_output = node.output[0]
        intermediate = f"{original_output}_svd_mid"
        node.output[0] = intermediate

        matmul2 = onnx.helper.make_node(
            "MatMul",
            inputs=[intermediate, v_name],
            outputs=[original_output],
            name=f"{action.operator_name}_svd_v",
        )

        nodes = list(model.graph.node)
        idx = nodes.index(node)
        model.graph.node.insert(idx + 1, matmul2)

        return model

    def _apply_bn_fusion(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Fuse BatchNormalization into preceding Conv/MatMul."""
        bn_node = self._find_node(model, action.operator_name)
        if bn_node is None or bn_node.op_type != "BatchNormalization":
            return model

        bn_input = bn_node.input[0]
        prev_node = None
        for node in model.graph.node:
            if bn_input in node.output:
                prev_node = node
                break

        if prev_node is None or prev_node.op_type not in ("Conv", "Gemm"):
            return model

        weight_init = self._find_initializer(model, prev_node.input[1])
        if weight_init is None:
            return model

        gamma = numpy_helper.to_array(self._find_initializer(model, bn_node.input[1]))
        beta = numpy_helper.to_array(self._find_initializer(model, bn_node.input[2]))
        mean = numpy_helper.to_array(self._find_initializer(model, bn_node.input[3]))
        var = numpy_helper.to_array(self._find_initializer(model, bn_node.input[4]))

        eps = 1e-5
        for attr in bn_node.attribute:
            if attr.name == "epsilon":
                eps = attr.f

        scale = gamma / np.sqrt(var + eps)
        weight = numpy_helper.to_array(weight_init)

        if weight.ndim == 4:  # Conv
            fused_weight = weight * scale.reshape(-1, 1, 1, 1)
        elif weight.ndim == 2:  # Gemm
            fused_weight = weight * scale.reshape(-1, 1)
        else:
            return model

        new_weight = numpy_helper.from_array(fused_weight.astype(np.float32), name=weight_init.name)
        self._replace_initializer(model, weight_init.name, new_weight)

        fused_bias = beta - mean * scale
        if len(prev_node.input) > 2:
            bias_init = self._find_initializer(model, prev_node.input[2])
            if bias_init is not None:
                old_bias = numpy_helper.to_array(bias_init)
                fused_bias = old_bias * scale + fused_bias
                new_bias = numpy_helper.from_array(fused_bias.astype(np.float32), name=bias_init.name)
                self._replace_initializer(model, bias_init.name, new_bias)
        else:
            bias_name = f"{prev_node.name}_fused_bias"
            bias_init = numpy_helper.from_array(fused_bias.astype(np.float32), name=bias_name)
            model.graph.initializer.append(bias_init)
            prev_node.input.append(bias_name)

        prev_node.output[0] = bn_node.output[0]
        model.graph.node.remove(bn_node)

        return model

    # --- Tree Ensemble Compression ---

    def _apply_tree_pruning(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Remove low-importance trees from a TreeEnsemble operator.

        Filters both node-level arrays (nodes_treeids, etc.) and leaf-level
        arrays (target_treeids, target_weights, etc.) to maintain consistency.
        """
        ratio = action.parameters.get("ratio", 0.0)
        if ratio <= 0:
            return model

        node = self._find_node_by_type(model, action.operator_name, action.op_type)
        if node is None:
            return model

        tree_ids_attr = self._get_node_attr(node, "nodes_treeids")
        if tree_ids_attr is None:
            return model

        tree_ids = np.array(list(tree_ids_attr.ints))
        unique_trees = np.unique(tree_ids)
        n_trees = len(unique_trees)
        n_keep = max(1, int(n_trees * (1 - ratio)))

        if n_keep >= n_trees:
            return model

        keep_trees = set(unique_trees[:n_keep].tolist())
        node_mask = np.isin(tree_ids, list(keep_trees))

        # Filter node-level arrays
        node_attrs_to_filter = [
            "nodes_treeids", "nodes_nodeids", "nodes_featureids",
            "nodes_values", "nodes_hitrates", "nodes_modes",
            "nodes_truenodeids", "nodes_falsenodeids",
            "nodes_missing_value_tracks_true",
        ]
        for attr_name in node_attrs_to_filter:
            self._filter_attr(node, attr_name, node_mask)

        # Filter target/leaf-level arrays (critical for consistency)
        target_tree_attr = self._get_node_attr(node, "target_treeids")
        if target_tree_attr is not None:
            target_tree_ids = np.array(list(target_tree_attr.ints))
            target_mask = np.isin(target_tree_ids, list(keep_trees))
            for attr_name in ["target_treeids", "target_nodeids", "target_ids", "target_weights"]:
                self._filter_attr(node, attr_name, target_mask)

        # Update class_treeids if present
        class_tree_attr = self._get_node_attr(node, "class_treeids")
        if class_tree_attr is not None:
            class_tree_ids = np.array(list(class_tree_attr.ints))
            class_mask = np.isin(class_tree_ids, list(keep_trees))
            for attr_name in ["class_treeids", "class_nodeids", "class_ids", "class_weights"]:
                self._filter_attr(node, attr_name, class_mask)

        return model

    def _filter_attr(
        self,
        node: onnx.NodeProto,
        attr_name: str,
        mask: np.ndarray,
    ) -> None:
        """Filter an attribute's values using a boolean mask."""
        attr = self._get_node_attr(node, attr_name)
        if attr is None:
            return
        if attr.type == onnx.AttributeProto.INTS:
            if len(attr.ints) != len(mask):
                return
            filtered = [v for v, m in zip(attr.ints, mask) if m]
            del attr.ints[:]
            attr.ints.extend(filtered)
        elif attr.type == onnx.AttributeProto.FLOATS:
            if len(attr.floats) != len(mask):
                return
            filtered = [v for v, m in zip(attr.floats, mask) if m]
            del attr.floats[:]
            attr.floats.extend(filtered)
        elif attr.type == onnx.AttributeProto.STRINGS:
            if len(attr.strings) != len(mask):
                return
            filtered = [v for v, m in zip(attr.strings, mask) if m]
            del attr.strings[:]
            attr.strings.extend(filtered)

    def _apply_leaf_quantization(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Quantize leaf values in TreeEnsemble to FP16.

        SIZE-NEUTRAL: ai.onnx.ml attributes are stored as protobuf FLOATS
        (fixed 4-byte storage), so this affects fidelity/robustness only —
        it can never shrink the file. Excluded from size accounting.
        """
        node = self._find_node_by_type(model, action.operator_name, action.op_type)
        if node is None:
            return model

        for attr_name in ["target_weights", "leaf_weights"]:
            attr = self._get_node_attr(node, attr_name)
            if attr is None or attr.type != onnx.AttributeProto.FLOATS:
                continue
            values = np.array(list(attr.floats), dtype=np.float32)
            quantized = values.astype(np.float16).astype(np.float32)
            del attr.floats[:]
            attr.floats.extend(quantized.tolist())

        return model

    def _apply_leaf_merging(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Merge leaves with similar values within each tree."""
        node = self._find_node_by_type(model, action.operator_name, action.op_type)
        if node is None:
            return model

        for attr_name in ["target_weights", "leaf_weights"]:
            attr = self._get_node_attr(node, attr_name)
            if attr is None or attr.type != onnx.AttributeProto.FLOATS:
                continue
            values = np.array(list(attr.floats), dtype=np.float32)
            if len(values) == 0:
                continue
            threshold = np.std(values) * 0.1 + 1e-8
            sorted_idx = np.argsort(values)
            merged = values.copy()
            i = 0
            while i < len(sorted_idx) - 1:
                j = i + 1
                while j < len(sorted_idx) and abs(values[sorted_idx[j]] - values[sorted_idx[i]]) < threshold:
                    j += 1
                group = sorted_idx[i:j]
                mean_val = values[group].mean()
                merged[group] = mean_val
                i = j
            del attr.floats[:]
            attr.floats.extend(merged.tolist())

        return model

    def _apply_depth_limiting(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Limit tree depth by converting deep internal nodes to leaves."""
        max_depth = action.parameters.get("max_depth", 5)
        node = self._find_node_by_type(model, action.operator_name, action.op_type)
        if node is None:
            return model

        modes_attr = self._get_node_attr(node, "nodes_modes")
        treeids_attr = self._get_node_attr(node, "nodes_treeids")
        nodeids_attr = self._get_node_attr(node, "nodes_nodeids")

        if modes_attr is None or treeids_attr is None or nodeids_attr is None:
            return model

        tree_ids = list(treeids_attr.ints)
        node_ids = list(nodeids_attr.ints)
        modes = list(modes_attr.strings)

        depths = self._compute_node_depths(tree_ids, node_ids, node)

        for i in range(len(modes)):
            if depths[i] >= max_depth and modes[i] != b"LEAF":
                modes[i] = b"LEAF"

        del modes_attr.strings[:]
        modes_attr.strings.extend(modes)

        return model

    def _compute_node_depths(
        self,
        tree_ids: list[int],
        node_ids: list[int],
        node: onnx.NodeProto,
    ) -> list[int]:
        """Compute depth of each node in the tree structure."""
        true_ids_attr = self._get_node_attr(node, "nodes_truenodeids")
        false_ids_attr = self._get_node_attr(node, "nodes_falsenodeids")

        if true_ids_attr is None or false_ids_attr is None:
            return [0] * len(tree_ids)

        true_ids = list(true_ids_attr.ints)
        false_ids = list(false_ids_attr.ints)

        depths = [0] * len(tree_ids)
        for tree_id in set(tree_ids):
            indices = [i for i, t in enumerate(tree_ids) if t == tree_id]
            if not indices:
                continue

            node_to_idx = {}
            for idx in indices:
                node_to_idx[node_ids[idx]] = idx

            root_idx = indices[0]
            queue = [(root_idx, 0)]
            while queue:
                curr_idx, depth = queue.pop(0)
                depths[curr_idx] = depth
                if curr_idx < len(true_ids):
                    true_child = true_ids[curr_idx]
                    false_child = false_ids[curr_idx]
                    if true_child in node_to_idx:
                        queue.append((node_to_idx[true_child], depth + 1))
                    if false_child in node_to_idx:
                        queue.append((node_to_idx[false_child], depth + 1))

        return depths

    # --- Classical ML Compression ---

    def _apply_coefficient_quantization(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Quantize coefficients in LinearClassifier/LinearRegressor.

        SIZE-NEUTRAL: protobuf FLOATS attributes keep 4-byte storage regardless
        of value precision; this is a fidelity/accuracy trade-off knob only.
        """
        node = self._find_node_by_type(model, action.operator_name, action.op_type)
        if node is None:
            return model

        attr = self._get_node_attr(node, "coefficients")
        if attr is None or attr.type != onnx.AttributeProto.FLOATS:
            return model

        values = np.array(list(attr.floats), dtype=np.float32)
        target_dtype = action.parameters.get("target_dtype", "FP16")

        if target_dtype == "FP16":
            quantized = values.astype(np.float16).astype(np.float32)
        elif target_dtype == "INT8":
            scale = np.abs(values).max() / 127.0 + 1e-12
            quantized = np.round(values / scale).clip(-127, 127) * scale
            quantized = quantized.astype(np.float32)
        else:
            return model

        del attr.floats[:]
        attr.floats.extend(quantized.tolist())
        return model

    def _apply_feature_pruning(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Zero out small coefficients in LinearClassifier/LinearRegressor."""
        ratio = action.parameters.get("ratio", 0.0)
        if ratio <= 0:
            return model

        node = self._find_node_by_type(model, action.operator_name, action.op_type)
        if node is None:
            return model

        attr = self._get_node_attr(node, "coefficients")
        if attr is None or attr.type != onnx.AttributeProto.FLOATS:
            return model

        values = np.array(list(attr.floats), dtype=np.float32)
        threshold = np.percentile(np.abs(values), ratio * 100)
        values[np.abs(values) < threshold] = 0.0

        del attr.floats[:]
        attr.floats.extend(values.tolist())
        return model

    def _apply_sv_pruning(
        self, model: onnx.ModelProto, action: ConcreteAction,
    ) -> onnx.ModelProto:
        """Prune support vectors with small coefficients."""
        ratio = action.parameters.get("ratio", 0.0)
        if ratio <= 0:
            return model

        node = self._find_node_by_type(model, action.operator_name, action.op_type)
        if node is None:
            return model

        coeff_attr = self._get_node_attr(node, "coefficients")
        if coeff_attr is None:
            return model

        values = np.array(list(coeff_attr.floats), dtype=np.float32)
        threshold = np.percentile(np.abs(values), ratio * 100)
        values[np.abs(values) < threshold] = 0.0

        del coeff_attr.floats[:]
        coeff_attr.floats.extend(values.tolist())
        return model

    # --- Utility Methods ---

    def _find_node(self, model: onnx.ModelProto, name: str) -> onnx.NodeProto | None:
        for node in model.graph.node:
            if node.name == name:
                return node
        return None

    def _find_node_by_type(
        self, model: onnx.ModelProto, name: str, op_type: str,
    ) -> onnx.NodeProto | None:
        node = self._find_node(model, name)
        if node is not None:
            return node
        for node in model.graph.node:
            if node.op_type == op_type:
                return node
        return None

    def _find_initializer(
        self, model: onnx.ModelProto, name: str,
    ) -> onnx.TensorProto | None:
        for init in model.graph.initializer:
            if init.name == name:
                return init
        return None

    def _replace_initializer(
        self, model: onnx.ModelProto, name: str, new_init: onnx.TensorProto,
    ) -> None:
        for i, init in enumerate(model.graph.initializer):
            if init.name == name:
                model.graph.initializer[i].CopyFrom(new_init)
                return

    def _remove_initializer(self, model: onnx.ModelProto, name: str) -> None:
        """Remove an initializer (and any matching graph input declaration)."""
        for i, init in enumerate(model.graph.initializer):
            if init.name == name:
                del model.graph.initializer[i]
                break
        # Older opsets may also declare initializers as graph inputs; the name
        # is now produced by a Cast/DequantizeLinear node instead.
        for i, inp in enumerate(model.graph.input):
            if inp.name == name:
                del model.graph.input[i]
                break

    @staticmethod
    def _unique_name(model: onnx.ModelProto, candidate: str) -> str:
        existing = {init.name for init in model.graph.initializer}
        existing.update(n.name for n in model.graph.node)
        name = candidate
        suffix = 2
        while name in existing:
            name = f"{candidate}_{suffix}"
            suffix += 1
        return name

    @staticmethod
    def _model_opset(model: onnx.ModelProto) -> int:
        for imp in model.opset_import:
            if imp.domain in ("", "ai.onnx"):
                return imp.version
        return 0

    @staticmethod
    def _get_node_attr(
        node: onnx.NodeProto, name: str,
    ) -> onnx.AttributeProto | None:
        for attr in node.attribute:
            if attr.name == name:
                return attr
        return None
