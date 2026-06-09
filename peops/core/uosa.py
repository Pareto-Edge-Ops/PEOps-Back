"""UOSA: Universal Operator Sensitivity Analysis

The core theoretical contribution of UGCO. Measures operator-level sensitivity
via perturbation — gradient-free, works across all ONNX operator types including
non-differentiable tree ensembles and SVMs.

S(op_i) = E_x [ ||f(x) - f_perturbed_at_i(x)||^2 ] / E_x [ ||f(x)||^2 ]

Under small-perturbation assumptions, S(op_i) ≈ ∂I(y_i; Task)/∂σ² (Theorem 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

from peops.graph.onnx_analyzer import GraphInfo, OperatorCategory, OperatorInfo


@dataclass
class SensitivityResult:
    """Per-operator sensitivity scores."""
    operator_name: str
    op_type: str
    category: OperatorCategory
    sensitivity: float
    rank: int = -1

    @property
    def is_highly_sensitive(self) -> bool:
        return self.sensitivity > 0.1


@dataclass
class SensitivityProfile:
    """Full sensitivity profile for an ONNX model."""
    model_path: str | None
    results: list[SensitivityResult]
    calibration_size: int
    perturbation_scale: float

    @property
    def ranked(self) -> list[SensitivityResult]:
        sorted_results = sorted(self.results, key=lambda r: r.sensitivity, reverse=True)
        for i, r in enumerate(sorted_results):
            r.rank = i
        return sorted_results

    def get_sensitivity(self, operator_name: str) -> float | None:
        for r in self.results:
            if r.operator_name == operator_name:
                return r.sensitivity
        return None

    def get_protection_set(self, top_p: float = 0.2) -> set[str]:
        """Return operator names in the top p fraction by sensitivity (to protect from aggressive compression)."""
        ranked = self.ranked
        n_protect = max(1, int(len(ranked) * top_p))
        return {r.operator_name for r in ranked[:n_protect]}

    def normalized_scores(self) -> dict[str, float]:
        """Return sensitivity scores normalized to [0, 1]."""
        if not self.results:
            return {}
        max_s = max(r.sensitivity for r in self.results)
        if max_s == 0:
            return {r.operator_name: 0.0 for r in self.results}
        return {r.operator_name: r.sensitivity / max_s for r in self.results}


def _to_float_array(value: Any) -> np.ndarray | None:
    """Convert model output to float64 numpy array, handling dicts/lists from ZipMap etc."""
    if isinstance(value, np.ndarray):
        try:
            return value.astype(np.float64)
        except (ValueError, TypeError):
            return None
    if isinstance(value, list):
        if not value:
            return None
        if isinstance(value[0], dict):
            arrays = []
            for d in value:
                arrays.append(np.array(list(d.values()), dtype=np.float64))
            return np.stack(arrays)
        try:
            return np.array(value, dtype=np.float64)
        except (ValueError, TypeError):
            return None
    if isinstance(value, dict):
        try:
            return np.array(list(value.values()), dtype=np.float64)
        except (ValueError, TypeError):
            return None
    return None


class PerturbationStrategy:
    """Generates perturbation noise appropriate for each operator category."""

    def __init__(self, scale: float = 0.01, rng: np.random.Generator | None = None):
        self.scale = scale
        self.rng = rng or np.random.default_rng(42)

    def generate(
        self,
        output_tensor: np.ndarray,
        category: OperatorCategory,
    ) -> np.ndarray:
        if category == OperatorCategory.TREE_ENSEMBLE:
            return self._tree_perturbation(output_tensor)
        if category == OperatorCategory.SVM:
            return self._svm_perturbation(output_tensor)
        return self._gaussian_perturbation(output_tensor)

    def _gaussian_perturbation(self, tensor: np.ndarray) -> np.ndarray:
        magnitude = np.abs(tensor).mean() + 1e-8
        noise = self.rng.normal(0, self.scale * magnitude, size=tensor.shape)
        return noise.astype(tensor.dtype)

    def _tree_perturbation(self, tensor: np.ndarray) -> np.ndarray:
        """For tree outputs: perturb leaf values proportional to output range."""
        value_range = np.ptp(tensor) + 1e-8
        noise = self.rng.normal(0, self.scale * value_range, size=tensor.shape)
        return noise.astype(tensor.dtype)

    def _svm_perturbation(self, tensor: np.ndarray) -> np.ndarray:
        magnitude = np.abs(tensor).mean() + 1e-8
        noise = self.rng.normal(0, self.scale * magnitude, size=tensor.shape)
        return noise.astype(tensor.dtype)


class UOSAEngine:
    """Universal Operator Sensitivity Analysis engine.

    Computes perturbation-based sensitivity for every compressible operator
    in an ONNX graph, regardless of operator type.
    """

    def __init__(
        self,
        perturbation_scale: float = 0.01,
        seed: int = 42,
    ):
        self.perturbation_scale = perturbation_scale
        self.strategy = PerturbationStrategy(scale=perturbation_scale, rng=np.random.default_rng(seed))

    def analyze(
        self,
        model: onnx.ModelProto,
        graph_info: GraphInfo,
        calibration_data: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    ) -> SensitivityProfile:
        if isinstance(calibration_data, dict):
            calibration_data = [calibration_data]

        compressible = graph_info.compressible_operators
        if not compressible:
            return SensitivityProfile(
                model_path=graph_info.model_path,
                results=[],
                calibration_size=len(calibration_data),
                perturbation_scale=self.perturbation_scale,
            )

        baseline_outputs = self._run_baseline(model, calibration_data)

        results = []
        for op in compressible:
            sensitivity = self._compute_operator_sensitivity(
                model, graph_info, op, calibration_data, baseline_outputs,
            )
            results.append(SensitivityResult(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                sensitivity=sensitivity,
            ))

        return SensitivityProfile(
            model_path=graph_info.model_path,
            results=results,
            calibration_size=len(calibration_data),
            perturbation_scale=self.perturbation_scale,
        )

    def _run_baseline(
        self,
        model: onnx.ModelProto,
        calibration_data: list[dict[str, np.ndarray]],
    ) -> list[list[np.ndarray]]:
        session = ort.InferenceSession(model.SerializeToString())
        output_names = [o.name for o in session.get_outputs()]
        results = []
        for sample in calibration_data:
            outputs = session.run(output_names, sample)
            results.append(outputs)
        return results

    def _compute_operator_sensitivity(
        self,
        model: onnx.ModelProto,
        graph_info: GraphInfo,
        op: OperatorInfo,
        calibration_data: list[dict[str, np.ndarray]],
        baseline_outputs: list[list[np.ndarray]],
    ) -> float:
        """Compute S(op_i) by injecting perturbation at op_i's output and measuring
        the deviation in the model's final output.

        For ML operators (tree, linear, SVM), falls back to attribute-perturbation
        since their outputs may lack shape info for graph surgery.
        """
        if op.is_ml_operator:
            return self._compute_ml_operator_sensitivity(
                model, op, calibration_data, baseline_outputs,
            )

        return self._compute_neural_operator_sensitivity(
            model, op, calibration_data, baseline_outputs,
        )

    def _compute_neural_operator_sensitivity(
        self,
        model: onnx.ModelProto,
        op: OperatorInfo,
        calibration_data: list[dict[str, np.ndarray]],
        baseline_outputs: list[list[np.ndarray]],
    ) -> float:
        """Graph-surgery approach: insert Add(noise) after op_i's output."""
        perturbed_model = self._create_perturbed_model(model, op)
        if perturbed_model is None:
            return 0.0

        try:
            session = ort.InferenceSession(perturbed_model.SerializeToString())
        except Exception:
            return 0.0

        output_names = [o.name for o in session.get_outputs()]

        total_deviation = 0.0
        total_baseline = 0.0

        for i, sample in enumerate(calibration_data):
            intermediate = self._get_intermediate_output(model, op, sample)
            if intermediate is None:
                continue

            noise = self.strategy.generate(intermediate, op.category)
            noise_feed = {f"__uosa_noise_{op.name}": noise}
            feed = {**sample, **noise_feed}

            try:
                perturbed_outputs = session.run(output_names, feed)
            except Exception:
                continue

            for base_out, pert_out in zip(baseline_outputs[i], perturbed_outputs):
                base_f = _to_float_array(base_out)
                pert_f = _to_float_array(pert_out)
                if base_f is None or pert_f is None or base_f.shape != pert_f.shape:
                    continue
                deviation = np.mean((base_f - pert_f) ** 2)
                baseline_mag = np.mean(base_f ** 2) + 1e-12
                total_deviation += deviation
                total_baseline += baseline_mag

        if total_baseline == 0:
            return 0.0

        return float(total_deviation / total_baseline)

    def _compute_ml_operator_sensitivity(
        self,
        model: onnx.ModelProto,
        op: OperatorInfo,
        calibration_data: list[dict[str, np.ndarray]],
        baseline_outputs: list[list[np.ndarray]],
    ) -> float:
        """Attribute-perturbation approach for ML operators (tree, linear, SVM).

        Instead of graph surgery, directly perturb the operator's parameters
        (stored as node attributes) and measure how the final output changes.
        This is equivalent to UOSA but applied at the parameter level.
        """
        perturb_attrs = self._get_perturbable_attributes(op)
        if not perturb_attrs:
            return 0.0

        total_deviation = 0.0
        total_baseline = 0.0

        for attr_name in perturb_attrs:
            perturbed_model = self._perturb_ml_attribute(model, op, attr_name)
            if perturbed_model is None:
                continue

            try:
                session = ort.InferenceSession(perturbed_model.SerializeToString())
            except Exception:
                continue

            output_names = [o.name for o in session.get_outputs()]

            for i, sample in enumerate(calibration_data):
                try:
                    perturbed_outputs = session.run(output_names, sample)
                except Exception:
                    continue

                for base_out, pert_out in zip(baseline_outputs[i], perturbed_outputs):
                    base_f = _to_float_array(base_out)
                    pert_f = _to_float_array(pert_out)
                    if base_f is None or pert_f is None:
                        continue
                    if base_f.shape != pert_f.shape:
                        continue
                    deviation = np.mean((base_f - pert_f) ** 2)
                    baseline_mag = np.mean(base_f ** 2) + 1e-12
                    total_deviation += deviation
                    total_baseline += baseline_mag

        if total_baseline == 0:
            return 0.0

        return float(total_deviation / total_baseline)

    def _get_perturbable_attributes(self, op: OperatorInfo) -> list[str]:
        """Return attribute names that can be numerically perturbed for this ML op."""
        if op.category == OperatorCategory.TREE_ENSEMBLE:
            candidates = ["class_weights", "target_weights", "leaf_weights"]
            return [a for a in candidates if a in op.attributes and op.attributes[a]]
        if op.category == OperatorCategory.LINEAR_MODEL:
            return [a for a in ["coefficients"] if a in op.attributes and op.attributes[a]]
        if op.category == OperatorCategory.SVM:
            return [a for a in ["coefficients"] if a in op.attributes and op.attributes[a]]
        return []

    def _perturb_ml_attribute(
        self,
        model: onnx.ModelProto,
        op: OperatorInfo,
        attr_name: str,
    ) -> onnx.ModelProto | None:
        """Create a copy of the model with a specific ML operator attribute perturbed."""
        perturbed = onnx.ModelProto()
        perturbed.CopyFrom(model)

        for node in perturbed.graph.node:
            if node.name != op.name and node.op_type != op.op_type:
                continue
            if node.name and node.name != op.name:
                continue

            for attr in node.attribute:
                if attr.name == attr_name:
                    if attr.type == onnx.AttributeProto.FLOATS:
                        values = np.array(list(attr.floats), dtype=np.float32)
                        noise = self.strategy.generate(values, op.category)
                        perturbed_values = values + noise
                        del attr.floats[:]
                        attr.floats.extend(perturbed_values.tolist())
                        return perturbed
                    if attr.type == onnx.AttributeProto.FLOAT:
                        noise_val = self.strategy.rng.normal(0, self.strategy.scale * (abs(attr.f) + 1e-8))
                        attr.f += noise_val
                        return perturbed

        return None

    def _get_intermediate_output(
        self,
        model: onnx.ModelProto,
        op: OperatorInfo,
        sample: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        """Run the model up to op_i and capture its output tensor."""
        if not op.output_names:
            return None

        target_output = op.output_names[0]

        augmented = onnx.ModelProto()
        augmented.CopyFrom(model)
        existing_output_names = {o.name for o in augmented.graph.output}

        if target_output not in existing_output_names:
            shape_info = None
            for vi in model.graph.value_info:
                if vi.name == target_output:
                    shape_info = vi
                    break

            if shape_info is not None:
                new_output = augmented.graph.output.add()
                new_output.CopyFrom(shape_info)
            else:
                new_output = augmented.graph.output.add()
                new_output.name = target_output

        try:
            session = ort.InferenceSession(augmented.SerializeToString())
            results = session.run([target_output], sample)
            return results[0]
        except Exception:
            return None

    def _create_perturbed_model(
        self,
        model: onnx.ModelProto,
        op: OperatorInfo,
    ) -> onnx.ModelProto | None:
        """Create a model variant where op_i's output has additive noise injected.

        Inserts: op_i_output_original -> Add(noise_input) -> op_i_output_perturbed
        and rewires downstream consumers.
        """
        if not op.output_names:
            return None

        target_output = op.output_names[0]

        perturbed = onnx.ModelProto()
        perturbed.CopyFrom(model)
        graph = perturbed.graph

        noise_input_name = f"__uosa_noise_{op.name}"
        renamed_output = f"__uosa_pre_perturb_{op.name}"

        for node in graph.node:
            for i, out in enumerate(node.output):
                if out == target_output:
                    node.output[i] = renamed_output

        for vi in graph.value_info:
            if vi.name == target_output:
                new_vi = graph.value_info.add()
                new_vi.CopyFrom(vi)
                new_vi.name = renamed_output
                break

        add_node = onnx.helper.make_node(
            "Add",
            inputs=[renamed_output, noise_input_name],
            outputs=[target_output],
            name=f"__uosa_add_{op.name}",
        )
        graph.node.append(add_node)

        target_shape = None
        for vi in list(graph.value_info) + list(graph.output):
            if vi.name == target_output:
                target_shape = vi
                break

        if target_shape is not None:
            noise_input = onnx.helper.make_tensor_value_info(
                noise_input_name,
                target_shape.type.tensor_type.elem_type,
                [d.dim_value if d.dim_value > 0 else d.dim_param or "N"
                 for d in target_shape.type.tensor_type.shape.dim],
            )
        else:
            noise_input = onnx.helper.make_tensor_value_info(
                noise_input_name, TensorProto.FLOAT, None,
            )

        graph.input.append(noise_input)

        return perturbed


def compute_uosa(
    model_or_path: onnx.ModelProto | str,
    calibration_data: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    graph_info: GraphInfo | None = None,
    perturbation_scale: float = 0.01,
    seed: int = 42,
) -> SensitivityProfile:
    """Convenience function to compute UOSA sensitivity for an ONNX model."""
    from peops.graph.onnx_analyzer import OnnxAnalyzer

    if isinstance(model_or_path, str):
        model = onnx.load(model_or_path)
    else:
        model = model_or_path

    if graph_info is None:
        analyzer = OnnxAnalyzer()
        graph_info = analyzer.analyze(model)

    engine = UOSAEngine(perturbation_scale=perturbation_scale, seed=seed)
    return engine.analyze(model, graph_info, calibration_data)
