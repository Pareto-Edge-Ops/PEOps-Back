"""Calibration Data Generator: synthesize model-aware probe inputs WITHOUT original data.

Key insight: the model's first-layer weights encode information about expected
input scales. Features with large weights likely have small input values (and vice
versa), because well-trained models balance weight × input magnitudes.

Inverse-weight heuristic:
    estimated_input_scale_j ≈ 1 / (std(W[:, j]) + eps)

This generates probes in the model's "natural" input space, not random noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper

from peops.graph.model_detector import ArchitectureType


@dataclass
class CalibrationInfo:
    """Generated calibration data with metadata."""
    probes: list[dict[str, np.ndarray]]
    strategy: str
    input_scales: dict[str, np.ndarray] | None
    n_probes: int


class CalibrationGenerator:
    """Generates synthetic calibration data from model weights alone."""

    def __init__(self, n_probes: int = 64, seed: int = 42):
        self.n_probes = n_probes
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        model: onnx.ModelProto,
        input_spec: dict[str, list[int]],
        architecture: ArchitectureType | None = None,
    ) -> CalibrationInfo:
        initializer_map = {
            init.name: numpy_helper.to_array(init)
            for init in model.graph.initializer
        }

        first_weight = self._find_first_weight(model, initializer_map)
        input_scales = self._estimate_input_scales(first_weight)

        probes = []
        input_name = next(iter(input_spec))
        shape_template = input_spec[input_name]
        batch_shape = [1 if d <= 0 else d for d in shape_template]

        is_image = architecture == ArchitectureType.CNN or len(batch_shape) == 4
        strategy = "image" if is_image else "tabular"

        if is_image:
            probes = self._generate_image_probes(batch_shape)
        else:
            probes = self._generate_tabular_probes(batch_shape, input_scales)

        probe_dicts = [{input_name: p} for p in probes]
        probe_dicts = self._validate_probes(model, probe_dicts)

        return CalibrationInfo(
            probes=probe_dicts,
            strategy=strategy,
            input_scales=input_scales,
            n_probes=len(probe_dicts),
        )

    def _find_first_weight(
        self,
        model: onnx.ModelProto,
        initializer_map: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        """Find the first layer's weight matrix (connects input to hidden)."""
        model_inputs = {
            inp.name for inp in model.graph.input
            if inp.name not in initializer_map
        }

        for node in model.graph.node:
            if node.op_type in ("MatMul", "Gemm", "Conv", "Linear"):
                for inp in node.input:
                    if inp in initializer_map:
                        w = initializer_map[inp]
                        other_inputs = [i for i in node.input if i in model_inputs or
                                        any(i in n.output for n in model.graph.node)]
                        if other_inputs:
                            return w

            if node.op_type in ("LinearClassifier", "LinearRegressor"):
                for attr in node.attribute:
                    if attr.name == "coefficients" and attr.floats:
                        vals = np.array(list(attr.floats), dtype=np.float32)
                        n_classes_attr = None
                        for a in node.attribute:
                            if a.name == "classlabels_int64s":
                                n_classes_attr = len(a.ints)
                            elif a.name == "classlabels_strings":
                                n_classes_attr = len(a.strings)
                        if n_classes_attr and n_classes_attr > 0:
                            n_features = len(vals) // n_classes_attr
                            return vals.reshape(n_classes_attr, n_features)
                        return vals.reshape(1, -1) if vals.ndim == 1 else vals

        return None

    def _estimate_input_scales(
        self, weight: np.ndarray | None,
    ) -> dict[str, np.ndarray] | None:
        """Estimate per-feature input scale from first-layer weights.

        Inverse-weight heuristic: if W[:, j] has large std, feature j likely
        has small values (the model amplifies it). So input_scale_j ≈ 1/std(W[:,j]).
        """
        if weight is None:
            return None

        if weight.ndim == 1:
            weight = weight.reshape(1, -1)

        if weight.ndim == 4:
            # Conv: [C_out, C_in, H, W] → treat C_in as features
            n_features = weight.shape[1]
            w_flat = weight.reshape(weight.shape[0], n_features, -1)
            w_std = w_flat.std(axis=(0, 2)) + 1e-8
        elif weight.ndim == 2:
            # MatMul/Gemm/Linear: [in, out] or [out, in]
            # Use the dimension that likely corresponds to input features
            if weight.shape[0] > weight.shape[1]:
                w_std = weight.std(axis=1) + 1e-8  # [in, out] → std across outputs
            else:
                w_std = weight.std(axis=0) + 1e-8  # [out, in] → std across outputs
        else:
            return None

        input_scale = 1.0 / w_std
        input_scale = np.clip(input_scale, 0.01, 1000.0)

        return {"estimated_scale": input_scale}

    def _generate_tabular_probes(
        self,
        batch_shape: list[int],
        input_scales: dict[str, np.ndarray] | None,
    ) -> list[np.ndarray]:
        """Generate tabular probes using weight-derived input scales.

        Handles arbitrary input shapes: (1, N), (1, seq_len, feat_dim), etc.
        """
        total_elements = int(np.prod(batch_shape))
        probes = []
        n = self.n_probes

        if input_scales and "estimated_scale" in input_scales:
            raw_scale = input_scales["estimated_scale"]
            if len(raw_scale) == total_elements:
                scale = raw_scale
            else:
                scale = np.ones(total_elements, dtype=np.float32)
        else:
            scale = np.ones(total_elements, dtype=np.float32)

        # Strategy 1: Scaled normal (50%)
        for _ in range(n // 2):
            x = self.rng.normal(0, 1, size=total_elements).astype(np.float32) * scale
            probes.append(x.reshape(batch_shape))

        # Strategy 2: Scaled uniform (25%)
        for _ in range(n // 4):
            x = self.rng.uniform(-3, 3, size=total_elements).astype(np.float32) * scale
            probes.append(x.reshape(batch_shape))

        # Strategy 3: Edge cases (25%)
        n_edge = n - len(probes)
        probes.append(np.zeros(batch_shape, dtype=np.float32))
        probes.append(np.ones(batch_shape, dtype=np.float32) * 0.5)
        for _ in range(max(0, n_edge - 2)):
            x = self.rng.standard_normal(total_elements).astype(np.float32) * scale * 0.3
            probes.append(x.reshape(batch_shape))

        return probes[:n]

    def _generate_image_probes(self, batch_shape: list[int]) -> list[np.ndarray]:
        """Generate image-like probes (normalized random, patterns)."""
        probes = []
        n = self.n_probes

        # Standard normal (ImageNet-style normalized)
        for _ in range(n // 2):
            probes.append(self.rng.standard_normal(batch_shape).astype(np.float32))

        # Uniform [0, 1]
        for _ in range(n // 4):
            probes.append(self.rng.uniform(0, 1, size=batch_shape).astype(np.float32))

        # Edge cases
        probes.append(np.zeros(batch_shape, dtype=np.float32))
        probes.append(np.ones(batch_shape, dtype=np.float32))
        remaining = n - len(probes)
        for _ in range(remaining):
            probes.append(self.rng.standard_normal(batch_shape).astype(np.float32) * 0.5)

        return probes[:n]

    def _validate_probes(
        self,
        model: onnx.ModelProto,
        probes: list[dict[str, np.ndarray]],
    ) -> list[dict[str, np.ndarray]]:
        """Remove probes that cause NaN/Inf outputs."""
        import onnxruntime as ort
        try:
            session = ort.InferenceSession(model.SerializeToString())
            output_names = [o.name for o in session.get_outputs()]
        except Exception:
            return probes

        valid = []
        for p in probes:
            try:
                outputs = session.run(output_names, p)
                is_clean = all(
                    not np.any(np.isnan(o)) and not np.any(np.isinf(o))
                    for o in outputs if isinstance(o, np.ndarray)
                )
                if is_clean:
                    valid.append(p)
            except Exception:
                continue

        return valid if valid else probes[:1]
