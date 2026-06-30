"""Data-Free Compression Validation (DFCV): validate compression quality
WITHOUT original training/test data.

Three orthogonal signals:
  1. Output Fidelity (OFS): original vs compressed outputs on synthetic probes
  2. Weight Fidelity (WFS): direct parameter-space comparison
  3. Structural Integrity (SIS): sanity checks (NaN, shape, executability)

Quality score Q = 0.5·OFS + 0.3·WFS + 0.2·SIS  ∈ [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from astra.core.calibration_generator import CalibrationGenerator
from astra.graph.model_detector import ArchitectureType


@dataclass
class ValidationDetail:
    ofs_per_output: dict[str, float]
    wfs_per_weight: dict[str, float]
    executability: bool
    nan_free: bool
    shape_preserved: bool
    output_range_ratio: float
    prediction_consistency: float | None


@dataclass
class ValidationResult:
    quality_score: float
    output_fidelity: float
    weight_fidelity: float
    structural_integrity: float
    risk_level: str
    detail: ValidationDetail
    n_probes: int

    @property
    def is_safe(self) -> bool:
        return self.quality_score >= 0.95

    @property
    def is_acceptable(self) -> bool:
        return self.quality_score >= 0.85


class CompressionValidator:
    """Validates compression quality without labeled data."""

    def __init__(
        self,
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        n_probes: int = 64,
        seed: int = 42,
    ):
        self.w_ofs, self.w_wfs, self.w_sis = weights
        self.n_probes = n_probes
        self.seed = seed

    def validate(
        self,
        original: onnx.ModelProto,
        compressed: onnx.ModelProto,
        input_spec: dict[str, list[int]] | None = None,
        architecture: ArchitectureType | None = None,
        probes: list[dict[str, np.ndarray]] | None = None,
    ) -> ValidationResult:
        if input_spec is None:
            input_spec = self._extract_input_spec(original)

        if probes is None:
            gen = CalibrationGenerator(n_probes=self.n_probes, seed=self.seed)
            cal_info = gen.generate(original, input_spec, architecture)
            probes = cal_info.probes

        # Signal 1: Output Fidelity
        ofs, ofs_detail, orig_outputs, comp_outputs = self._compute_output_fidelity(
            original, compressed, probes)

        # Signal 2: Weight Fidelity
        wfs, wfs_detail = self._compute_weight_fidelity(original, compressed)

        # Signal 3: Structural Integrity
        sis, sis_detail = self._compute_structural_integrity(
            original, compressed, probes, orig_outputs, comp_outputs)

        quality = self.w_ofs * ofs + self.w_wfs * wfs + self.w_sis * sis
        quality = float(np.clip(quality, 0, 1))

        if quality >= 0.95:
            risk = "safe"
        elif quality >= 0.85:
            risk = "moderate"
        elif quality >= 0.70:
            risk = "risky"
        else:
            risk = "failed"

        return ValidationResult(
            quality_score=quality,
            output_fidelity=ofs,
            weight_fidelity=wfs,
            structural_integrity=sis,
            risk_level=risk,
            detail=ValidationDetail(
                ofs_per_output=ofs_detail,
                wfs_per_weight=wfs_detail,
                **sis_detail,
            ),
            n_probes=len(probes),
        )

    def _compute_output_fidelity(
        self,
        original: onnx.ModelProto,
        compressed: onnx.ModelProto,
        probes: list[dict[str, np.ndarray]],
    ) -> tuple[float, dict, list, list]:
        """OFS = 1 - E[||f(x)-f_c(x)||² / (||f(x)||² + eps)]"""
        try:
            orig_sess = ort.InferenceSession(original.SerializeToString())
            comp_sess = ort.InferenceSession(compressed.SerializeToString())
        except Exception:
            return 0.0, {}, [], []

        orig_out_names = [o.name for o in orig_sess.get_outputs()]
        comp_out_names = [o.name for o in comp_sess.get_outputs()]

        all_orig = []
        all_comp = []
        per_output_divergence: dict[str, list[float]] = {n: [] for n in orig_out_names}
        discrete_outputs: set[str] = set()

        for probe in probes:
            try:
                o_out = orig_sess.run(orig_out_names, probe)
                c_out = comp_sess.run(comp_out_names, probe)
            except Exception:
                continue

            all_orig.append(o_out)
            all_comp.append(c_out)

            for i, name in enumerate(orig_out_names):
                if i >= len(c_out):
                    continue
                # Discrete label outputs (int64 class predictions) are excluded
                # from relative-L2 fidelity: a single 0->1 label flip makes the
                # ratio diverge (near-zero baseline). Label agreement is covered
                # by prediction_consistency in the structural-integrity signal.
                raw = o_out[i]
                if isinstance(raw, np.ndarray) and np.issubdtype(raw.dtype, np.integer):
                    discrete_outputs.add(name)
                    continue
                o_arr = self._to_float(o_out[i])
                c_arr = self._to_float(c_out[i])
                if o_arr is None or c_arr is None or o_arr.shape != c_arr.shape:
                    per_output_divergence[name].append(1.0)
                    continue
                diff = np.mean((o_arr - c_arr) ** 2)
                base = np.mean(o_arr ** 2) + 1e-12
                per_output_divergence[name].append(float(diff / base))

        ofs_per = {}
        for name, divs in per_output_divergence.items():
            if name in discrete_outputs and not divs:
                continue
            if divs:
                ofs_per[name] = float(1.0 - np.clip(np.mean(divs), 0, 1))
            else:
                ofs_per[name] = 0.0

        if not ofs_per and discrete_outputs and all_orig:
            # Model exposes ONLY discrete outputs: fall back to exact label
            # agreement as the fidelity signal.
            idx = [i for i, n in enumerate(orig_out_names) if n in discrete_outputs]
            matches, total = 0, 0
            for o_out, c_out in zip(all_orig, all_comp):
                for i in idx:
                    if i < len(c_out):
                        matches += int(np.array_equal(o_out[i], c_out[i]))
                        total += 1
            for name in discrete_outputs:
                ofs_per[name] = matches / total if total else 0.0

        ofs = float(np.mean(list(ofs_per.values()))) if ofs_per else 0.0
        return ofs, ofs_per, all_orig, all_comp

    def _compute_weight_fidelity(
        self,
        original: onnx.ModelProto,
        compressed: onnx.ModelProto,
    ) -> tuple[float, dict]:
        """WFS = 1 - mean(||W-W_c||_F / (||W||_F + eps))"""
        orig_map = {i.name: numpy_helper.to_array(i) for i in original.graph.initializer}
        comp_map = {i.name: numpy_helper.to_array(i) for i in compressed.graph.initializer}

        # Also compare ML operator attributes
        orig_attrs = self._extract_float_attributes(original)
        comp_attrs = self._extract_float_attributes(compressed)
        orig_map.update(orig_attrs)
        comp_map.update(comp_attrs)

        per_weight = {}
        for name, orig_w in orig_map.items():
            if name not in comp_map:
                # Real-bytes quantization renames initializers ({name}_fp16 /
                # {name}_q + scale + zp behind Cast/DequantizeLinear nodes).
                # Reconstruct the effective fp32 weight so genuinely-quantized
                # models aren't scored as if the weight vanished.
                reconstructed = self._reconstruct_quantized_weight(comp_map, name)
                if reconstructed is None:
                    per_weight[name] = 0.0
                    continue
                comp_map[name] = reconstructed
            comp_w = comp_map[name]
            if orig_w.shape != comp_w.shape:
                per_weight[name] = 0.0
                continue
            orig_f = orig_w.astype(np.float64).flatten()
            comp_f = comp_w.astype(np.float64).flatten()
            norm_orig = np.linalg.norm(orig_f) + 1e-12
            diff_norm = np.linalg.norm(orig_f - comp_f)
            per_weight[name] = float(1.0 - np.clip(diff_norm / norm_orig, 0, 1))

        wfs = float(np.mean(list(per_weight.values()))) if per_weight else 1.0
        return wfs, per_weight

    def _compute_structural_integrity(
        self,
        original: onnx.ModelProto,
        compressed: onnx.ModelProto,
        probes: list[dict[str, np.ndarray]],
        orig_outputs: list,
        comp_outputs: list,
    ) -> tuple[float, dict]:
        """SIS: executability, NaN-free, shape preserved, prediction consistency."""
        # Executability
        executable = len(comp_outputs) > 0

        # NaN/Inf check
        nan_free = True
        for outputs in comp_outputs:
            for o in outputs:
                arr = self._to_float(o)
                if arr is not None and (np.any(np.isnan(arr)) or np.any(np.isinf(arr))):
                    nan_free = False
                    break

        # Shape check
        shape_ok = True
        if orig_outputs and comp_outputs:
            for o_out, c_out in zip(orig_outputs[0], comp_outputs[0]):
                o_arr = self._to_float(o_out)
                c_arr = self._to_float(c_out)
                if o_arr is not None and c_arr is not None:
                    if o_arr.shape != c_arr.shape:
                        shape_ok = False

        # Output range ratio
        range_ratio = 1.0
        if orig_outputs and comp_outputs:
            orig_ranges = []
            comp_ranges = []
            for o_list, c_list in zip(orig_outputs, comp_outputs):
                for o, c in zip(o_list, c_list):
                    o_arr = self._to_float(o)
                    c_arr = self._to_float(c)
                    if o_arr is not None:
                        orig_ranges.append(np.ptp(o_arr))
                    if c_arr is not None:
                        comp_ranges.append(np.ptp(c_arr))
            if orig_ranges and comp_ranges:
                o_mean = np.mean(orig_ranges) + 1e-12
                c_mean = np.mean(comp_ranges) + 1e-12
                range_ratio = min(o_mean, c_mean) / max(o_mean, c_mean)

        # Prediction consistency (classifiers: does argmax match?)
        pred_consistency = None
        if orig_outputs and comp_outputs:
            matches = 0
            total = 0
            for o_list, c_list in zip(orig_outputs, comp_outputs):
                o_arr = self._to_float(o_list[0])
                c_arr = self._to_float(c_list[0])
                if o_arr is not None and c_arr is not None and o_arr.ndim >= 1:
                    if o_arr.dtype in (np.int64, np.int32, np.float64, np.float32):
                        if o_arr.ndim >= 2 and o_arr.shape[-1] > 1:
                            matches += int(np.argmax(o_arr) == np.argmax(c_arr))
                        else:
                            matches += int(np.array_equal(
                                np.round(o_arr).astype(int),
                                np.round(c_arr).astype(int)))
                        total += 1
            if total > 0:
                pred_consistency = matches / total

        scores = [
            1.0 if executable else 0.0,
            1.0 if nan_free else 0.0,
            1.0 if shape_ok else 0.0,
            float(range_ratio),
        ]
        if pred_consistency is not None:
            scores.append(pred_consistency)

        sis = float(np.mean(scores))

        detail = {
            "executability": executable,
            "nan_free": nan_free,
            "shape_preserved": shape_ok,
            "output_range_ratio": float(range_ratio),
            "prediction_consistency": pred_consistency,
        }
        return sis, detail

    @staticmethod
    def _reconstruct_quantized_weight(
        comp_map: dict[str, np.ndarray], name: str,
    ) -> np.ndarray | None:
        """Rebuild the effective fp32 weight from real-quant storage tensors."""
        fp16 = comp_map.get(f"{name}_fp16")
        if fp16 is not None:
            return fp16.astype(np.float32)

        q = comp_map.get(f"{name}_q")
        scale = comp_map.get(f"{name}_scale")
        if q is None or scale is None:
            return None
        zp = comp_map.get(f"{name}_zp")
        qf = q.astype(np.float32)
        if zp is not None:
            zpf = zp.astype(np.float32)
            if zpf.size == 1:
                qf = qf - float(zpf.reshape(-1)[0])
            elif zpf.ndim == 1 and qf.ndim >= 1 and qf.shape[0] == zpf.size:
                qf = qf - zpf.reshape(-1, *([1] * (qf.ndim - 1)))
        if scale.size == 1:
            return qf * float(np.asarray(scale).reshape(-1)[0])
        if scale.ndim == 1 and qf.ndim >= 1 and qf.shape[0] == scale.size:
            return qf * scale.astype(np.float32).reshape(-1, *([1] * (qf.ndim - 1)))
        return None

    def _extract_float_attributes(self, model: onnx.ModelProto) -> dict[str, np.ndarray]:
        """Extract FLOATS attributes from ML operators for comparison."""
        attrs = {}
        target_attrs = {"class_weights", "target_weights", "coefficients"}
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.name in target_attrs and attr.type == onnx.AttributeProto.FLOATS:
                    key = f"{node.name or node.op_type}__{attr.name}"
                    attrs[key] = np.array(list(attr.floats), dtype=np.float32)
        return attrs

    def _extract_input_spec(self, model: onnx.ModelProto) -> dict[str, list[int]]:
        init_names = {i.name for i in model.graph.initializer}
        spec = {}
        for inp in model.graph.input:
            if inp.name in init_names:
                continue
            shape = []
            try:
                for d in inp.type.tensor_type.shape.dim:
                    shape.append(d.dim_value if d.dim_value > 0 else -1)
            except Exception:
                shape = [-1]
            spec[inp.name] = shape
        return spec

    @staticmethod
    def _to_float(val) -> np.ndarray | None:
        if isinstance(val, np.ndarray):
            try:
                return val.astype(np.float64)
            except (ValueError, TypeError):
                return None
        if isinstance(val, list):
            if not val:
                return None
            if isinstance(val[0], dict):
                return np.array([list(d.values()) for d in val], dtype=np.float64)
            try:
                return np.array(val, dtype=np.float64)
            except (ValueError, TypeError):
                return None
        if isinstance(val, dict):
            try:
                return np.array(list(val.values()), dtype=np.float64)
            except (ValueError, TypeError):
                return None
        return None


def validate_compression(
    original: onnx.ModelProto,
    compressed: onnx.ModelProto,
    input_spec: dict[str, list[int]] | None = None,
    architecture: ArchitectureType | None = None,
    n_probes: int = 64,
) -> ValidationResult:
    """Convenience: validate compression quality without any user data."""
    validator = CompressionValidator(n_probes=n_probes)
    return validator.validate(original, compressed, input_spec, architecture)
