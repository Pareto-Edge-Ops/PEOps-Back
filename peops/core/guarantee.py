"""Guarantee-by-Construction Compression.

A progressive compression ladder with a data-free validation gate. Rungs are
tried from most aggressive to most conservative:

  R3  INT8_uniform     real ORT dynamic INT8, no protection
  R2  INT8_uosa_mixed  real ORT dynamic INT8, UOSA top-p sensitive ops excluded
  R1  FP16             real float16 weight conversion (onnxconverter-common)
  R0  ORIGINAL         the input model, untouched

Gate: the first rung whose candidate satisfies

    OFS(original, candidate; P_gate) >= tau
    AND  size(candidate) < size(original)

is returned, where P_gate POOLS k independent calibration probe draws
(default k=3, i.e. 3*n_probes gate probes). Pooling guards against
probe-sampling variance: with a single 64-probe draw, INT8 MLP-Adult showed
bimodal per-draw OFS (0.48–0.999) — a lucky draw could admit a candidate
whose fidelity does not generalize across the probe distribution. The pooled
estimate concentrates near the true expected fidelity and rejects such
candidates consistently.

The final rung R0 trivially satisfies OFS = 1.0 >= tau, so the returned
model ALWAYS satisfies the fidelity floor (Proposition 1) — with worst-case
compression ratio 1.0.

Honest scope: the guarantee is with respect to output fidelity on the
calibration probe distribution, NOT downstream task accuracy on unseen data.
Any runtime error while building or validating a rung is treated as rejection,
so control always reaches R0.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
import onnx

from peops.core.calibration_generator import CalibrationGenerator
from peops.core.compression_actions import (
    ActionTranslator,
    CompressionConfig,
    PrecisionLevel,
)
from peops.core.uosa import SensitivityProfile, compute_uosa
from peops.core.validation import CompressionValidator
from peops.graph.model_detector import ArchitectureType
from peops.graph.onnx_analyzer import GraphInfo, OnnxAnalyzer, OperatorCategory
from peops.graph.onnx_transformer import OnnxTransformer

_STANDARD_CATEGORIES = (
    OperatorCategory.DENSE_COMPUTE,
    OperatorCategory.NORMALIZATION,
    OperatorCategory.EMBEDDING,
)
_ML_CATEGORIES = (
    OperatorCategory.TREE_ENSEMBLE,
    OperatorCategory.LINEAR_MODEL,
    OperatorCategory.SVM,
)

RUNG_INT8_UNIFORM = "INT8_uniform"
RUNG_INT8_UOSA = "INT8_uosa_mixed"
RUNG_W8 = "W8_weight_only"
RUNG_FP16 = "FP16"
RUNG_ORIGINAL = "ORIGINAL"

# v1: the ladder validated in the 2026-06-06 guarantee experiments (18-model
# zoo, 0/18 violations). Kept as the default so published claims stay tied to
# the exact configuration that produced them.
LADDER_V1 = [RUNG_INT8_UNIFORM, RUNG_INT8_UOSA, RUNG_FP16, RUNG_ORIGINAL]
# v2 adds W8 (weight-only INT8 QDQ storage — ~4x weight bytes, no activation
# quantization) between the dynamic-INT8 rungs and FP16: more aggressive than
# FP16 in size, typically higher fidelity than dynamic INT8 because
# activations stay fp32. Candidates that inflated under dynamic INT8
# (RandTransformer 1.30x) or failed its fidelity gate (GRU) get a real
# compression path before falling back.
LADDER_V2 = [RUNG_INT8_UNIFORM, RUNG_INT8_UOSA, RUNG_W8, RUNG_FP16, RUNG_ORIGINAL]
LADDERS = {"v1": LADDER_V1, "v2": LADDER_V2}

LADDER = LADDER_V1  # backward-compatible alias


@dataclass
class RungReport:
    """Outcome of attempting one ladder rung."""
    rung: str
    attempted: bool
    accepted: bool
    reason: str
    output_fidelity: float | None = None
    quality_score: float | None = None
    size_bytes: int | None = None
    size_ratio: float | None = None


@dataclass
class GuaranteeResult:
    """Result of guarantee_compress: the returned model always satisfies
    OFS >= tau on the gate probes (Proposition 1)."""
    model: onnx.ModelProto = field(repr=False)
    rung: str
    output_fidelity: float
    size_ratio: float
    tau: float
    n_probes_gate: int
    seed: int
    rung_reports: list[RungReport]
    uosa_time_ms: float

    @property
    def is_compressed(self) -> bool:
        return self.rung != RUNG_ORIGINAL

    def certificate(self) -> str:
        """Human-readable fidelity certificate."""
        lines = [
            "─" * 56,
            "  PEOps Guarantee Certificate",
            "─" * 56,
            f"  Fidelity floor (tau):   {self.tau}",
            f"  Gate probes:            {self.n_probes_gate} (seed={self.seed})",
            f"  Selected rung:          {self.rung}",
            f"  Achieved OFS:           {self.output_fidelity:.6f}",
            f"  Size ratio:             {self.size_ratio:.4f}",
            "  Ladder trace:",
        ]
        for r in self.rung_reports:
            ofs = f"OFS={r.output_fidelity:.4f}" if r.output_fidelity is not None else "OFS=n/a"
            ratio = f"size={r.size_ratio:.3f}x" if r.size_ratio is not None else "size=n/a"
            mark = "✓" if r.accepted else "✗"
            lines.append(f"    {mark} {r.rung:18s} {ofs:14s} {ratio:14s} {r.reason}")
        lines.append("─" * 56)
        return "\n".join(lines)


def _quantize_int8_real(
    model: onnx.ModelProto,
    nodes_to_exclude: list[str] | None,
) -> onnx.ModelProto:
    """Real ONNX Runtime dynamic INT8 quantization (file-based round trip)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization import preprocess as qpreprocess

    tmpdir = tempfile.mkdtemp(prefix="peops_guarantee_")
    src = os.path.join(tmpdir, "model.onnx")
    prep = os.path.join(tmpdir, "model_prep.onnx")
    dst = os.path.join(tmpdir, "model_int8.onnx")
    try:
        onnx.save(model, src)
        try:
            qpreprocess.quant_pre_process(src, prep)
            source = prep
        except Exception:
            source = src

        kwargs = dict(model_input=source, model_output=dst, weight_type=QuantType.QInt8)
        if nodes_to_exclude:
            kwargs["nodes_to_exclude"] = nodes_to_exclude
        quantize_dynamic(**kwargs)
        return onnx.load(dst)
    finally:
        for p in (src, prep, dst):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmpdir)


def _quantize_ml_native(
    model: onnx.ModelProto,
    graph_info: GraphInfo,
    protected: set[str],
    precision: PrecisionLevel,
) -> onnx.ModelProto:
    """PEOps native attribute quantization for ai.onnx.ml operators."""
    translator = ActionTranslator()
    transformer = OnnxTransformer()
    actions = []
    for op in graph_info.compressible_operators:
        if op.category not in _ML_CATEGORIES or op.name in protected:
            continue
        actions.extend(translator.translate(op, CompressionConfig(precision_level=precision)))
    return transformer.apply(model, actions) if actions else model


def _convert_fp16_real(model: onnx.ModelProto) -> onnx.ModelProto:
    """Real float16 weight conversion (halves standard-domain weight storage)."""
    from onnxconverter_common import float16

    return float16.convert_float_to_float16(model, keep_io_types=True)


def _quantize_w8_weight_only(
    model: onnx.ModelProto,
    graph_info: GraphInfo,
) -> onnx.ModelProto | None:
    """Weight-only INT8 QDQ storage for standard-domain dense/embedding ops.

    Weights are stored as int8 initializers behind DequantizeLinear (~4x byte
    reduction); activations are untouched, so compute runs at fp32 fidelity.
    Returns None when no standard-domain weight surface exists.
    """
    translator = ActionTranslator()
    transformer = OnnxTransformer()
    actions = []
    for op in graph_info.compressible_operators:
        if op.category not in (OperatorCategory.DENSE_COMPUTE, OperatorCategory.EMBEDDING):
            continue
        actions.extend(
            translator.translate(op, CompressionConfig(precision_level=PrecisionLevel.INT8)))
    if not actions:
        return None
    return transformer.apply(model, actions)


def _build_candidate(
    rung: str,
    model: onnx.ModelProto,
    graph_info: GraphInfo,
    profile: SensitivityProfile,
    top_p: float,
) -> onnx.ModelProto | None:
    """Build the candidate model for a rung. None = rung not applicable."""
    has_standard = any(
        op.category in _STANDARD_CATEGORIES for op in graph_info.compressible_operators)
    has_ml = any(
        op.category in _ML_CATEGORIES for op in graph_info.compressible_operators)

    if not has_standard and not has_ml:
        return None  # no compressible surface at all

    if rung == RUNG_INT8_UNIFORM:
        protected: set[str] = set()
    elif rung == RUNG_INT8_UOSA:
        if not profile.results:
            return None
        protected = profile.get_protection_set(top_p=top_p)
    elif rung == RUNG_W8:
        if not has_standard:
            return None  # weight-only path needs standard-domain weights
        return _quantize_w8_weight_only(model, graph_info)
    elif rung == RUNG_FP16:
        if has_standard:
            return _convert_fp16_real(model)
        return _quantize_ml_native(model, graph_info, set(), PrecisionLevel.FP16)
    else:
        raise ValueError(f"unknown rung: {rung}")

    candidate = model
    if has_standard:
        candidate = _quantize_int8_real(candidate, list(protected) or None)
    if has_ml:
        candidate = _quantize_ml_native(candidate, graph_info, protected, PrecisionLevel.INT8)
    return candidate


def build_gate_probes(
    model: onnx.ModelProto,
    input_spec: dict[str, list[int]],
    architecture: ArchitectureType | None = None,
    n_probes: int = 64,
    seed: int = 42,
    n_gate_draws: int = 3,
) -> list[dict[str, np.ndarray]]:
    """Pooled gate probe set: k independent draws (seeds disjoint from the
    UOSA seed) concatenated into one large set — low-variance OFS estimate."""
    gate_probes: list[dict[str, np.ndarray]] = []
    for i in range(max(1, n_gate_draws)):
        gen = CalibrationGenerator(n_probes=n_probes, seed=seed + 7919 * (i + 1))
        gate_probes.extend(gen.generate(model, input_spec, architecture).probes)
    return gate_probes


def gate_check(
    original: onnx.ModelProto,
    candidate: onnx.ModelProto,
    gate_probes: list[dict[str, np.ndarray]],
    input_spec: dict[str, list[int]] | None = None,
    architecture: ArchitectureType | None = None,
    seed: int = 42,
) -> tuple[float, float]:
    """(OFS, Q) of a candidate on the pooled gate probes (error → 0.0).

    The same check guarantee_compress applies per rung — exposed so callers
    can certify externally-built candidates (e.g. Pareto picks) against the
    identical gate before serving them.
    """
    validator = CompressionValidator(n_probes=len(gate_probes), seed=seed)
    try:
        vr = validator.validate(original, candidate, input_spec,
                                architecture, probes=gate_probes)
        return vr.output_fidelity, vr.quality_score
    except Exception:
        return 0.0, 0.0


def guarantee_compress(
    model: onnx.ModelProto,
    tau: float = 0.95,
    n_probes: int = 64,
    seed: int = 42,
    top_p: float = 0.3,
    n_gate_draws: int = 3,
    graph_info: GraphInfo | None = None,
    profile: SensitivityProfile | None = None,
    architecture: ArchitectureType | None = None,
    input_spec: dict[str, list[int]] | None = None,
    ladder: str | list[str] = "v1",
    gate_probes: list[dict[str, np.ndarray]] | None = None,
    verbose: bool = False,
) -> GuaranteeResult:
    """Compress with a structural fidelity guarantee.

    Returns the most aggressive ladder rung whose candidate satisfies
    OFS >= tau on a pooled gate probe set (`n_gate_draws` independent draws of
    `n_probes` each) AND is strictly smaller than the original. Falls back to
    the original model (always admissible).

    `ladder` selects the rung sequence: "v1" (validated 2026-06-06 zoo
    configuration, default) or "v2" (adds the W8 weight-only rung); a custom
    rung list may be passed directly. Passing `gate_probes` reuses an
    externally built pooled probe set (see `build_gate_probes`).
    """
    if graph_info is None:
        graph_info = OnnxAnalyzer().analyze(model)
    if input_spec is None:
        input_spec = _extract_input_spec(model)

    rungs = LADDERS[ladder] if isinstance(ladder, str) else list(ladder)
    if rungs[-1] != RUNG_ORIGINAL:
        rungs = rungs + [RUNG_ORIGINAL]

    uosa_time_ms = 0.0
    if profile is None:
        gen = CalibrationGenerator(n_probes=n_probes, seed=seed)
        probes_uosa = gen.generate(model, input_spec, architecture).probes
        t0 = time.time()
        profile = compute_uosa(model, probes_uosa, graph_info, seed=seed)
        uosa_time_ms = (time.time() - t0) * 1000

    if gate_probes is None:
        gate_probes = build_gate_probes(
            model, input_spec, architecture,
            n_probes=n_probes, seed=seed, n_gate_draws=n_gate_draws)

    def gate_ofs(candidate: onnx.ModelProto) -> tuple[float, float]:
        return gate_check(model, candidate, gate_probes, input_spec, architecture, seed)

    n_gate_probes = len(gate_probes)
    orig_size = model.ByteSize()
    reports: list[RungReport] = []

    for rung in rungs[:-1]:
        try:
            candidate = _build_candidate(rung, model, graph_info, profile, top_p)
        except Exception as e:
            reports.append(RungReport(
                rung=rung, attempted=True, accepted=False,
                reason=f"build error: {type(e).__name__}"))
            continue

        if candidate is None:
            reports.append(RungReport(
                rung=rung, attempted=False, accepted=False, reason="not applicable"))
            continue

        ofs, quality = gate_ofs(candidate)

        cand_size = candidate.ByteSize()
        ratio = cand_size / orig_size if orig_size > 0 else 1.0

        if ofs < tau:
            reason = f"fidelity below tau ({ofs:.4f} < {tau})"
            accepted = False
        elif cand_size >= orig_size:
            reason = f"no size reduction ({ratio:.3f}x)"
            accepted = False
        else:
            reason = "accepted"
            accepted = True

        reports.append(RungReport(
            rung=rung, attempted=True, accepted=accepted, reason=reason,
            output_fidelity=ofs, quality_score=quality,
            size_bytes=cand_size, size_ratio=ratio))

        if verbose:
            print(f"  [{rung}] OFS={ofs:.4f} size={ratio:.3f}x → {reason}")

        if accepted:
            return GuaranteeResult(
                model=candidate, rung=rung, output_fidelity=ofs, size_ratio=ratio,
                tau=tau, n_probes_gate=n_gate_probes, seed=seed,
                rung_reports=reports, uosa_time_ms=uosa_time_ms)

    # R0 — the original is always admissible: OFS = 1.0 on every draw, ratio = 1.0.
    reports.append(RungReport(
        rung=RUNG_ORIGINAL, attempted=True, accepted=True, reason="fallback to original",
        output_fidelity=1.0, quality_score=1.0,
        size_bytes=orig_size, size_ratio=1.0))
    if verbose:
        print(f"  [{RUNG_ORIGINAL}] fallback → guaranteed (OFS=1.0, size=1.0x)")

    return GuaranteeResult(
        model=model, rung=RUNG_ORIGINAL, output_fidelity=1.0, size_ratio=1.0,
        tau=tau, n_probes_gate=n_gate_probes, seed=seed,
        rung_reports=reports, uosa_time_ms=uosa_time_ms)


def _extract_input_spec(model: onnx.ModelProto) -> dict[str, list[int]]:
    init_names = {i.name for i in model.graph.initializer}
    spec = {}
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        shape = []
        try:
            for d in inp.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.dim_value > 0 else 1)
        except Exception:
            shape = [1]
        spec[inp.name] = shape
    return spec
