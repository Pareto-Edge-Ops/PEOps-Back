"""Adaptive Optuna trial budget + deterministic 2D search.

Covers the per-model budget formula, the effective-dimensionality count, the
hypervolume helper used by the optional early stop, and — most importantly —
that the 2-objective search is now reproducible at a fixed seed (the property
that motivated dropping wall-clock latency from the TPE objective).
"""
from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from astra.graph.onnx_analyzer import OnnxAnalyzer, OperatorCategory
from astra.core.compression_actions import ActionSpace, PrecisionLevel
from astra.search.pareto_search import ParetoSearch, _dominated_hv_2d


# ───────────────────────── budget formula ─────────────────────────
def test_budget_scales_with_dimensionality():
    s = ParetoSearch(n_trials=150, adaptive=True, trials_per_dim=10,
                     startup_trials=10, min_trials=30)
    assert s._budget(2) == 30      # 10*2+10=30 -> floor
    assert s._budget(4) == 50      # 10*4+10
    assert s._budget(9) == 100
    assert s._budget(14) == 150    # 10*14+10 = 150 (== ceiling)
    assert s._budget(40) == 150    # clamped to ceiling


def test_budget_ceiling_is_hard_cap_even_below_floor():
    # A small caller ceiling must win even when below min_trials.
    s = ParetoSearch(n_trials=15, adaptive=True, min_trials=30)
    assert s._budget(2) == 15
    assert s._budget(50) == 15


def test_budget_fixed_when_not_adaptive():
    s = ParetoSearch(n_trials=42, adaptive=False)
    assert s._budget(2) == 42
    assert s._budget(99) == 42


# ───────────────────────── effective dimensionality ─────────────────────────
def _space(precisions, *, category, fuse):
    return ActionSpace(
        operator_name="op",
        category=category,
        allowed_precisions=list(precisions),
        prune_ratio_range=(0.0, 0.0),
        fuse_available=fuse,
        approx_rank_range=None,
    )


class _Op:
    def __init__(self, name, category):
        self.name = name
        self.category = category


def test_effective_dimensionality_counts_precision_and_fuse():
    P = PrecisionLevel
    ops = [
        _Op("conv", OperatorCategory.DENSE_COMPUTE),     # 3 prec -> +1, dense fuse not effective -> +0
        _Op("bn", OperatorCategory.NORMALIZATION),       # 2 prec -> +1, fuse-effective -> +1
        _Op("relu", OperatorCategory.ACTIVATION),        # 1 prec -> +0
    ]
    spaces = {
        "conv": _space([P.FP32, P.FP16, P.INT8], category=OperatorCategory.DENSE_COMPUTE, fuse=True),
        "bn": _space([P.FP32, P.FP16], category=OperatorCategory.NORMALIZATION, fuse=True),
        "relu": _space([P.FP32], category=OperatorCategory.ACTIVATION, fuse=True),
    }
    D, bits = ParetoSearch._effective_dimensionality(spaces, ops, allow_pruning=False)
    assert D == 3                       # conv(1) + bn(1 prec + 1 fuse) = 3
    assert bits == pytest.approx(np.log2(3) + np.log2(2) + 1.0)


# ───────────────────────── hypervolume helper ─────────────────────────
def test_hv_single_point():
    # box from [0.5,0.5] up to ref [1,1] = 0.5*0.5 = 0.25
    assert _dominated_hv_2d([[0.5, 0.5]], ref=(1.0, 1.0)) == pytest.approx(0.25)


def test_hv_two_points_inclusion_exclusion():
    # [0.2,0.8] and [0.8,0.2] vs ref [1,1]:
    #   0.8*0.2 + 0.2*0.8 - 0.2*0.2 = 0.16 + 0.16 - 0.04 = 0.28
    hv = _dominated_hv_2d([[0.2, 0.8], [0.8, 0.2]], ref=(1.0, 1.0))
    assert hv == pytest.approx(0.28)


def test_hv_dominated_point_ignored():
    hv = _dominated_hv_2d([[0.5, 0.5], [0.7, 0.7]], ref=(1.0, 1.0))
    assert hv == pytest.approx(0.25)


# ───────────────────────── determinism (the headline property) ───────────────
def _tiny_cnn(n_conv=4, ch=8, seed=0):
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    cin, x = 3, "input"
    for i in range(n_conv):
        w = (rng.standard_normal((ch, cin, 3, 3)) * 0.1).astype(np.float32)
        b = np.zeros((ch,), np.float32)
        inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
        nodes.append(helper.make_node("Conv", [x, f"w{i}", f"b{i}"], [f"c{i}"],
                                      kernel_shape=[3, 3], pads=[1, 1, 1, 1], name=f"conv{i}"))
        nodes.append(helper.make_node("Relu", [f"c{i}"], [f"r{i}"], name=f"relu{i}"))
        x, cin = f"r{i}", ch
    nodes.append(helper.make_node("GlobalAveragePool", [x], ["gap"], name="gap"))
    nodes.append(helper.make_node("Flatten", ["gap"], ["flat"], axis=1, name="flat"))
    inits += [numpy_helper.from_array((rng.standard_normal((ch, 10)) * 0.1).astype(np.float32), "wg"),
              numpy_helper.from_array(np.zeros((10,), np.float32), "bg")]
    nodes.append(helper.make_node("Gemm", ["flat", "wg", "bg"], ["output"], name="gemm"))
    g = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def _run(model, seed):
    gi = OnnxAnalyzer().analyze(model)
    from astra.core.calibration_generator import CalibrationGenerator
    from astra.core.uosa import compute_uosa
    from astra.core.validation import CompressionValidator
    from astra.graph.model_detector import ModelDetector

    spec = {"input": [1, 3, 16, 16]}
    cal = CalibrationGenerator(n_probes=8, seed=seed).generate(
        model, spec, ModelDetector().detect(model).architecture)
    profile = compute_uosa(model, cal.probes, gi)
    validator = CompressionValidator(n_probes=8, seed=seed)

    def ev(cm):
        try:
            return validator.validate(model, cm, spec).quality_score
        except Exception:
            return 0.0

    search = ParetoSearch(n_trials=60, seed=seed, adaptive=True)
    return search.search(model, gi, profile, ev, cal.probes[0])


def _signature(result):
    return tuple((p.trial_number, round(p.accuracy, 6), p.model_size_bytes)
                 for p in sorted(result.all_trials, key=lambda p: p.trial_number))


def test_search_is_deterministic_at_fixed_seed():
    """Same seed → identical trial trajectory (configs, accuracy, size). This is
    the property that the noisy 3-objective search did NOT have."""
    model = _tiny_cnn()
    r1 = _run(model, seed=42)
    r2 = _run(model, seed=42)
    assert _signature(r1) == _signature(r2)
    # the chosen compressed model (smallest on the frontier) is reproducible
    assert r1.best_size().model_size_bytes == r2.best_size().model_size_bytes
    # budget telemetry is populated
    assert r1.effective_dim > 0
    assert r1.n_trials == r1.n_trials_completed  # ran the planned budget (no early stop)
