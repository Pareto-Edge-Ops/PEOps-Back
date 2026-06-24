"""Unit tests — op_kind table, layout algorithm, pareto mapper, surrogate."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.mappers.layout import compute_layout
from app.services.mappers.op_kind import find_attention_ops, kind_for
from app.services.mappers.pareto_mapper import (
    baseline_experiment,
    derive_quant_label,
    map_pareto,
    trial_score,
)
from app.services.surrogate import SurrogateModel

# ── op_kind ─────────────────────────────────────────────────────────────────

def test_direct_kinds():
    assert kind_for("c1", "Conv") == "conv"
    assert kind_for("b1", "BatchNormalization") == "bn"
    assert kind_for("l1", "LayerNormalization") == "norm"
    assert kind_for("r1", "Relu") == "relu"
    assert kind_for("g1", "Gelu") == "relu"
    assert kind_for("s1", "Softmax") == "softmax"
    assert kind_for("p1", "MaxPool") == "pool"
    assert kind_for("m1", "Gemm") == "dense"
    assert kind_for("e1", "Gather") == "embed"
    assert kind_for("l2", "LSTM") == "lstm"
    assert kind_for("u1", "Resize") == "upsample"
    assert kind_for("t1", "TreeEnsembleClassifier") == "dense"


def test_data_movement_dropped():
    for op_type in ("Reshape", "Transpose", "Concat", "Flatten", "Cast", "Add", "Mul"):
        assert kind_for("x", op_type) is None


def test_attention_detection_strict_window():
    ordered = [("q_mm", "MatMul"), ("sm", "Softmax"), ("v_mm", "MatMul"), ("out", "Gemm")]
    attn = find_attention_ops(ordered)
    assert attn == {"q_mm", "v_mm"}
    assert kind_for("q_mm", "MatMul", in_attention_window=True) == "attn"
    assert kind_for("plain", "MatMul") == "dense"
    assert kind_for("/attn/qkv/MatMul", "MatMul") == "attn"  # name heuristic


def test_attention_detection_real_export_sequence():
    """Real torch exports insert a scale Mul between QK^T and Softmax —
    this is the actual TinyAttention topo order from torch 2.12."""
    ordered = [
        ("/q/MatMul", "MatMul"), ("/k/MatMul", "MatMul"), ("/v/MatMul", "MatMul"),
        ("/Constant", "Constant"), ("/Transpose", "Transpose"),
        ("/MatMul", "MatMul"), ("/Mul", "Mul"), ("/Softmax", "Softmax"),
        ("/MatMul_1", "MatMul"), ("/Add", "Add"),
        ("/norm/LayerNormalization", "LayerNormalization"),
        ("/ffn/MatMul", "MatMul"), ("/Add_1", "Add"), ("/act/Relu", "Relu"),
        ("/ReduceMean", "ReduceMean"), ("/head/Gemm", "Gemm"),
    ]
    attn = find_attention_ops(ordered)
    assert "/MatMul" in attn and "/MatMul_1" in attn  # score + context MatMuls
    # q/k/v projections caught by the name tokens even outside the window
    assert kind_for("/q/MatMul", "MatMul") == "attn"
    assert kind_for("/k/MatMul", "MatMul") == "attn"
    assert kind_for("/v/MatMul", "MatMul") == "attn"
    # ffn / head stay dense
    assert kind_for("/ffn/MatMul", "MatMul") == "dense"
    assert kind_for("/head/Gemm", "Gemm") == "dense"


# ── layout ──────────────────────────────────────────────────────────────────

def test_layout_chain():
    nodes = ["a", "b", "c"]
    pos = compute_layout(nodes, [("a", "b"), ("b", "c")], {n: i for i, n in enumerate(nodes)})
    assert [pos[n].depth for n in nodes] == [0, 1, 2]
    assert all(pos[n].col == 0 for n in nodes)
    assert all(pos[n].z_col is None for n in nodes)


def test_layout_diamond():
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    pos = compute_layout(nodes, edges, {n: i for i, n in enumerate(nodes)})
    assert pos["b"].depth == pos["c"].depth == 1
    assert pos["d"].depth == 2
    assert pos["b"].col == -pos["c"].col != 0  # symmetric offsets


def test_layout_grid_for_many_siblings():
    sibs = [f"s{i}" for i in range(6)]
    nodes = ["a", *sibs, "z"]
    edges = [("a", s) for s in sibs] + [(s, "z") for s in sibs]
    pos = compute_layout(nodes, edges, {n: i for i, n in enumerate(nodes)})
    assert any(pos[s].z_col is not None for s in sibs)  # y-z grid engaged


# ── pareto mapper ───────────────────────────────────────────────────────────

@dataclass
class FakePoint:
    trial_number: int
    accuracy: float
    model_size_bytes: int
    latency_ms: float
    compression_config: dict = field(default_factory=dict)
    accuracy_retention: float = 1.0
    size_ratio: float = 1.0
    speedup: float = 1.0


@dataclass
class FakeResult:
    pareto_points: list
    all_trials: list
    original_accuracy: float = 1.0
    original_size: int = 2_000_000
    original_latency_ms: float = 4.0
    n_trials: int = 4


def test_quant_label():
    assert derive_quant_label({}) == "FP32 (baseline)"
    assert derive_quant_label({"a": {"precision": "FP32"}}) == "FP32 (baseline)"
    assert derive_quant_label({"a": {"precision": "INT8"}, "b": {"precision": "INT8"}}) == "INT8"
    mixed = derive_quant_label({
        "a": {"precision": "INT8", "prune_ratio": 0.3},
        "b": {"precision": "FP16", "prune_ratio": 0.1},
    })
    assert mixed.startswith("INT8-mix") and "r0.2" in mixed
    fused = derive_quant_label({"a": {"precision": "FP16", "fuse": True}})
    assert "+fuse" in fused


def test_trial_score_bounds():
    assert trial_score(1.0, 1.0, 1.0, 1.0) == 50.0  # retention only
    assert trial_score(1.0, 1.0, 0.5, 2.0) == 75.0
    assert 0 <= trial_score(0.0, 1.0, 2.0, 0.1) <= 100


def test_map_pareto():
    good = FakePoint(0, 0.99, 1_000_000, 2.0,
                     {"op": {"precision": "INT8"}}, 0.99, 0.5, 2.0)
    bad = FakePoint(1, 0.90, 2_000_000, 4.0,
                    {"op": {"precision": "FP16"}}, 0.90, 1.0, 1.0)
    res = FakeResult(pareto_points=[good], all_trials=[good, bad])
    exp = map_pareto("m_x", "Demo Model", "exp_1", res)
    assert exp.baseAccuracy == 100.0
    assert exp.trials[0].accuracy == 99.0
    assert exp.trials[0].size == 1.0          # bytes → MB
    assert exp.trials[0].onFrontier is True
    assert exp.trials[1].onFrontier is False
    assert exp.trials[0].quant == "INT8"
    assert exp.iterCurrent == 2 and exp.iterTotal == 4


def test_baseline_experiment():
    exp = baseline_experiment("m_x", "Demo", "exp_0", size_mb=1.5, latency_ms=2.0, quality=0.98)
    assert len(exp.trials) == 1
    assert exp.trials[0].quant == "FP32 (baseline)"
    assert exp.trials[0].onFrontier is True
    # The single baseline trial IS the served artifact.
    assert exp.trials[0].trialNumber == 0
    assert exp.servedTrialNumber == 0


def test_map_pareto_served_trial_number():
    good = FakePoint(0, 0.99, 1_000_000, 2.0,
                     {"op": {"precision": "INT8"}}, 0.99, 0.5, 2.0)
    res = FakeResult(pareto_points=[good], all_trials=[good])
    # Served trial is surfaced so the Studio can badge the default download…
    assert map_pareto("m_x", "Demo", "exp_1", res, served_trial_number=0).servedTrialNumber == 0
    # …and is null when the served artifact is a ladder/fallback (not a trial).
    assert map_pareto("m_x", "Demo", "exp_1", res).servedTrialNumber is None


# ── surrogate ───────────────────────────────────────────────────────────────

def test_surrogate_fit_predict_backfill():
    ops = ["op_a", "op_b"]
    points = []
    for i in range(10):
        prec = ["FP32", "FP16", "INT8"][i % 3]
        cfg = {"op_a": {"precision": prec, "prune_ratio": 0.0},
               "op_b": {"precision": "FP16", "prune_ratio": 0.1 * (i % 2)}}
        points.append(FakePoint(
            trial_number=i,
            accuracy=1.0 - 0.01 * (i % 3),
            model_size_bytes=2_000_000 - i * 50_000,
            latency_ms=0.0 if i == 9 else 1.0 + 0.1 * i,
            compression_config=cfg,
        ))
    sur = SurrogateModel(op_order=ops, seed=42)
    metrics = sur.fit(points)
    assert metrics is not None
    assert metrics.n_train == 10
    assert metrics.latency_mae_ms >= 0
    pred = sur.predict(points[0].compression_config)
    assert pred and {"accuracy", "latency", "size"} == set(pred)
    repaired = sur.backfill_latencies(points, original_latency_ms=2.0)
    assert repaired == 1
    assert points[9].latency_ms > 0
    # speedup must be recomputed from the repaired latency, not left at 1.0
    assert points[9].speedup == 2.0 / points[9].latency_ms


def test_surrogate_too_few_points():
    sur = SurrogateModel(op_order=["a"], seed=1)
    assert sur.fit([FakePoint(0, 1.0, 1, 1.0, {"a": {"precision": "INT8"}})]) is None
