"""WindowAggregator: classifier detection, reservoir bounds, robustness."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from peops_sdk.stats import WindowAggregator


def test_classifier_shaped_output():
    agg = WindowAggregator()
    rng = np.random.default_rng(1)
    for _ in range(50):
        agg.observe({"x": rng.standard_normal((1, 4))},
                    np.array([[0.1, 0.2, 5.0, 0.1]]))
    w = agg.flush()
    assert w["n"] == 50
    assert w["output"]["classDist"] == {"2": 1.0}
    assert w["output"]["top1ConfMean"] > 0.9
    assert sum(w["output"]["hist"]) > 0


def test_non_classifier_output_gets_value_hist():
    agg = WindowAggregator()
    for _ in range(5):
        agg.observe({"x": np.zeros((1, 2))}, np.zeros((1, 8, 8)))
    w = agg.flush()
    assert "classDist" not in w["output"]
    assert len(w["output"]["hist"]) == 16


def test_input_stats_with_nans():
    agg = WindowAggregator()
    x = np.array([1.0, 2.0, np.nan, 3.0])
    agg.observe({"x": x}, None)
    w = agg.flush()
    s = w["inputs"]["x"]
    assert s["nanPct"] == pytest.approx(25.0, abs=0.1)
    assert s["mean"] == pytest.approx(2.0, abs=1e-6)


def test_flush_resets_and_empty_returns_none():
    agg = WindowAggregator()
    agg.observe({"x": np.zeros(3)}, None)
    assert agg.flush() is not None
    assert agg.flush() is None


def test_reservoir_bounded():
    agg = WindowAggregator()
    for _ in range(10_000):
        agg.observe({"x": np.zeros(2)}, None)
    assert len(agg._inputs) <= 32
    assert agg.flush()["n"] == 10_000
