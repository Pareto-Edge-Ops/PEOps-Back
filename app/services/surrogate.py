"""Surrogate model service — the PoC's `peops/surrogate/` is an empty stub;
the backend implements it here.

Trains sklearn gradient-boosted regressors on completed Pareto trials
(features: per-op precision/prune/fuse encoding → targets: accuracy / latency /
size). Used to (a) backfill failed latency measurements (ParetoSearch returns
0.0 on ORT errors), (b) report real MAE/R² in the Phase-3 ingestion log, and
(c) cheaply score candidate configs for drift re-optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_PRECISION_ORD = {"FP32": 0.0, "FP16": 1.0, "INT8": 2.0, "INT4": 3.0}


@dataclass
class SurrogateMetrics:
    acc_mae: float
    latency_mae_ms: float
    size_mae_mb: float
    r2: float
    n_train: int


class SurrogateModel:
    """Per-model surrogate over that model's trial cloud."""

    def __init__(self, op_order: list[str], seed: int = 42):
        self.op_order = op_order
        self.seed = seed
        self._models = None
        self.metrics: SurrogateMetrics | None = None

    def featurize(self, config: dict[str, dict]) -> np.ndarray:
        feats: list[float] = []
        for op in self.op_order:
            cfg = config.get(op, {})
            feats.append(_PRECISION_ORD.get(str(cfg.get("precision", "FP32")), 0.0))
            feats.append(float(cfg.get("prune_ratio", 0.0)))
            feats.append(1.0 if cfg.get("fuse") else 0.0)
        return np.asarray(feats, dtype=np.float64)

    def fit(self, points) -> SurrogateMetrics | None:
        """`points`: iterable of peops ParetoPoint (duck-typed:
        .compression_config/.accuracy/.latency_ms/.model_size_bytes)."""
        points = [p for p in points if p.compression_config]
        if len(points) < 4:
            return None

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error, r2_score

        X = np.stack([self.featurize(p.compression_config) for p in points])
        targets = {
            "accuracy": np.array([p.accuracy for p in points]),
            "latency": np.array([p.latency_ms for p in points]),
            "size": np.array([p.model_size_bytes / 1e6 for p in points]),
        }
        self._models = {}
        maes: dict[str, float] = {}
        r2s: list[float] = []
        for key, y in targets.items():
            est = GradientBoostingRegressor(
                n_estimators=60, max_depth=3, random_state=self.seed,
            )
            est.fit(X, y)
            pred = est.predict(X)
            maes[key] = float(mean_absolute_error(y, pred))
            if float(np.var(y)) > 1e-12:
                r2s.append(float(r2_score(y, pred)))
            self._models[key] = est

        self.metrics = SurrogateMetrics(
            acc_mae=round(maes["accuracy"], 4),
            latency_mae_ms=round(maes["latency"], 3),
            size_mae_mb=round(maes["size"], 4),
            r2=round(float(np.mean(r2s)) if r2s else 1.0, 3),
            n_train=len(points),
        )
        return self.metrics

    def predict(self, config: dict[str, dict]) -> dict[str, float] | None:
        if not self._models:
            return None
        x = self.featurize(config).reshape(1, -1)
        return {key: float(est.predict(x)[0]) for key, est in self._models.items()}

    def backfill_latencies(self, points, original_latency_ms: float = 0.0) -> int:
        """Replace 0.0 latency measurements with surrogate predictions.
        Also recomputes the point's `speedup` (ParetoSearch hardcodes 1.0 when
        the measurement failed) so downstream scores stay consistent with the
        repaired latency. Returns the number of points repaired."""
        if not self._models:
            return 0
        repaired = 0
        positive = [p.latency_ms for p in points if p.latency_ms > 0]
        floor = min(positive) * 0.5 if positive else 0.05
        for p in points:
            if p.latency_ms <= 0:
                pred = self.predict(p.compression_config)
                if pred is not None:
                    p.latency_ms = max(floor, pred["latency"])
                    if original_latency_ms > 0:
                        p.speedup = original_latency_ms / p.latency_ms
                    repaired += 1
        return repaired
