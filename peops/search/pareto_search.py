"""3D Pareto Search: Multi-objective optimization over [accuracy, size, latency]
using Optuna with UOSA-guided search space.

Finds the Pareto frontier of compressed model variants that optimally trade off
accuracy retention, model size reduction, and inference latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
import optuna

from peops.core.compression_actions import (
    ActionTranslator,
    CompressionConfig,
    PrecisionLevel,
    get_action_space,
)
from peops.core.uosa import SensitivityProfile
from peops.graph.onnx_analyzer import (
    GraphInfo,
    OnnxAnalyzer,
    OperatorCategory,
    OperatorInfo,
    initializer_bytes,
)
from peops.graph.onnx_transformer import OnnxTransformer

# Categories whose "fuse" flag maps to a real transformer handler
# (bn_fusion / leaf_merging). For every other category the fuse action is a
# no-op, so searching it would only waste TPE dimensions.
_FUSE_EFFECTIVE_CATEGORIES = frozenset({
    OperatorCategory.NORMALIZATION,
    OperatorCategory.TREE_ENSEMBLE,
})


@dataclass
class ParetoPoint:
    """A single point on the Pareto frontier."""
    trial_number: int
    accuracy: float
    model_size_bytes: int
    latency_ms: float
    compression_config: dict[str, dict[str, Any]]
    accuracy_retention: float
    size_ratio: float
    speedup: float
    weights_bytes: int = 0
    weights_ratio: float = 1.0


@dataclass
class ParetoResult:
    """Result of a 3D Pareto search."""
    pareto_points: list[ParetoPoint]
    all_trials: list[ParetoPoint]
    original_accuracy: float
    original_size: int
    original_latency_ms: float
    n_trials: int
    search_time_sec: float

    @property
    def n_pareto(self) -> int:
        return len(self.pareto_points)

    def best_accuracy(self) -> ParetoPoint | None:
        return max(self.pareto_points, key=lambda p: p.accuracy) if self.pareto_points else None

    def best_size(self) -> ParetoPoint | None:
        return min(self.pareto_points, key=lambda p: p.model_size_bytes) if self.pareto_points else None

    def best_latency(self) -> ParetoPoint | None:
        return min(self.pareto_points, key=lambda p: p.latency_ms) if self.pareto_points else None


class ParetoSearch:
    """3D Pareto search engine for model compression."""

    def __init__(
        self,
        n_trials: int = 50,
        seed: int = 42,
        verbose: bool = False,
        allow_pruning: bool = False,
    ):
        self.n_trials = n_trials
        self.seed = seed
        self.verbose = verbose
        self.allow_pruning = allow_pruning
        self._transformer = OnnxTransformer()
        self._translator = ActionTranslator()

    def search(
        self,
        model: onnx.ModelProto,
        graph_info: GraphInfo,
        sensitivity: SensitivityProfile,
        eval_fn: Callable[[onnx.ModelProto], float],
        calibration_input: dict[str, np.ndarray] | None = None,
    ) -> ParetoResult:
        """Run 3D Pareto search.

        Args:
            model: Original ONNX model.
            graph_info: Analyzed graph info.
            sensitivity: UOSA sensitivity profile.
            eval_fn: Function that takes an ONNX model and returns accuracy (0-1).
            calibration_input: Sample input for latency measurement.
        """
        compressible = graph_info.compressible_operators
        if not compressible:
            raise ValueError("No compressible operators found")

        # Baseline measurements
        original_acc = eval_fn(model)
        original_size = model.ByteSize()
        original_weights = max(1, initializer_bytes(model))
        original_latency = self._measure_latency(model, calibration_input)

        if self.verbose:
            print(f"  Baseline: acc={original_acc:.4f}, size={original_size}B, lat={original_latency:.2f}ms")

        # Build per-operator action spaces. Protection is rank-based top-p
        # membership (the configuration validated in the guarantee experiments),
        # and negligible-byte operators collapse to singleton spaces so TPE
        # spends its budget on layers that can actually move the size objective.
        protected = sensitivity.get_protection_set(top_p=0.3)
        total_params = max(1, sum(op.param_count for op in graph_info.operators))
        action_spaces = {}
        for op in compressible:
            action_spaces[op.name] = get_action_space(
                op,
                is_protected=op.name in protected,
                param_share=op.param_count / total_params,
            )

        all_trials: list[ParetoPoint] = []

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            directions=["maximize", "minimize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        start_time = time.time()

        def objective(trial: optuna.Trial) -> tuple[float, float, float]:
            config = {}
            for op in compressible:
                space = action_spaces[op.name]

                if len(space.allowed_precisions) == 1:
                    # Singleton space: fixed config, no search dimension.
                    precision = space.allowed_precisions[0]
                else:
                    precision_choices = [p.value for p in space.allowed_precisions]
                    precision_val = trial.suggest_categorical(
                        f"{op.name}_precision", precision_choices)
                    precision = PrecisionLevel(precision_val)

                if self.allow_pruning:
                    prune_lo, prune_hi = space.prune_ratio_range
                    prune_ratio = trial.suggest_float(
                        f"{op.name}_prune", prune_lo, prune_hi, step=0.1) if prune_hi > 0 else 0.0
                else:
                    prune_ratio = 0.0

                fuse = (
                    trial.suggest_categorical(f"{op.name}_fuse", [True, False])
                    if space.fuse_available and op.category in _FUSE_EFFECTIVE_CATEGORIES
                    else False
                )

                config[op.name] = CompressionConfig(
                    precision_level=precision,
                    prune_ratio=prune_ratio,
                    fuse_enabled=fuse,
                )

            # Apply compression
            actions = []
            for op in compressible:
                op_config = config[op.name]
                if not op_config.is_no_compression():
                    actions.extend(self._translator.translate(op, op_config))

            if not actions:
                return original_acc, float(original_size), original_latency

            compressed = self._transformer.apply(model, actions)

            # Evaluate
            try:
                acc = eval_fn(compressed)
            except Exception:
                return 0.0, float(original_size), original_latency * 10

            size = float(compressed.ByteSize())
            weights = initializer_bytes(compressed)
            latency = self._measure_latency(compressed, calibration_input)

            point = ParetoPoint(
                trial_number=trial.number,
                accuracy=acc,
                model_size_bytes=int(size),
                latency_ms=latency,
                compression_config={
                    op_name: {
                        "precision": cfg.precision_level.name,
                        "prune_ratio": cfg.prune_ratio,
                        "fuse": cfg.fuse_enabled,
                    }
                    for op_name, cfg in config.items()
                },
                accuracy_retention=acc / original_acc if original_acc > 0 else 0,
                size_ratio=size / original_size,
                speedup=original_latency / latency if latency > 0 else 1.0,
                weights_bytes=weights,
                weights_ratio=weights / original_weights,
            )
            all_trials.append(point)

            if self.verbose and trial.number % 10 == 0:
                print(f"    Trial {trial.number}: acc={acc:.4f}, size_r={size/original_size:.3f}, "
                      f"lat={latency:.2f}ms")

            return acc, size, latency

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        search_time = time.time() - start_time

        # Extract Pareto frontier
        pareto_trials = study.best_trials
        pareto_points = []
        for t in pareto_trials:
            matching = [p for p in all_trials if p.trial_number == t.number]
            if matching:
                pareto_points.append(matching[0])

        # If no trials mapped (e.g., all-original configs), use all_trials filtering
        if not pareto_points and all_trials:
            pareto_points = self._compute_pareto_front(all_trials)

        return ParetoResult(
            pareto_points=pareto_points,
            all_trials=all_trials,
            original_accuracy=original_acc,
            original_size=original_size,
            original_latency_ms=original_latency,
            n_trials=self.n_trials,
            search_time_sec=search_time,
        )

    def _measure_latency(
        self,
        model: onnx.ModelProto,
        sample_input: dict[str, np.ndarray] | None,
        n_warmup: int = 3,
        n_measure: int = 10,
    ) -> float:
        """Measure inference latency in milliseconds."""
        if sample_input is None:
            return 0.0

        try:
            session = ort.InferenceSession(model.SerializeToString())
            output_names = [o.name for o in session.get_outputs()]

            for _ in range(n_warmup):
                session.run(output_names, sample_input)

            times = []
            for _ in range(n_measure):
                t0 = time.perf_counter()
                session.run(output_names, sample_input)
                times.append((time.perf_counter() - t0) * 1000)

            return float(np.median(times))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_pareto_front(points: list[ParetoPoint]) -> list[ParetoPoint]:
        """Compute Pareto frontier: maximize accuracy, minimize size and latency."""
        pareto = []
        for p in points:
            dominated = False
            for q in points:
                if (q.accuracy >= p.accuracy and
                    q.model_size_bytes <= p.model_size_bytes and
                    q.latency_ms <= p.latency_ms and
                    (q.accuracy > p.accuracy or
                     q.model_size_bytes < p.model_size_bytes or
                     q.latency_ms < p.latency_ms)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)
        return pareto
