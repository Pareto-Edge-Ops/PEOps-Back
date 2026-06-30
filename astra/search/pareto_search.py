"""2D Pareto Search: multi-objective optimization over [accuracy, size] using
Optuna with a UOSA-guided search space, plus a per-model adaptive trial budget.

Finds the Pareto frontier of compressed model variants that trade off accuracy
retention against model-size reduction. Inference latency is measured per trial
and attached to every candidate for reporting, but it is intentionally NOT a TPE
search objective: wall-clock latency is noisy, and feeding that noise back into
the sampler made the whole search non-reproducible (same seed -> different
trajectory). Optimizing only the two deterministic objectives makes the search
fully reproducible and lets the trial budget scale with the real search
dimensionality D instead of a fixed guess.

Trial budget:
  D          = number of live Optuna dimensions (suggest_* calls) = sum over
               compressible ops of (allowed_precisions>1) + (fuse-effective)
               + (prune dim, when pruning is enabled). Computed for free before
               the study runs, from the already-built action spaces.
  n_trials   = clamp(round(per_dim * D + startup), min_trials, max_trials)

Empirically (see scripts/calibrate_trial_budget.py) the deterministic frontier
converges by ~140 trials even for the most complex models (D~24) and by <10 for
trivial ones, so a flat 150 over-searched simple models while a D-scaled budget
tracks each model's real need. An optional HV-plateau early stop is available
(early_stop=True) but defaults OFF: the frontier exhibits long plateaus followed
by unpredictable late jumps, so a plateau stop cannot preserve frontier quality.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
import optuna

from astra.core.compression_actions import (
    ActionTranslator,
    CompressionConfig,
    PrecisionLevel,
    get_action_space,
)
from astra.core.uosa import SensitivityProfile
from astra.graph.onnx_analyzer import (
    GraphInfo,
    OnnxAnalyzer,
    OperatorCategory,
    OperatorInfo,
    initializer_bytes,
)
from astra.graph.onnx_transformer import OnnxTransformer

# Categories whose "fuse" flag maps to a real transformer handler
# (bn_fusion / leaf_merging). For every other category the fuse action is a
# no-op, so searching it would only waste TPE dimensions.
_FUSE_EFFECTIVE_CATEGORIES = frozenset({
    OperatorCategory.NORMALIZATION,
    OperatorCategory.TREE_ENSEMBLE,
})

# Reference (nadir) point for the normalized 2D minimization hypervolume used by
# the optional early stop. Both axes are normalized to ~[0, 1] (1.1 leaves head
# room so the "do nothing" baseline at [0, 1] still contributes volume).
_HV_REF = (1.1, 1.1)


def _load_hv_backend():
    """Optuna's exact hypervolume helper if importable, else None."""
    try:
        from optuna._hypervolume import compute_hypervolume
        return compute_hypervolume
    except Exception:
        return None


_OPTUNA_HV = _load_hv_backend()


def _pareto_filter_2d(pts: np.ndarray) -> np.ndarray:
    """Keep non-dominated points for 2D minimization."""
    keep = []
    for i, p in enumerate(pts):
        if not any(i != j and q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1])
                   for j, q in enumerate(pts)):
            keep.append(p)
    return np.array(keep) if keep else pts[:1]


def _dominated_hv_2d(losses, ref=_HV_REF) -> float:
    """Dominated hypervolume of a 2D minimization point set vs reference.

    Prefers Optuna's tested helper (clipping points into the reference box,
    which it requires); falls back to a self-contained sweep.
    """
    ref_arr = np.asarray(ref, dtype=float)
    pts = np.clip(np.asarray(losses, dtype=float).reshape(-1, 2), 0.0, ref_arr)
    pts = _pareto_filter_2d(pts)
    if _OPTUNA_HV is not None:
        try:
            return float(_OPTUNA_HV(pts, ref_arr, assume_pareto=True))
        except TypeError:
            try:
                return float(_OPTUNA_HV(pts, ref_arr))
            except Exception:
                pass
        except Exception:
            pass
    # Self-contained fallback: sort by x ascending, sweep.
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    y_min = ref_arr[1]
    for i in range(len(pts)):
        y_min = min(y_min, pts[i, 1])
        x_next = pts[i + 1, 0] if i + 1 < len(pts) else ref_arr[0]
        dx = x_next - pts[i, 0]
        if dx > 0:
            hv += dx * (ref_arr[1] - y_min)
    return hv


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
    """Result of a 2D (accuracy, size) Pareto search."""
    pareto_points: list[ParetoPoint]
    all_trials: list[ParetoPoint]
    original_accuracy: float
    original_size: int
    original_latency_ms: float
    n_trials: int
    search_time_sec: float
    # Adaptive-budget telemetry (additive; safe for existing consumers).
    effective_dim: int = 0
    complexity_bits: float = 0.0
    n_trials_completed: int = 0
    stopped_early: bool = False

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
    """2D Pareto search engine for model compression with an adaptive budget."""

    def __init__(
        self,
        n_trials: int = 50,
        seed: int = 42,
        verbose: bool = False,
        allow_pruning: bool = False,
        *,
        adaptive: bool = True,
        trials_per_dim: int = 10,
        startup_trials: int = 10,
        min_trials: int = 30,
        max_trials: int | None = None,
        early_stop: bool = False,
        hv_patience: int = 20,
        hv_epsilon: float = 1e-3,
    ):
        """
        Args:
            n_trials: Backward-compatible trial count. When ``adaptive`` is False
                the search runs exactly this many trials. When ``adaptive`` is
                True it acts as the ceiling (``max_trials`` defaults to it).
            adaptive: Scale the trial count with the search dimensionality D.
            trials_per_dim: Trials added per live Optuna dimension.
            startup_trials: Constant added to the budget (covers TPE's random
                startup phase); also the floor below which early stop can't fire.
            min_trials: Lower clamp on the adaptive budget.
            max_trials: Upper clamp on the adaptive budget (defaults to
                ``n_trials``).
            early_stop: Enable the HV-plateau early stop. Default False — the
                frontier's late-jump behavior makes plateau stopping lossy; the
                D-scaled budget is the reliable adaptive mechanism.
            hv_patience / hv_epsilon: Early-stop window and relative-improvement
                threshold (only used when ``early_stop`` is True).
        """
        self.n_trials = n_trials
        self.seed = seed
        self.verbose = verbose
        self.allow_pruning = allow_pruning
        self.adaptive = adaptive
        self.trials_per_dim = trials_per_dim
        self.startup_trials = startup_trials
        self.min_trials = min_trials
        self.max_trials = max_trials if max_trials is not None else n_trials
        self.early_stop = early_stop
        self.hv_patience = hv_patience
        self.hv_epsilon = hv_epsilon
        self._transformer = OnnxTransformer()
        self._translator = ActionTranslator()

    @staticmethod
    def _effective_dimensionality(action_spaces, compressible, allow_pruning):
        """Count live Optuna search dimensions D (and the log2 search-space size
        ``complexity_bits``) from the already-built per-op action spaces. This is
        exactly the number of ``trial.suggest_*`` calls one objective evaluation
        makes, so it is the right per-model signal for sizing the trial budget."""
        D = 0
        bits = 0.0
        for op in compressible:
            sp = action_spaces[op.name]
            n_prec = len(sp.allowed_precisions)
            if n_prec > 1:
                D += 1
                bits += math.log2(n_prec)
            if sp.fuse_available and op.category in _FUSE_EFFECTIVE_CATEGORIES:
                D += 1
                bits += 1.0
            if allow_pruning:
                lo, hi = sp.prune_ratio_range
                if hi > 0:
                    levels = int(round((hi - lo) / 0.1)) + 1
                    D += 1
                    bits += math.log2(max(2, levels))
        return D, bits

    def _budget(self, D: int) -> int:
        """Adaptive trial count from the search dimensionality D.

        ``max_trials`` is the hard ceiling (applied last), so a caller passing a
        small ``n_trials`` always caps the budget even when it is below
        ``min_trials``."""
        if not self.adaptive:
            return self.n_trials
        raw = round(self.trials_per_dim * D + self.startup_trials)
        return int(min(self.max_trials, max(self.min_trials, raw)))

    def search(
        self,
        model: onnx.ModelProto,
        graph_info: GraphInfo,
        sensitivity: SensitivityProfile,
        eval_fn: Callable[[onnx.ModelProto], float],
        calibration_input: dict[str, np.ndarray] | None = None,
    ) -> ParetoResult:
        """Run 2D (accuracy, size) Pareto search.

        Args:
            model: Original ONNX model.
            graph_info: Analyzed graph info.
            sensitivity: UOSA sensitivity profile.
            eval_fn: Function that takes an ONNX model and returns accuracy (0-1).
            calibration_input: Sample input for latency measurement (reporting).
        """
        compressible = graph_info.compressible_operators
        if not compressible:
            raise ValueError("No compressible operators found")

        # Baseline measurements
        original_acc = eval_fn(model)
        original_size = model.ByteSize()
        original_weights = max(1, initializer_bytes(model))
        original_latency = self._measure_latency(model, calibration_input)

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

        # Per-model adaptive budget from the live search dimensionality.
        eff_dim, complexity_bits = self._effective_dimensionality(
            action_spaces, compressible, self.allow_pruning)
        n_trials = self._budget(eff_dim)
        # The early stop must not fire during TPE's random startup phase.
        floor = max(self.startup_trials, min(self.min_trials, n_trials))

        if self.verbose:
            print(f"  Baseline: acc={original_acc:.4f}, size={original_size}B, "
                  f"lat={original_latency:.2f}ms")
            print(f"  Search dims D={eff_dim} (~{complexity_bits:.1f} bits) -> "
                  f"budget={n_trials} trials"
                  f"{' +HV-early-stop' if self.early_stop else ''}")

        all_trials: list[ParetoPoint] = []

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # 2D deterministic objective: maximize accuracy, minimize size. Latency
        # is deliberately excluded from the objective (see module docstring).
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        start_time = time.time()

        def objective(trial: optuna.Trial) -> tuple[float, float]:
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
                return original_acc, float(original_size)

            compressed = self._transformer.apply(model, actions)

            # Evaluate accuracy (deterministic). Size is exact bytes.
            try:
                acc = eval_fn(compressed)
            except Exception:
                return 0.0, float(original_size)

            size = float(compressed.ByteSize())
            weights = initializer_bytes(compressed)
            # Latency is measured for reporting only; it does not feed the search.
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
                print(f"    Trial {trial.number}: acc={acc:.4f}, "
                      f"size_r={size/original_size:.3f}, lat={latency:.2f}ms")

            return acc, size

        # Optional HV-plateau early stop on the clean 2D frontier. Disabled by
        # default; the budget is the reliable adaptive mechanism.
        callbacks = []
        stop_flag = {"early": False}
        if self.early_stop and self.adaptive:
            hv_history: list[float] = []

            def _early_stop(study: optuna.Study, trial: optuna.trial.FrozenTrial):
                trials_done = trial.number + 1
                if trials_done < floor:
                    return
                losses = []
                for t in study.best_trials:
                    a, s = t.values
                    losses.append((
                        1.0 - (a / original_acc if original_acc > 0 else 0.0),
                        s / original_size if original_size > 0 else 0.0,
                    ))
                if not losses:
                    return
                hv = _dominated_hv_2d(losses)
                hv_history.append(hv)
                if len(hv_history) > self.hv_patience:
                    past = hv_history[-self.hv_patience - 1]
                    rel = (hv - past) / past if past > 1e-12 else float("inf")
                    if rel < self.hv_epsilon:
                        stop_flag["early"] = True
                        study.stop()

            callbacks.append(_early_stop)

        study.optimize(objective, n_trials=n_trials, callbacks=callbacks,
                       show_progress_bar=False)

        search_time = time.time() - start_time

        # Extract Pareto frontier (2D non-dominated set over accuracy, size).
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
            n_trials=n_trials,
            search_time_sec=search_time,
            effective_dim=eff_dim,
            complexity_bits=complexity_bits,
            n_trials_completed=len(study.trials),
            stopped_early=stop_flag["early"],
        )

    def _measure_latency(
        self,
        model: onnx.ModelProto,
        sample_input: dict[str, np.ndarray] | None,
        n_warmup: int = 3,
        n_measure: int = 10,
    ) -> float:
        """Measure inference latency in milliseconds (reporting only)."""
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
        """Compute Pareto frontier: maximize accuracy, minimize size."""
        pareto = []
        for p in points:
            dominated = False
            for q in points:
                if (q.accuracy >= p.accuracy and
                    q.model_size_bytes <= p.model_size_bytes and
                    (q.accuracy > p.accuracy or
                     q.model_size_bytes < p.model_size_bytes)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)
        return pareto
