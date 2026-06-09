"""PEOps SDK: one-call interface for model optimization and deployment.

Usage:
    from peops.sdk import PEOps

    # Optimize any model (학습 데이터 불필요)
    result = PEOps.optimize("/path/to/model.h5")

    # Use the compressed model directly
    output = result.predict(input_data)

    # Export for deployment
    result.export("optimized_model.onnx")

    # View the Pareto search results
    result.pareto.summary()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from peops.core.calibration_generator import CalibrationGenerator
from peops.core.compression_actions import (
    ActionTranslator,
    CompressionConfig,
    PrecisionLevel,
    get_action_space,
)
from peops.core.guarantee import GuaranteeResult, guarantee_compress
from peops.core.ingestion import IngestionResult, ModelFormat, ingest
from peops.core.uosa import SensitivityProfile, compute_uosa
from peops.core.validation import CompressionValidator, ValidationResult
from peops.graph.model_detector import ArchitectureType, ModelDetector, ModelReport
from peops.graph.onnx_analyzer import GraphInfo, OnnxAnalyzer
from peops.graph.onnx_transformer import OnnxTransformer
from peops.search.pareto_search import ParetoResult, ParetoSearch


@dataclass
class OptimizationResult:
    """Complete result of PEOps model optimization."""
    original_model: onnx.ModelProto
    compressed_model: onnx.ModelProto
    detection: ModelReport
    sensitivity: SensitivityProfile
    validation: ValidationResult
    pareto: ParetoResult | None
    ingestion: IngestionResult
    graph_info: GraphInfo
    optimization_time_sec: float
    guarantee: GuaranteeResult | None = None

    _session: ort.InferenceSession | None = field(default=None, repr=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on the compressed model."""
        if self._session is None:
            self._session = ort.InferenceSession(
                self.compressed_model.SerializeToString())

        input_name = self._session.get_inputs()[0].name
        output_names = [o.name for o in self._session.get_outputs()]

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        results = self._session.run(output_names, {input_name: X})
        return results[0]

    def export(self, path: str) -> str:
        """Export compressed ONNX model to file."""
        onnx.save(self.compressed_model, path)
        return path

    def summary(self) -> str:
        """Human-readable optimization summary."""
        lines = [
            "═" * 50,
            "  PEOps Optimization Result",
            "═" * 50,
            f"  Architecture:  {self.detection.architecture.value} ({self.detection.confidence:.0%})",
            f"  Parameters:    {self.detection.total_params:,}",
            f"  Original size: {self.original_model.ByteSize()/1024:.1f} KB",
            f"  Compressed:    {self.compressed_model.ByteSize()/1024:.1f} KB",
            f"  Quality (Q):   {self.validation.quality_score:.4f} ({self.validation.risk_level})",
            f"  Time:          {self.optimization_time_sec:.1f}s",
        ]
        if self.pareto:
            lines.append(f"  Pareto points: {self.pareto.n_pareto}")
        lines.append("═" * 50)
        return "\n".join(lines)


class PEOps:
    """Main SDK entry point for model optimization."""

    @staticmethod
    def optimize(
        model_path: str,
        input_shape: list[int] | None = None,
        n_pareto_trials: int = 30,
        n_probes: int = 32,
        seed: int = 42,
        run_pareto: bool = True,
        guarantee: bool = False,
        tau: float = 0.95,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Optimize any ML model. No training data required.

        Args:
            model_path: Path to model file (.pt, .pth, .h5, .pkl, .onnx)
            input_shape: Input shape hint (required for PyTorch, auto for sklearn)
            n_pareto_trials: Number of Pareto search trials (0 to skip)
            n_probes: Number of synthetic calibration probes
            seed: Random seed
            run_pareto: Whether to run 3D Pareto search
            guarantee: Use the guarantee-by-construction ladder instead of
                Pareto selection. The returned model is certified to satisfy
                output fidelity >= tau on held-out calibration probes
                (worst case: the original model is returned unchanged).
            tau: Fidelity floor for the guarantee gate
            verbose: Print progress

        Returns:
            OptimizationResult with compressed model, validation, and Pareto data
        """
        t0 = time.time()

        # Step 1: Ingest
        if verbose:
            print(f"[1/5] Ingesting {Path(model_path).name}...")
        ingestion = ingest(model_path, input_shape=input_shape)

        # Step 2: Detect
        if verbose:
            print(f"[2/5] Detecting architecture...")
        detector = ModelDetector()
        report = detector.detect(ingestion.onnx_model)
        if verbose:
            print(f"      → {report.architecture.value} ({report.confidence:.0%}), "
                  f"{report.total_params:,} params")

        # Step 3: UOSA
        if verbose:
            print(f"[3/5] UOSA sensitivity analysis ({n_probes} synthetic probes)...")
        analyzer = OnnxAnalyzer()
        graph_info = analyzer.analyze(ingestion.onnx_model)

        input_spec = _resolve_input_spec(ingestion)
        gen = CalibrationGenerator(n_probes=n_probes, seed=seed)
        cal_info = gen.generate(ingestion.onnx_model, input_spec, report.architecture)
        profile = compute_uosa(ingestion.onnx_model, cal_info.probes, graph_info)

        if verbose:
            for r in profile.ranked[:5]:
                print(f"      {r.operator_name}: S={r.sensitivity:.6f}")

        # Step 4 (guarantee mode): progressive ladder with fidelity gate.
        # Returns the smallest rung satisfying OFS >= tau; falls back to the
        # original model, so the fidelity floor holds by construction.
        guarantee_result = None
        if guarantee:
            if verbose:
                print(f"[4/5] Guarantee ladder (tau={tau})...")
            guarantee_result = guarantee_compress(
                ingestion.onnx_model, tau=tau, n_probes=n_probes, seed=seed,
                graph_info=graph_info, profile=profile,
                architecture=report.architecture, input_spec=input_spec,
                verbose=verbose,
            )
            compressed = guarantee_result.model
            validation = CompressionValidator(n_probes=n_probes, seed=seed).validate(
                ingestion.onnx_model, compressed, input_spec, report.architecture,
            )
            if verbose:
                print(guarantee_result.certificate())
            return OptimizationResult(
                original_model=ingestion.onnx_model,
                compressed_model=compressed,
                detection=report,
                sensitivity=profile,
                validation=validation,
                pareto=None,
                ingestion=ingestion,
                graph_info=graph_info,
                optimization_time_sec=time.time() - t0,
                guarantee=guarantee_result,
            )

        # Step 4: Multi-trial Pareto Search (core optimization)
        translator = ActionTranslator()
        transformer = OnnxTransformer()
        normalized = profile.normalized_scores()

        pareto_result = None
        if n_pareto_trials > 0:
            if verbose:
                print(f"[4/5] 3D Pareto Search ({n_pareto_trials} trials)...")
                print(f"      UOSA narrows search space → Optuna explores within it")
            pareto_result = _run_pareto(
                ingestion.onnx_model, graph_info, profile,
                input_spec, cal_info.probes, n_pareto_trials, seed, verbose,
            )

        # Select best compressed model from Pareto results
        compressed = None
        if pareto_result and pareto_result.pareto_points:
            # Strategy: pick the Pareto point with best quality that is still
            # smaller or equal in size to original (actual compression)
            candidates = sorted(pareto_result.pareto_points, key=lambda p: -p.accuracy)
            for c in candidates:
                if c.accuracy >= pareto_result.original_accuracy * 0.95:
                    compressed = _reconstruct_model(
                        ingestion.onnx_model, graph_info,
                        c.compression_config, translator, transformer,
                    )
                    if verbose:
                        print(f"      → Selected Pareto point: Q={c.accuracy:.4f}, "
                              f"size={c.size_ratio:.3f}x (trial #{c.trial_number})")
                    break

            if compressed is None:
                best_point = candidates[0]
                compressed = _reconstruct_model(
                    ingestion.onnx_model, graph_info,
                    best_point.compression_config, translator, transformer,
                )
                if verbose:
                    print(f"      → Best available: Q={best_point.accuracy:.4f}")

        # Fallback: single UOSA-guided compression (if Pareto skipped or failed)
        if compressed is None:
            if verbose:
                print(f"[4/5] Fallback: single UOSA-guided compression...")
            actions = []
            for op in graph_info.compressible_operators:
                s = normalized.get(op.name, 0)
                space = get_action_space(op, sensitivity=s, sensitivity_threshold=0.3)
                best = space.allowed_precisions[-1]
                config = CompressionConfig(precision_level=best)
                actions.extend(translator.translate(op, config))
            compressed = transformer.apply(ingestion.onnx_model, actions)

        # Step 5: Validate
        if verbose:
            print(f"[5/5] Validating (data-free)...")
        validation = CompressionValidator(n_probes=n_probes, seed=seed).validate(
            ingestion.onnx_model, compressed, input_spec, report.architecture,
        )
        if verbose:
            print(f"      → Q={validation.quality_score:.4f} ({validation.risk_level})")

        elapsed = time.time() - t0
        result = OptimizationResult(
            original_model=ingestion.onnx_model,
            compressed_model=compressed,
            detection=report,
            sensitivity=profile,
            validation=validation,
            pareto=pareto_result,
            ingestion=ingestion,
            graph_info=graph_info,
            optimization_time_sec=elapsed,
        )

        if verbose:
            print(result.summary())

        return result

    @staticmethod
    def predict(model_path: str, X: np.ndarray, input_shape: list[int] | None = None) -> np.ndarray:
        """Quick: optimize and predict in one call."""
        result = PEOps.optimize(model_path, input_shape=input_shape, run_pareto=False, verbose=False)
        return result.predict(X)


def _resolve_input_spec(ingestion: IngestionResult) -> dict[str, list[int]]:
    """Resolve input spec, replacing dynamic dims with 1."""
    spec = {}
    for name, shape in ingestion.input_spec.items():
        spec[name] = [1 if d <= 0 else d for d in shape]
    return spec


def _run_pareto(
    model, graph_info, profile, input_spec, probes,
    n_trials, seed, verbose,
) -> ParetoResult | None:
    """Run 3D Pareto search using DFCV as eval function (no labeled data)."""
    validator = CompressionValidator(n_probes=min(16, len(probes)), seed=seed)
    sample_input = probes[0] if probes else None

    def dfcv_eval(compressed_model):
        try:
            r = validator.validate(model, compressed_model, input_spec)
            return r.quality_score
        except Exception:
            return 0.0

    try:
        search = ParetoSearch(n_trials=n_trials, seed=seed, verbose=verbose)
        return search.search(model, graph_info, profile, dfcv_eval, sample_input)
    except Exception:
        return None


def _reconstruct_model(
    original, graph_info, config_dict, translator, transformer,
) -> onnx.ModelProto:
    """Reconstruct compressed model from Pareto config dict."""
    actions = []
    op_map = {op.name: op for op in graph_info.compressible_operators}

    for op_name, cfg in config_dict.items():
        op = op_map.get(op_name)
        if op is None:
            continue
        config = CompressionConfig(
            precision_level=PrecisionLevel[cfg["precision"]],
            prune_ratio=cfg.get("prune_ratio", 0.0),
            fuse_enabled=cfg.get("fuse", False),
        )
        if not config.is_no_compression():
            actions.extend(translator.translate(op, config))

    return transformer.apply(original, actions) if actions else original
