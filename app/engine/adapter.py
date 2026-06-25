"""The ONLY module that imports `peops` — runs the real compression pipeline.

Mirrors `PEOps.optimize()` (PEOps-PoC/peops/sdk.py) step-by-step but emits an
IngestionLog line per event (6 phases, real numbers) and returns artifacts
already mapped to the frontend contract shapes.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.services.mappers.architecture_mapper import map_architecture
from app.services.mappers.pareto_mapper import baseline_experiment, map_pareto
from app.services.surrogate import SurrogateModel


class PipelineCancelled(Exception):
    pass


@dataclass
class PipelineArtifacts:
    architecture: dict          # frontend Architecture JSON (response-ready)
    pareto: dict                # frontend ParetoExperiment JSON
    benchmark: dict             # real ORT benchmark (original vs compressed)
    best_accuracy: float        # % for ModelListItem.bestAccuracy
    quality_score: float        # DFCV Q ∈ [0,1]
    risk_level: str
    artifact_path: str | None   # exported compressed ONNX
    elapsed_sec: float
    max_sensitivity: float      # highest normalized UOSA sensitivity ∈ [0,1]
    ingested_path: str | None = None        # post-ingestion ONNX (per-trial export source)
    trial_configs: dict | None = None       # trial_number -> compression_config dict
    guarantee_rung: str | None = None       # selected ladder rung (certified floor)
    guarantee_ofs: float | None = None      # pooled-gate OFS of the SERVED artifact
    guarantee_certificate: str | None = None
    artifact_meta: dict | None = None        # served-artifact provenance (source/trial/rung)


Emit = Callable[[str, str], None]          # (level, message)
Progress = Callable[[int], None]           # 0..100


def run_pipeline(
    *,
    model_id: str,
    model_name: str,
    file_name: str,
    source_path: str,
    input_shape: list[int] | None,
    declared_format: str,
    run_id: str,
    emit: Emit,
    progress: Progress,
    n_trials: int,
    n_probes: int,
    seed: int,
    max_compressible_ops: int,
    storage_dir: str,
    should_cancel: Callable[[], bool],
    benchmark_samples: int = 200,
    guarantee_mode: bool = True,
    tau: float = 0.95,
    adaptive: bool = True,
    trials_per_dim: int = 10,
    startup_trials: int = 10,
    min_trials: int = 30,
    early_stop: bool = False,
    hv_patience: int = 20,
    hv_epsilon: float = 1e-3,
) -> PipelineArtifacts:
    import onnx  # noqa: F401 — fail fast if the engine extra is missing
    from peops.core.calibration_generator import CalibrationGenerator
    from peops.core.compression_actions import (
        ActionTranslator,
        get_action_space,
    )
    from peops.core.guarantee import build_gate_probes, gate_check, guarantee_compress
    from peops.core.ingestion import ingest
    from peops.core.uosa import compute_uosa
    from peops.core.validation import CompressionValidator
    from peops.graph.model_detector import ModelDetector
    from peops.graph.onnx_analyzer import OnnxAnalyzer, initializer_bytes
    from peops.graph.onnx_transformer import OnnxTransformer
    from peops.sdk import _reconstruct_model
    from peops.search.pareto_search import ParetoSearch

    t0 = time.time()
    phase_timings: list[dict] = []
    _phase_started = [t0]

    def check_cancel() -> None:
        if should_cancel():
            raise PipelineCancelled(run_id)

    def _close_phase() -> None:
        if phase_timings:
            phase_timings[-1]["sec"] = round(time.time() - _phase_started[0], 3)
        _phase_started[0] = time.time()

    def phase(n: int, label: str) -> None:
        _close_phase()
        phase_timings.append({"name": label.split("·")[0].strip(), "sec": 0.0})
        emit("INFO", f"═══ Phase {n}/6 · {label} ═══")

    # ── Phase 1 · Universal Ingestion ───────────────────────────────────────
    phase(1, "Universal Ingestion")
    src = Path(source_path)
    size_mb = src.stat().st_size / 1e6
    emit("INFO", f"Reading {file_name} ({size_mb:.2f} MB)")
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    emit("DEBUG", "Computing artifact checksum (sha256)…")
    emit("INFO", f"sha256:{digest[:16]}…{digest[-3:]}")
    emit("INFO", "Detecting model format from file headers")
    ingestion = ingest(source_path, input_shape=input_shape)
    emit("INFO", f"Format detected: {ingestion.detected_format.value} (declared {declared_format})")
    emit("INFO", "Converting to ONNX intermediate representation"
         if ingestion.detected_format.value != "onnx"
         else "Already ONNX — graph parsed")
    model = ingestion.onnx_model
    emit("INFO", f"Loaded {len(model.graph.node)} ops, {len(model.graph.initializer)} weight tensors")
    sig_in = ", ".join(f"{k}{v}" for k, v in ingestion.input_spec.items())
    emit("INFO", f"Inputs: {sig_in}")
    emit("INFO", f"Outputs: {', '.join(o.name for o in model.graph.output)}")
    # Persist the post-ingestion ONNX: per-trial Pareto export re-applies any
    # trial's action config to exactly this graph later.
    ingested_dir = Path(storage_dir)
    ingested_dir.mkdir(parents=True, exist_ok=True)
    ingested_path = ingested_dir / f"{model_id}_ingested.onnx"
    onnx.save(model, str(ingested_path))
    emit("INFO", f"Model artifact ingested into peops-registry (run {run_id})")
    progress(12)
    check_cancel()

    # ── Phase 2 · Architecture Analyzer · UOSA sensitivity ────────────────
    phase(2, "Architecture Analyzer · UOSA Sensitivity Profiling")
    emit("INFO", "Phase 1 complete — starting peops-analyzer")
    analyzer = OnnxAnalyzer()
    graph_info = analyzer.analyze(model)
    emit("INFO", f"Graph analysis: {len(graph_info.operators)} operators, "
                 f"{graph_info.total_params:,} params, {graph_info.total_flops:,} FLOPs")

    detector = ModelDetector()
    report = detector.detect(model)
    emit("INFO", f"Detected architecture: {report.architecture.value} "
                 f"(confidence {report.confidence:.0%})")
    emit("INFO", f"Pattern: {report.architecture_pattern}")
    if report.architecture.value == "Transformer":
        # Be explicit that a Transformer/LLM's certificate is NOT a task-performance
        # guarantee: probes are float tensors (not token IDs), fidelity is output
        # similarity (not perplexity/accuracy), and no LLM-specific compression runs.
        emit("WARN", "Transformer detected — calibration probes are float tensors, "
                     "not token IDs; the fidelity guarantee is output similarity on "
                     "synthetic probes, NOT task accuracy/perplexity, and no "
                     "LLM-specific compression (e.g. GPTQ/AWQ) is applied.")

    compressible = graph_info.compressible_operators
    if len(compressible) > max_compressible_ops:
        ranked_ops = sorted(
            compressible, key=lambda o: (o.flops_estimate, o.param_count), reverse=True,
        )
        for op in ranked_ops[max_compressible_ops:]:
            op.is_compressible = False
        emit("WARN", f"{len(compressible)} compressible ops > cap {max_compressible_ops} — "
                     f"protecting {len(compressible) - max_compressible_ops} low-FLOPs ops at FP32")
        compressible = graph_info.compressible_operators

    input_spec = {n: [1 if d <= 0 else d for d in s] for n, s in ingestion.input_spec.items()}
    gen = CalibrationGenerator(n_probes=n_probes, seed=seed)
    cal_info = gen.generate(model, input_spec, report.architecture)
    emit("INFO", f"Generated {len(cal_info.probes)} weight-derived synthetic calibration probes "
                 f"(no labeled data required)")
    progress(25)
    check_cancel()

    emit("INFO", f"Running UOSA perturbation analysis over {len(compressible)} operators")
    profile = compute_uosa(model, cal_info.probes, graph_info)
    for r in profile.ranked[:5]:
        emit("INFO", f"  S({r.operator_name}) = {r.sensitivity:.6f} [{r.op_type}]")
    if profile.ranked:
        top = profile.ranked[0]
        emit("WARN", f"Operator {top.operator_name} shows highest sensitivity "
                     f"(S={top.sensitivity:.6f}) — aggressive compression restricted")
    normalized = profile.normalized_scores()
    emit("INFO", "Sensitivity-guided action spaces constructed (UOSA narrows the search space)")
    progress(45)
    check_cancel()

    # ── Phase 3 · Sensitivity-Guided Pareto Search ─────────────────────────
    phase(3, "Sensitivity-Guided Pareto Search (Optuna)")
    validator = CompressionValidator(n_probes=min(16, max(2, len(cal_info.probes))), seed=seed)

    search_state = {"n": 0, "best": 0.0}

    def dfcv_eval(compressed_model) -> float:
        # Called once per Optuna trial — the hook to (a) abort a long search
        # mid-phase (peops' search loop has no cancellation parameter) and
        # (b) stream live per-trial progress into the ingestion log. The emit is
        # paced by the search itself (one ORT validation per call), so it never
        # outruns the work it reports.
        if should_cancel():
            raise PipelineCancelled(run_id)
        try:
            score = validator.validate(model, compressed_model, input_spec).quality_score
        except Exception:
            score = 0.0
        search_state["n"] += 1
        n = search_state["n"]
        search_state["best"] = max(search_state["best"], score)
        emit("INFO", f"  trial {n}/{n_trials} · Q={score:.4f} · best Q={search_state['best']:.4f}")
        # Fill the 45→75 progress band reserved for Phase 3 as trials complete.
        progress(min(75, 45 + (30 * n) // max(1, n_trials)))
        return score

    pareto_result = None
    if compressible:
        emit("INFO", f"Initializing Optuna study — deterministic 2D objective "
                     f"(accuracy ↑ / size ↓; latency measured for reporting), "
                     f"TPESampler(seed={seed})")
        budget_desc = (f"adaptive (≈{trials_per_dim}·D + {startup_trials}, "
                       f"{min_trials}–{n_trials})" if adaptive else f"fixed {n_trials}")
        emit("INFO", f"Budget: {budget_desc} trials · eval = DFCV quality score (data-free)")
        search = ParetoSearch(
            n_trials=n_trials, seed=seed, verbose=False,
            adaptive=adaptive, trials_per_dim=trials_per_dim,
            startup_trials=startup_trials, min_trials=min_trials,
            max_trials=n_trials, early_stop=early_stop,
            hv_patience=hv_patience, hv_epsilon=hv_epsilon,
        )
        try:
            pareto_result = search.search(
                model, graph_info, profile, dfcv_eval, cal_info.probes[0] if cal_info.probes else None,
            )
        except ValueError as exc:
            emit("WARN", f"Pareto search skipped: {exc}")
    else:
        emit("WARN", "No compressible operators — Pareto search skipped, FP32 baseline kept")
    progress(75)
    check_cancel()

    surrogate_metrics = None
    if pareto_result is not None:
        emit("INFO", f"Search dimensionality D={pareto_result.effective_dim} "
                     f"(~{pareto_result.complexity_bits:.1f} bits) → "
                     f"{pareto_result.n_trials_completed} trials run"
                     f"{' (early-stopped)' if pareto_result.stopped_early else ''}")
        emit("INFO", f"Baseline: Q={pareto_result.original_accuracy:.4f}, "
                     f"size={pareto_result.original_size / 1e6:.3f}MB, "
                     f"latency={pareto_result.original_latency_ms:.3f}ms")
        for p in pareto_result.all_trials[:: max(1, len(pareto_result.all_trials) // 6)]:
            emit("DEBUG", f"  trial #{p.trial_number}: Q={p.accuracy:.4f}, "
                          f"size_ratio={p.size_ratio:.3f}, lat={p.latency_ms:.3f}ms")
        emit("INFO", f"Search finished in {pareto_result.search_time_sec:.1f}s — "
                     f"frontier: {pareto_result.n_pareto} non-dominated / "
                     f"{len(pareto_result.all_trials)} trials")

        # Surrogate (backend implementation of the PoC's empty surrogate stub)
        surrogate = SurrogateModel(op_order=[op.name for op in compressible], seed=seed)
        surrogate_metrics = surrogate.fit(pareto_result.all_trials)
        if surrogate_metrics:
            emit("INFO", f"Surrogate trained on {surrogate_metrics.n_train} measurements — "
                         f"accuracy MAE {surrogate_metrics.acc_mae:.4f}, "
                         f"latency MAE {surrogate_metrics.latency_mae_ms:.3f}ms, "
                         f"R² {surrogate_metrics.r2:.2f}")
            repaired = surrogate.backfill_latencies(
                pareto_result.all_trials, pareto_result.original_latency_ms,
            )
            if repaired:
                emit("INFO", f"Surrogate backfilled {repaired} failed latency measurements")
        else:
            emit("DEBUG", "Surrogate skipped — not enough distinct trials")

    # ── Phase 4 · Optimizer · Guarantee Ladder & Pareto Selection ──────────
    phase(4, "Optimizer · Guarantee Ladder & Pareto Selection")
    translator = ActionTranslator()
    transformer = OnnxTransformer()
    compressed = None
    selected_precisions: dict[str, str] = {}
    guarantee_rung: str | None = None
    guarantee_ofs: float | None = None
    guarantee_certificate: str | None = None

    # Certified floor: the guarantee ladder (the configuration validated in
    # the UOSA paper experiments — pooled-probe OFS >= tau gate + original
    # fallback). The served artifact is its candidate unless a Pareto pick
    # passes the SAME gate and is strictly smaller.
    gate_probes = None
    ladder_result = None
    if guarantee_mode and compressible:
        emit("INFO", f"Running guarantee ladder (tau={tau}, pooled gate probes)")
        gate_probes = build_gate_probes(
            model, input_spec, report.architecture, n_probes=n_probes, seed=seed)
        try:
            ladder_result = guarantee_compress(
                model, tau=tau, n_probes=n_probes, seed=seed,
                graph_info=graph_info, profile=profile,
                architecture=report.architecture, input_spec=input_spec,
                ladder="v2", gate_probes=gate_probes,
            )
            emit("INFO", f"Ladder candidate: rung {ladder_result.rung} — "
                         f"OFS={ladder_result.output_fidelity:.4f}, "
                         f"size×{ladder_result.size_ratio:.3f}")
        except Exception as exc:  # ladder is best-effort; Pareto path remains
            emit("WARN", f"Guarantee ladder failed: {type(exc).__name__}: {exc}")
    check_cancel()

    chosen = None
    if pareto_result and pareto_result.pareto_points:
        candidates = sorted(pareto_result.pareto_points, key=lambda p: -p.accuracy)
        chosen = next(
            (c for c in candidates
             if c.accuracy >= pareto_result.original_accuracy * 0.95),
            candidates[0],
        )
        emit("INFO", f"Best Pareto point: trial #{chosen.trial_number} — "
                     f"Q={chosen.accuracy:.4f}, size×{chosen.size_ratio:.3f}, "
                     f"speedup×{chosen.speedup:.2f}")
        compressed = _reconstruct_model(
            model, graph_info, chosen.compression_config, translator, transformer,
        )

    if ladder_result is not None:
        ladder_bytes = ladder_result.model.ByteSize()
        pareto_ok = False
        if compressed is not None and gate_probes is not None:
            ofs_p, _q_p = gate_check(
                model, compressed, gate_probes, input_spec, report.architecture, seed)
            pareto_ok = ofs_p >= tau and compressed.ByteSize() < ladder_bytes
            emit("INFO", f"Pareto candidate gate check: OFS={ofs_p:.4f} "
                         f"(tau={tau}) · {compressed.ByteSize() / 1e6:.3f} MB "
                         f"vs ladder {ladder_bytes / 1e6:.3f} MB")
            if pareto_ok:
                guarantee_rung = "PARETO_CERTIFIED"
                guarantee_ofs = ofs_p
                emit("INFO", "Serving Pareto candidate — passes the guarantee gate "
                             "and beats the ladder on size")
        if not pareto_ok:
            compressed = ladder_result.model
            chosen = None
            guarantee_rung = ladder_result.rung
            guarantee_ofs = ladder_result.output_fidelity
            emit("INFO", f"Serving ladder candidate (rung {ladder_result.rung}) — "
                         "certified fidelity floor")
        guarantee_certificate = ladder_result.certificate()
        for line in guarantee_certificate.splitlines():
            emit("INFO", line)

    if chosen is not None:
        selected_precisions = {
            name: cfg.get("precision", "FP32")
            for name, cfg in chosen.compression_config.items()
        }
    elif guarantee_rung and guarantee_rung not in ("ORIGINAL",):
        rung_precision = {
            "INT8_uniform": "INT8", "INT8_uosa_mixed": "INT8",
            "W8_weight_only": "INT8", "FP16": "FP16",
        }.get(guarantee_rung, "FP32")
        protected_set = profile.get_protection_set(top_p=0.3) if profile.results else set()
        for op in compressible:
            if guarantee_rung == "INT8_uosa_mixed" and op.name in protected_set:
                selected_precisions[op.name] = "FP32"
            else:
                selected_precisions[op.name] = rung_precision

    if compressed is None:
        emit("INFO", "Fallback: single UOSA-guided compression pass")
        from peops.core.compression_actions import CompressionConfig

        actions = []
        for op in compressible:
            s = normalized.get(op.name, 0)
            space = get_action_space(op, sensitivity=s, sensitivity_threshold=0.3)
            best = space.allowed_precisions[-1]
            selected_precisions[op.name] = best.name if best.name != "INT4" else "FP16"
            cfg = CompressionConfig(precision_level=best)
            if not cfg.is_no_compression():
                actions.extend(translator.translate(op, cfg))
        compressed = transformer.apply(model, actions) if actions else model

    # Provenance of the SERVED artifact: which candidate `model.artifact_key`
    # actually holds, so the SDK Hub can label it exactly (and the Pareto Studio
    # can mark that trial). `chosen` is a Pareto point only when the certified
    # Pareto pick won; a ladder/fallback candidate is NOT in the trial list.
    if chosen is not None:
        artifact_meta: dict = {
            "source": "pareto",
            "trialNumber": chosen.trial_number,
            "rung": guarantee_rung,  # "PARETO_CERTIFIED" when it cleared the gate
            "accuracy": round(chosen.accuracy * 100, 2),
            "sizeRatio": round(chosen.size_ratio, 4),
            "sizeBytes": int(compressed.ByteSize()),
        }
    elif guarantee_rung:
        artifact_meta = {
            "source": "ladder",
            "rung": guarantee_rung,
            "ofs": round(guarantee_ofs, 4) if guarantee_ofs is not None else None,
            "sizeBytes": int(compressed.ByteSize()),
        }
    else:
        artifact_meta = {"source": "fallback", "sizeBytes": int(compressed.ByteSize())}

    validation = CompressionValidator(n_probes=n_probes, seed=seed).validate(
        model, compressed, input_spec, report.architecture,
    )
    emit("INFO", f"DFCV validation: Q={validation.quality_score:.4f} "
                 f"(OFS={validation.output_fidelity:.4f}, WFS={validation.weight_fidelity:.4f}, "
                 f"SIS={validation.structural_integrity:.4f}) → {validation.risk_level}")

    out_dir = Path(storage_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{model_id}_compressed.onnx"
    onnx.save(compressed, str(artifact_path))
    emit("INFO", f"Compiled artifact · {artifact_path.name} "
                 f"({compressed.ByteSize() / 1e6:.3f} MB, was {model.ByteSize() / 1e6:.3f} MB)")
    emit("INFO", f"Weights-only: {initializer_bytes(compressed) / 1e6:.3f} MB "
                 f"(was {initializer_bytes(model) / 1e6:.3f} MB)")
    progress(88)
    check_cancel()

    # ── Phase 5 · MLOps registration ───────────────────────────────────────
    phase(5, "MLOps · Artifact Registry")
    emit("INFO", f"Registering artifact → storage/{artifact_path.name}")
    emit("INFO", f"Run manifest {run_id} recorded (model {model_id})")
    emit("INFO", "Pareto frontier + sensitivity profile persisted to result cache")

    # ── Phase 6 · Real benchmark (original vs compressed, actual ORT runs) ──
    phase(6, "Benchmark · Real Inference Measurements")
    sample = cal_info.probes[0] if cal_info.probes else None
    emit("INFO", f"Benchmarking original vs compressed — {benchmark_samples} real "
                 f"onnxruntime inferences each (warmup 5)")
    bench_original = _benchmark_model(model, sample, benchmark_samples)
    check_cancel()
    bench_compressed = _benchmark_model(compressed, sample, benchmark_samples)
    benchmark = _build_benchmark_payload(
        bench_original, bench_compressed,
        agreement_pct=round(validation.output_fidelity * 100, 2),
        phases=phase_timings,
    )
    if benchmark:
        o, c = benchmark["original"], benchmark["compressed"]
        emit("INFO", f"Original   · p50 {o['p50']:.2f}ms · p95 {o['p95']:.2f}ms · "
                     f"{o['throughputPerMin']:,.0f} inf/min")
        emit("INFO", f"Compressed · p50 {c['p50']:.2f}ms · p95 {c['p95']:.2f}ms · "
                     f"{c['throughputPerMin']:,.0f} inf/min")
        emit("INFO", f"Output agreement vs original (DFCV OFS): {benchmark['agreementPct']:.2f}%")
    else:
        emit("WARN", "Benchmark skipped — no calibration probe available for inference")

    # ── Map results to the frontend contract ──────────────────────────────
    def recommend_for(op, sens: float) -> str:
        space = get_action_space(op, sensitivity=sens, sensitivity_threshold=0.3)
        allowed = [p for p in space.allowed_precisions if p.name in ("FP32", "FP16", "INT8")]
        if not allowed:
            return "FP32"
        return max(allowed, key=lambda p: int(p)).name

    total_latency = pareto_result.original_latency_ms if pareto_result else 0.0
    if total_latency <= 0:
        total_latency = _measure_latency_ms(model, cal_info.probes[0] if cal_info.probes else None)

    architecture = map_architecture(
        model_id, graph_info, normalized,
        architecture_name=report.architecture.value,
        total_latency_ms=total_latency,
        recommend_for=recommend_for,
        selected_precisions=selected_precisions or None,
    )

    if pareto_result is not None and pareto_result.all_trials:
        experiment = map_pareto(
            model_id, model_name, f"exp_{run_id}", pareto_result, status="completed",
            served_trial_number=chosen.trial_number if chosen is not None else None,
        )
    else:
        experiment = baseline_experiment(
            model_id, model_name, f"exp_{run_id}",
            size_mb=model.ByteSize() / 1e6, latency_ms=total_latency,
            quality=validation.quality_score,
        )

    best_pct = max((t.accuracy for t in experiment.trials), default=validation.quality_score * 100)
    _close_phase()
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    emit("INFO", f"All six phases complete · PEOps pipeline finished in {mins}m {secs}s")
    emit("INFO", "Sensitivity analysis ready")
    progress(100)

    trial_configs = None
    if pareto_result is not None and pareto_result.all_trials:
        trial_configs = {
            str(p.trial_number): p.compression_config for p in pareto_result.all_trials
        }

    return PipelineArtifacts(
        architecture=architecture.to_response(),
        pareto=experiment.model_dump(),
        benchmark=benchmark,
        best_accuracy=round(best_pct, 1),
        quality_score=validation.quality_score,
        risk_level=validation.risk_level,
        artifact_path=str(artifact_path),
        elapsed_sec=elapsed,
        max_sensitivity=max(normalized.values(), default=0.0),
        ingested_path=str(ingested_path),
        trial_configs=trial_configs,
        guarantee_rung=guarantee_rung,
        guarantee_ofs=guarantee_ofs,
        guarantee_certificate=guarantee_certificate,
        artifact_meta=artifact_meta,
    )


def _benchmark_model(model, sample_input, n: int) -> list[tuple[str, float]] | None:
    """Run `n` REAL onnxruntime inferences; return (iso_timestamp, latency_ms) pairs."""
    if sample_input is None:
        return None
    try:
        import time as _t
        from datetime import datetime, timezone

        import onnxruntime as ort

        session = ort.InferenceSession(model.SerializeToString())
        outs = [o.name for o in session.get_outputs()]
        for _ in range(5):  # warmup
            session.run(outs, sample_input)
        samples: list[tuple[str, float]] = []
        for _ in range(n):
            t0 = _t.perf_counter()
            session.run(outs, sample_input)
            ms = (_t.perf_counter() - t0) * 1000
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            samples.append((ts, ms))
        return samples
    except Exception:
        return None


def _percentile(values: list[float], q: float) -> float:
    import numpy as np

    return float(np.percentile(np.asarray(values), q))


def _summarize(samples: list[tuple[str, float]]) -> dict:
    lats = [ms for _, ms in samples]
    mean_ms = sum(lats) / len(lats)
    return {
        "p50": round(_percentile(lats, 50), 3),
        "p95": round(_percentile(lats, 95), 3),
        "p99": round(_percentile(lats, 99), 3),
        "throughputPerMin": round(60_000.0 / mean_ms, 1) if mean_ms > 0 else 0.0,
    }


def _build_benchmark_payload(
    bench_original, bench_compressed, *, agreement_pct: float, phases: list[dict],
) -> dict:
    """Assemble the result_cache `benchmark` JSON — every number is measured."""
    from datetime import datetime, timezone

    if not bench_original or not bench_compressed:
        return {}
    n_buckets = min(24, len(bench_compressed))
    size = max(1, len(bench_compressed) // n_buckets)
    buckets = []
    for i in range(0, len(bench_compressed), size):
        chunk = bench_compressed[i:i + size]
        lats = [ms for _, ms in chunk]
        window_min = sum(lats) / 60_000.0  # actual wall time spent inferring
        buckets.append({
            "t": chunk[0][0],
            # real inferences completed in this bucket, scaled to per-minute rate
            "requests": round(len(chunk) / window_min, 1) if window_min > 0 else 0.0,
            "p95": round(_percentile(lats, 95), 3),
            "p50": round(_percentile(lats, 50), 3),
            "p99": round(_percentile(lats, 99), 3),
        })
    return {
        "measuredAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "sampleCount": len(bench_compressed),
        "original": _summarize(bench_original),
        "compressed": _summarize(bench_compressed),
        "agreementPct": agreement_pct,
        "buckets": buckets,
        "phases": phases,
    }


def _measure_latency_ms(model, sample_input) -> float:
    if sample_input is None:
        return 0.5
    try:
        import time as _t

        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(model.SerializeToString())
        outs = [o.name for o in session.get_outputs()]
        for _ in range(3):
            session.run(outs, sample_input)
        times = []
        for _ in range(10):
            t0 = _t.perf_counter()
            session.run(outs, sample_input)
            times.append((_t.perf_counter() - t0) * 1000)
        return float(np.median(times)) or 0.5
    except Exception:
        return 0.5
