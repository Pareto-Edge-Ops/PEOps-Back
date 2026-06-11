"""peops ParetoResult → frontend ParetoExperiment (duck-typed inputs)."""

from __future__ import annotations

from collections import Counter

from app.schemas.pareto import ParetoBudget, ParetoExperiment, Trial


def derive_quant_label(config: dict[str, dict]) -> str:
    """Summarize a per-op compression config, e.g. "INT8-mix r0.2"."""
    if not config:
        return "FP32 (baseline)"
    precisions = Counter(str(cfg.get("precision", "FP32")) for cfg in config.values())
    if set(precisions) == {"FP32"}:
        return "FP32 (baseline)"
    dominant, _ = precisions.most_common(1)[0]
    mixed = len(precisions) > 1
    prunes = [float(cfg.get("prune_ratio", 0.0)) for cfg in config.values()]
    avg_prune = sum(prunes) / len(prunes) if prunes else 0.0
    label = f"{dominant}{'-mix' if mixed else ''}"
    if avg_prune > 0.01:
        label += f" r{avg_prune:.1f}"
    if any(cfg.get("fuse") for cfg in config.values()):
        label += " +fuse"
    return label


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def trial_score(
    accuracy: float, base_accuracy: float, size_ratio: float, speedup: float,
) -> float:
    """Composite 0..100: 0.5·retention + 0.3·size-saving + 0.2·speedup."""
    acc_term = _clamp(accuracy / base_accuracy if base_accuracy > 0 else 0, 0, 1)
    size_term = _clamp(1 - size_ratio, 0, 1)
    speed_term = _clamp(1 - 1 / max(speedup, 1e-6), 0, 1)
    return round(100 * (0.5 * acc_term + 0.3 * size_term + 0.2 * speed_term), 1)


def map_pareto(
    model_id: str,
    model_name: str,
    experiment_id: str,
    pareto,  # peops ParetoResult (duck-typed)
    *,
    status: str = "completed",
) -> ParetoExperiment:
    base_acc = pareto.original_accuracy or 1e-9
    frontier_nums = {p.trial_number for p in pareto.pareto_points}
    short = model_name.lower().replace(" ", "-")[:24]

    trials: list[Trial] = []
    for p in pareto.all_trials:
        trials.append(Trial(
            id=f"t_{p.trial_number}",
            name=f"{short} · #{p.trial_number + 1:03d}",
            accuracy=round(p.accuracy * 100, 2),
            latency=round(p.latency_ms, 3),
            size=round(p.model_size_bytes / 1e6, 3),
            score=trial_score(p.accuracy, base_acc, p.size_ratio, p.speedup),
            quant=derive_quant_label(p.compression_config),
            onFrontier=p.trial_number in frontier_nums,
            trialNumber=p.trial_number,
        ))

    original_size_mb = pareto.original_size / 1e6
    return ParetoExperiment(
        modelId=model_id,
        modelName=model_name,
        experimentId=experiment_id,
        status=status,  # type: ignore[arg-type]
        iterCurrent=len(pareto.all_trials),
        iterTotal=pareto.n_trials,
        budget=ParetoBudget(
            maxLatency=round(max(pareto.original_latency_ms, 0.5), 2),
            maxAccuracyDrop=5.0,
            maxSize=round(max(original_size_mb, 0.01), 3),
        ),
        baseAccuracy=round(base_acc * 100, 2),
        trials=trials,
    )


def baseline_experiment(
    model_id: str, model_name: str, experiment_id: str,
    *, size_mb: float, latency_ms: float, quality: float,
) -> ParetoExperiment:
    """Single FP32 trial when no operator was compressible — keeps /pareto
    non-empty and schema-valid (ParetoSearch raises on zero compressible ops)."""
    return ParetoExperiment(
        modelId=model_id,
        modelName=model_name,
        experimentId=experiment_id,
        status="completed",
        iterCurrent=1,
        iterTotal=1,
        budget=ParetoBudget(
            maxLatency=round(max(latency_ms, 0.5), 2),
            maxAccuracyDrop=5.0,
            maxSize=round(max(size_mb, 0.01), 3),
        ),
        baseAccuracy=round(quality * 100, 2),
        trials=[Trial(
            id="t_0", name=f"{model_name.lower().replace(' ', '-')[:24]} · #001",
            accuracy=round(quality * 100, 2), latency=round(latency_ms, 3),
            size=round(size_mb, 3), score=50.0, quant="FP32 (baseline)",
            onFrontier=True,
        )],
    )
