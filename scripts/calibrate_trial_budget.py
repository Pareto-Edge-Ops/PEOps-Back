#!/usr/bin/env python3
"""Calibrate the adaptive Optuna trial budget.

For each model, run ONE long FIXED Pareto sweep (``adaptive=False`` so the engine
runs the full ``--trials``), reconstruct the running 2D hypervolume HV(t) of the
deterministic (accuracy, size) frontier, and locate the knee trial counts
t99/t995/t999 (first trial reaching that fraction of the final HV). Record the
model's effective search dimensionality D. This is the evidence behind the budget
formula  n = clamp(per_dim·D + startup, min_trials, max_trials).

Because the search now optimizes only the two deterministic objectives, the
HV curve is reproducible at a fixed seed (unlike the old latency-in-the-objective
search). The optional early-stop simulation shows where an HV-plateau stop would
fire and what fraction of the final frontier it would capture.

Read-only w.r.t. the repo: writes CSV/JSON only to --out.

Usage:
    python scripts/calibrate_trial_budget.py --model squeezenet1.1-7.onnx --trials 300 --out /tmp/calib
    python scripts/calibrate_trial_budget.py --all --trials 300 --out /tmp/calib
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from peops.sdk import PEOps
from peops.search.pareto_search import (
    ParetoSearch,
    _dominated_hv_2d,           # reuse the engine's exact HV helper
    get_action_space,
)
from peops.graph.onnx_analyzer import OperatorCategory  # noqa: F401 (clarity)

KNEE_FRACS = {"t99": 0.99, "t995": 0.995, "t999": 0.999}
TEST_DIR = Path("/Users/kwonminjae/Desktop/PEOps/test-models")
DEFAULT_MODELS = [  # ordered light → heavy; non-ONNX may not ingest in all envs
    "squeezenet1.1-7.onnx",
    "mobilenet_v1_0.25_128.tflite",
    "har-cnn-full.h5",
    "vgg16-weights-notop.h5",
]


def effective_dim(graph_info, sensitivity, allow_pruning=False):
    """D and complexity_bits, computed exactly as the engine builds action
    spaces (see ParetoSearch._effective_dimensionality)."""
    compressible = graph_info.compressible_operators
    protected = sensitivity.get_protection_set(top_p=0.3)
    total = max(1, sum(op.param_count for op in graph_info.operators))
    spaces = {op.name: get_action_space(op, is_protected=op.name in protected,
                                        param_share=op.param_count / total)
              for op in compressible}
    return ParetoSearch._effective_dimensionality(spaces, compressible, allow_pruning)


def hv_curve(pareto_result):
    """Running 2D (accuracy, size) HV after each trial, seeded with the original
    ('do nothing') point. Monotonic non-decreasing."""
    oa, os_ = pareto_result.original_accuracy, pareto_result.original_size
    trials = sorted(pareto_result.all_trials, key=lambda p: p.trial_number)
    front = [[0.0, 1.0]]  # original model normalized loss
    xs, ys = [], []
    for p in trials:
        front.append([1.0 - (p.accuracy / oa if oa > 0 else 0.0),
                      p.model_size_bytes / os_ if os_ > 0 else 0.0])
        xs.append(p.trial_number + 1)
        ys.append(_dominated_hv_2d(front))
    return xs, ys


def knees(xs, ys):
    if not ys:
        return {k: None for k in KNEE_FRACS}
    final = max(ys)
    return {nm: next((x for x, y in zip(xs, ys) if y >= fr * final), xs[-1])
            for nm, fr in KNEE_FRACS.items()}


def sim_early_stop(xs, ys, floor, patience, eps):
    """Where an HV-plateau early stop would fire, and the %-of-final it captures."""
    hist, final = [], (max(ys) if ys else 0.0)
    for x, y in zip(xs, ys):
        if x < floor:
            continue
        hist.append(y)
        if len(hist) > patience:
            past = hist[-patience - 1]
            if past > 1e-12 and (y - past) / past < eps:
                return x, (100 * y / final if final else 100.0)
    return (xs[-1] if xs else 0), 100.0


def run_model(path: Path, n_trials: int, seed: int):
    t0 = time.time()
    result = PEOps.optimize(str(path), n_pareto_trials=n_trials, seed=seed,
                            run_pareto=True, guarantee=False, verbose=False,
                            adaptive=False)  # fixed-length sweep for calibration
    elapsed = time.time() - t0
    pr = result.pareto
    if pr is None or not pr.all_trials:
        raise RuntimeError("no pareto trials recorded")
    D, bits = effective_dim(result.graph_info, result.sensitivity)
    xs, ys = hv_curve(pr)
    kn = knees(xs, ys)
    es_t, es_pct = sim_early_stop(xs, ys, floor=max(30, 10 + 2 * D),
                                  patience=20, eps=1e-3)
    return {
        "model": path.name,
        "arch": result.detection.architecture.value,
        "D": D,
        "complexity_bits": round(bits, 3),
        "n_trials_run": n_trials,
        "n_pareto": pr.n_pareto,
        "hv_final": round(ys[-1], 6) if ys else 0.0,
        **kn,
        "earlystop_at": es_t,
        "earlystop_pct_final": round(es_pct, 2),
        "sweep_sec": round(elapsed, 1),
    }, {"model": path.name, "D": D, "trials": xs, "hv": [round(y, 6) for y in ys]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="single model filename under test-models/")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/tmp/calib")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = ([args.model] if args.model else
              DEFAULT_MODELS if args.all else ["squeezenet1.1-7.onnx"])

    rows, curves = [], []
    for name in models:
        path = TEST_DIR / name
        if not path.exists():
            print(f"[skip] {name}: not found")
            continue
        print(f"\n[run] {name} (trials={args.trials}) ...", flush=True)
        try:
            row, curve = run_model(path, args.trials, args.seed)
            rows.append(row)
            curves.append(curve)
            # persist incrementally so a slow tail model can't lose earlier data
            with open(out_dir / "budget_calib.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            with open(out_dir / "hv_curves.json", "w") as f:
                json.dump(curves, f, indent=2)
            print(f"[done] {name}: D={row['D']} t99={row['t99']} t995={row['t995']} "
                  f"t999={row['t999']} earlystop@{row['earlystop_at']}"
                  f"({row['earlystop_pct_final']}%) [{row['sweep_sec']}s]", flush=True)
        except Exception as e:
            print(f"[fail] {name}: {e}")
            traceback.print_exc()

    if rows:
        print(f"\n{'model':<30}{'D':>4}{'t99':>6}{'t995':>6}{'t999':>6}{'ES@':>7}{'ES%':>8}")
        for r in rows:
            print(f"{r['model']:<30}{r['D']:>4}{str(r['t99']):>6}{str(r['t995']):>6}"
                  f"{str(r['t999']):>6}{str(r['earlystop_at']):>7}{r['earlystop_pct_final']:>7}%")
        print(f"\nCSV: {out_dir / 'budget_calib.csv'}")


if __name__ == "__main__":
    main()
