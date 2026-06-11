"""Windowed input/output distribution statistics.

The aggregator samples a bounded reservoir of requests per window and reduces
them to compact stats: per-input tensor mean/std/min/max/NaN%, and — when the
output looks classifier-shaped ([B, C], C <= 10000) — the argmax class
distribution, a 16-bin top-1 confidence histogram, mean entropy and mean
top-1 confidence. These windows are what the PEOps drift monitor compares
against the deployment's reference to raise prediction/input drift alerts.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Any

_RESERVOIR = 32
_HIST_BINS = 16
_MAX_CLASSES = 10_000
_TOP_CLASSES = 10


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _softmax(row: "Any") -> "Any":
    import numpy as np

    shifted = row - np.max(row)
    e = np.exp(shifted)
    return e / max(1e-12, float(e.sum()))


class WindowAggregator:
    """Accumulates one stats window; `flush()` emits the dict and resets."""

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.window_start = datetime.now(timezone.utc)
        self.n = 0
        self._seen = 0
        self._inputs: list[dict[str, Any]] = []   # reservoir of {name: ndarray}
        self._outputs: list[Any] = []             # reservoir of first-output ndarrays

    def observe(self, inputs: dict[str, Any] | None, output: Any | None) -> None:
        """Reservoir-sample one request (numpy arrays; cheap references only)."""
        self.n += 1
        self._seen += 1
        if len(self._inputs) < _RESERVOIR:
            self._store(inputs, output)
        else:
            j = random.randint(0, self._seen - 1)
            if j < _RESERVOIR:
                self._store(inputs, output, replace_at=j)

    def _store(self, inputs, output, replace_at: int | None = None) -> None:
        item_in = inputs or {}
        if replace_at is None:
            self._inputs.append(item_in)
            self._outputs.append(output)
        else:
            self._inputs[replace_at] = item_in
            self._outputs[replace_at] = output

    def flush(self) -> dict | None:
        """Emit the window stats dict (None when nothing was observed)."""
        if self.n == 0:
            return None
        try:
            window = self._build()
        except Exception:   # stats must never break serving
            window = None
        self._reset()
        return window

    def _build(self) -> dict:
        import numpy as np

        input_stats: dict[str, dict] = {}
        names: set[str] = set()
        for sample in self._inputs:
            names.update(sample.keys())
        for name in names:
            arrays = [
                np.asarray(s[name]).ravel() for s in self._inputs if name in s
            ]
            if not arrays:
                continue
            flat = np.concatenate([a[:4096] for a in arrays]).astype(np.float64)
            finite = flat[np.isfinite(flat)]
            nan_pct = 100.0 * (1 - len(finite) / max(1, len(flat)))
            if len(finite) == 0:
                finite = np.zeros(1)
            input_stats[name] = {
                "mean": round(float(finite.mean()), 6),
                "std": round(float(finite.std()), 6),
                "min": round(float(finite.min()), 6),
                "max": round(float(finite.max()), 6),
                "nanPct": round(nan_pct, 3),
            }

        output: dict = {}
        outs = [o for o in self._outputs if o is not None]
        if outs:
            first = np.asarray(outs[0])
            if first.ndim == 2 and 1 < first.shape[1] <= _MAX_CLASSES:
                class_counts: dict[str, int] = {}
                hist = [0] * _HIST_BINS
                conf_sum = ent_sum = 0.0
                rows = 0
                for o in outs:
                    arr = np.asarray(o, dtype=np.float64)
                    if arr.ndim != 2 or arr.shape[1] != first.shape[1]:
                        continue
                    for row in arr:
                        probs = _softmax(row)
                        top = int(np.argmax(probs))
                        conf = float(probs[top])
                        class_counts[str(top)] = class_counts.get(str(top), 0) + 1
                        hist[min(_HIST_BINS - 1, int(conf * _HIST_BINS))] += 1
                        conf_sum += conf
                        ent_sum += float(-np.sum(probs * np.log(probs + 1e-12)))
                        rows += 1
                if rows:
                    top_items = sorted(class_counts.items(), key=lambda kv: -kv[1])
                    output = {
                        "classDist": {
                            k: round(v / rows, 4) for k, v in top_items[:_TOP_CLASSES]
                        },
                        "hist": hist,
                        "top1ConfMean": round(conf_sum / rows, 4),
                        "entropyMean": round(ent_sum / rows, 4),
                    }
            else:
                sample = np.asarray(outs[0], dtype=np.float64).ravel()[:4096]
                finite = sample[np.isfinite(sample)]
                if len(finite):
                    counts, _edges = np.histogram(finite, bins=_HIST_BINS)
                    output = {"hist": [int(c) for c in counts]}

        return {
            "windowStart": _iso(self.window_start),
            "windowEnd": _iso(datetime.now(timezone.utc)),
            "n": self.n,
            "inputs": input_stats,
            "output": output,
        }
