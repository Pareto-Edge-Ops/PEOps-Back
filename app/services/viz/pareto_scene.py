"""Pareto 3D scene builder — serves every value the SPA's ParetoFrontierPlot3D
computes client-side (Astra-Front/src/features/pareto/components/
ParetoFrontierPlot3D.tsx + components/three/AxesGrid.tsx): padded axis
domains, 0..AXIS point positions, frontier/base colors and scales,
constraint-based dimming, adaptive tick labels and tooltip strings.
"""

from __future__ import annotations

from app.services.detrand import js_round

# ParetoFrontierPlot3D.tsx constants
AXIS = 4.0
DIVISIONS = 4
POINT_BASE = 0.06
PAD_FRAC = 0.06

COLORS = {
    "frontier": "#E1FF6B",
    "point": "#ADB4F3",
    "hoverLerp": "#ffffff",
    "hoverLerpFactor": 0.45,
    "dimFactor": 0.22,
    # Category champions (mirrored in features/pareto/lib/highlights.ts)
    "highlights": {
        "best": "#FFC857",      # highest composite score
        "accuracy": "#40BF6B",  # highest accuracy
        "size": "#5EEAD4",      # smallest size
        "latency": "#F29926",   # lowest latency
    },
}
SCALES = {"base": 1.0, "frontier": 1.4, "hover": 2.3, "highlight": 1.8}

# Color priority when one trial holds several titles.
HIGHLIGHT_PRIORITY = ("best", "accuracy", "size", "latency")


def compute_highlights(trials: list[dict]) -> dict[str, str]:
    """trial id → highlight category. Ties resolve to the FIRST trial in the
    experiment's stable trial order. Exact mirror of the SPA's
    computeHighlights (features/pareto/lib/highlights.ts)."""
    if not trials:
        return {}
    best = max(trials, key=lambda t: t["score"])
    accuracy = max(trials, key=lambda t: t["accuracy"])
    size = min(trials, key=lambda t: t["size"])
    latency = min(trials, key=lambda t: t["latency"])
    by_id: dict[str, str] = {}
    # Reverse priority so higher-priority categories overwrite lower ones.
    for category, trial in (
        ("latency", latency), ("size", size), ("accuracy", accuracy), ("best", best),
    ):
        by_id[trial["id"]] = category
    return by_id


def _pad(lo: float, hi: float, frac: float = PAD_FRAC) -> list[float]:
    span = hi - lo
    margin = span * frac
    return [lo - margin, hi + margin]


def _map_range(value: float, in_min: float, in_max: float,
               out_min: float, out_max: float) -> float:
    """Port of lib/three/scales.ts mapRange (normalize → lerp)."""
    t = 0.5 if in_max - in_min == 0 else (value - in_min) / (in_max - in_min)
    return out_min + (out_max - out_min) * t


# All label formatting goes through js_round so strings match JS toFixed
# byte-for-byte (Python's :.Nf is banker's-rounded and diverges on midpoints).

def fmt_ms(v: float) -> str:
    if v >= 100:
        return f"{js_round(v, 0):.0f} ms"
    if v >= 10:
        return f"{js_round(v, 1):.1f} ms"
    return f"{js_round(v, 2):.2f} ms"


def fmt_mb(v: float) -> str:
    if v >= 100:
        return f"{js_round(v, 0):.0f} MB"
    if v >= 10:
        return f"{js_round(v, 1):.1f} MB"
    return f"{js_round(v, 2):.2f} MB"


def fmt_acc(v: float) -> str:
    return f"{js_round(v, 1):.1f}%"


def _axis(label: str, direction: str, domain: list[float], fmt) -> dict:
    lo, hi = domain
    ticks = []
    for i in range(DIVISIONS + 1):
        value = lo + ((hi - lo) * i) / DIVISIONS  # AxesGrid tickValue
        ticks.append({
            "position": (i * AXIS) / DIVISIONS,   # world coord along the axis
            "value": value,
            "label": fmt(value),
        })
    return {"label": label, "direction": direction, "domain": domain, "ticks": ticks}


def build_pareto_scene(
    exp: dict,
    *,
    max_latency: float | None = None,
    max_accuracy_drop: float | None = None,
    max_size: float | None = None,
) -> dict:
    """`exp` is the frontend-shaped ParetoExperiment payload. Constraint
    parameters default to the experiment's own budget (the SPA's sliders can
    override via query params)."""
    trials = exp["trials"]
    budget = exp["budget"]
    constraints = {
        "maxLatency": budget["maxLatency"] if max_latency is None else max_latency,
        "maxAccuracyDrop": (
            budget["maxAccuracyDrop"] if max_accuracy_drop is None else max_accuracy_drop
        ),
        "maxSize": budget["maxSize"] if max_size is None else max_size,
    }

    if trials:
        lats = [t["latency"] for t in trials]
        accs = [t["accuracy"] for t in trials]
        sizes = [t["size"] for t in trials]
        lat_domain = _pad(min(lats), max(lats))
        acc_domain = _pad(min(accs), max(accs))
        size_domain = _pad(min(sizes), max(sizes))
    else:
        lat_domain = acc_domain = size_domain = [0.0, 1.0]

    base_acc = exp["baseAccuracy"]
    highlights = compute_highlights(trials)
    points = []
    frontier_count = 0
    for t in trials:
        acc_drop = base_acc - t["accuracy"]
        highlight = highlights.get(t["id"])
        dimmed = highlight is None and not (
            t["latency"] <= constraints["maxLatency"]
            and acc_drop <= constraints["maxAccuracyDrop"]
            and t["size"] <= constraints["maxSize"]
        )
        if t["onFrontier"]:
            frontier_count += 1
        x = _map_range(t["latency"], lat_domain[0], lat_domain[1], 0, AXIS)
        y = _map_range(t["accuracy"], acc_domain[0], acc_domain[1], 0, AXIS)
        z = _map_range(t["size"], size_domain[0], size_domain[1], 0, AXIS)
        if highlight is not None:
            color = COLORS["highlights"][highlight]
            scale = SCALES["highlight"]
        elif t["onFrontier"]:
            color, scale = COLORS["frontier"], SCALES["frontier"]
        else:
            color, scale = COLORS["point"], SCALES["base"]
        points.append({
            **t,
            "position": {"x": x, "y": y, "z": z},
            # group offset [-AXIS/2, 0, -AXIS/2] applied — absolute world coords
            "worldPosition": {"x": x - AXIS / 2, "y": y, "z": z - AXIS / 2},
            "color": color,
            "scale": scale,
            "highlight": highlight,
            "dimmed": dimmed,
            "accuracyDrop": acc_drop,
            "tooltip": {
                "title": t["name"],
                "quant": t["quant"],
                "accuracy": f"{js_round(t['accuracy'], 2):.2f}%",
                "latency": fmt_ms(t["latency"]),
                "size": fmt_mb(t["size"]),
                "score": f"{js_round(t['score'], 1):.1f}",
                "frontier": t["onFrontier"],
            },
        })

    return {
        "modelId": exp["modelId"],
        "modelName": exp["modelName"],
        "experimentId": exp["experimentId"],
        "status": exp["status"],
        "baseAccuracy": base_acc,
        "budget": budget,
        "constraints": constraints,
        "axis": {
            "size": AXIS,
            "divisions": DIVISIONS,
            "x": _axis("Latency", "down", lat_domain, fmt_ms),
            "y": _axis("Accuracy", "up", acc_domain, fmt_acc),
            "z": _axis("Size", "down", size_domain, fmt_mb),
        },
        "groupOffset": {"x": -AXIS / 2, "y": 0.0, "z": -AXIS / 2},
        "camera": {"position": [8, 6, 8], "fov": 45, "near": 0.1, "far": 100},
        "constants": {
            "pointRadius": POINT_BASE,
            "colors": COLORS,
            "scales": SCALES,
        },
        "points": points,
        "counts": {"points": len(points), "frontier": frontier_count},
    }
