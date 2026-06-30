"""Architecture 3D scene builder — serves every value the SPA's LayerGraph3D
computes client-side (Astra-Front/src/features/architecture/components/
LayerGraph3D.tsx): per-perceptron world positions, effective layer widths,
bipartite edge geometry, sensitivity/viridis colors, camera framing, and the
hover/inspector descriptions (real per-op metadata, see layer_descriptions.py).

All math is a faithful port — constants and formulas match the component
byte-for-byte so a backend-driven renderer is pixel-identical.
"""

from __future__ import annotations

import math

from app.services.detrand import js_round
from app.services.viz.layer_descriptions import describe_kind_fallback

# LayerGraph3D.tsx constants
COL_SPACING = 1.55
ROW_SPACING = 0.22
NEURON_RADIUS = 0.075
SENSITIVITY_THRESHOLD = 0.55
CAMERA_FOV = 38.0

COLORS = {
    "neuron": "#a4a5a8",
    "neuronDim": "#6a6b6e",
    "accent": "#7783e3",
    "edge": "#3a3b40",
    "edgeAccent": "#4a55b8",
    "selected": "#ffffff",
}
EDGE_OPACITY = {"base": 0.18, "sensitivity": 0.3}

# widthFor() fallback table (LayerGraph3D.tsx:28-59)
_KIND_WIDTH = {
    "input": 1, "output": 1,
    "embed": 18, "conv": 16, "attn": 16, "ffn": 20,
    "dense": 14, "lstm": 14,
    "bn": 10, "norm": 10,
    "relu": 8, "pool": 8, "softmax": 8,
    "upsample": 16,
}

# viridis stops (lib/three/colorRamp.ts)
_VIRIDIS_STOPS = [
    (0.0, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.5, (33, 144, 141)),
    (0.75, (94, 201, 98)),
    (1.0, (253, 231, 37)),
]


def width_for(node: dict) -> int:
    w = node.get("width")
    if isinstance(w, (int, float)) and w > 0:
        return int(w)
    return _KIND_WIDTH.get(node.get("kind", ""), 12)


def viridis_hex(t: float) -> str:
    p = max(0.0, min(1.0, t))
    for (ap, argb), (bp, brgb) in zip(_VIRIDIS_STOPS, _VIRIDIS_STOPS[1:], strict=False):
        if ap <= p <= bp:
            k = (p - ap) / (bp - ap)
            # js_round mirrors JS Math.round (half toward +∞), matching colorRamp.ts
            r, g, b = (
                int(js_round(argb[i] + (brgb[i] - argb[i]) * k)) for i in range(3)
            )
            return f"#{r:02x}{g:02x}{b:02x}"
    r, g, b = _VIRIDIS_STOPS[-1][1]
    return f"#{r:02x}{g:02x}{b:02x}"


def _camera(neurons: list[dict]) -> dict:
    """Port of the framing effect (LayerGraph3D.tsx:148-186)."""
    min_x = min(n["x"] for n in neurons)
    max_x = max(n["x"] for n in neurons)
    min_y = min(n["y"] for n in neurons)
    max_y = max(n["y"] for n in neurons)
    min_z = min(n["z"] for n in neurons)
    max_z = max(n["z"] for n in neurons)

    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2
    z_center = (min_z + max_z) / 2
    x_span = max_x - min_x + 1.2
    y_span = max_y - min_y + 1.2
    z_span = max(0.001, max_z - min_z)
    fov_rad = CAMERA_FOV * (math.pi / 180)
    z_for_y = y_span / 2 / math.tan(fov_rad / 2)
    z_for_x = x_span / 2 / (math.tan(fov_rad / 2) * 1.55)
    dist = max(10.0, max(z_for_x, z_for_y) + 1.5 + z_span * 0.6)
    side_offset = min(dist * 0.18, x_span * 0.18)
    up_offset = min(dist * 0.18, y_span * 0.55 + 1.2)
    return {
        "position": [x_center + side_offset, y_center + up_offset, z_center + dist],
        "target": [x_center, y_center, z_center],
        "fov": CAMERA_FOV,
        "near": 0.1,
        "far": 500,
        "bounds": {
            "min": [min_x, min_y, min_z],
            "max": [max_x, max_y, max_z],
        },
    }


def build_architecture_scene(arch: dict, *, include_segments: bool = False) -> dict:
    """`arch` is the frontend-shaped Architecture payload (cached real-pipeline
    mapping or the deterministic generator output)."""
    nodes = arch["nodes"]
    model_type = arch["modelType"]

    # 1) Layer geometry — neurons stacked vertically per layer.
    layers: list[dict] = []
    neurons: list[dict] = []
    cursor = 0
    for li, node in enumerate(nodes):
        w = width_for(node)
        x = node["depth"] * COL_SPACING
        center_y = node["col"]
        center_z = node.get("zCol") or 0
        for i in range(w):
            neurons.append({
                "layerIndex": li,
                "layerId": node["id"],
                "x": x,
                "y": center_y + (i - (w - 1) / 2) * ROW_SPACING,
                "z": center_z,
            })
        sens = float(node["sensitivity"])
        is_sensitive = sens >= SENSITIVITY_THRESHOLD
        layers.append({
            **node,
            "effectiveWidth": w,
            "neuronStart": cursor,
            "neuronCount": w,
            "center": {"x": x, "y": center_y, "z": center_z},
            "isSensitive": is_sensitive,
            "colors": {
                "base": COLORS["neuron"],
                "sensitivity": COLORS["accent"] if is_sensitive else COLORS["neuronDim"],
                "viridis": viridis_hex(sens),
            },
            # Real-pipeline nodes carry a per-op description generated from the
            # actual ONNX metadata; nodes without one (input/output, weights-only
            # layers) get the honest kind-level fallback.
            "description": node.get("description") or describe_kind_fallback(
                node["kind"], node["name"], model_type,
            ),
        })
        cursor += w

    # 2) Edge geometry — full bipartite segments between connected layers.
    layer_idx_by_id = {n["id"]: i for i, n in enumerate(nodes)}
    edges: list[dict] = []
    segments: list[list[float]] = []
    total_segments = 0
    for edge in arch["edges"]:
        fi = layer_idx_by_id.get(edge["from"])
        ti = layer_idx_by_id.get(edge["to"])
        if fi is None or ti is None:
            continue
        f_layer, t_layer = layers[fi], layers[ti]
        f_sens = nodes[fi]["sensitivity"] >= SENSITIVITY_THRESHOLD
        t_sens = nodes[ti]["sensitivity"] >= SENSITIVITY_THRESHOLD
        seg_count = f_layer["neuronCount"] * t_layer["neuronCount"]
        edges.append({
            "from": edge["from"],
            "to": edge["to"],
            "fromLayer": fi,
            "toLayer": ti,
            "fromNeuronStart": f_layer["neuronStart"],
            "fromNeuronCount": f_layer["neuronCount"],
            "toNeuronStart": t_layer["neuronStart"],
            "toNeuronCount": t_layer["neuronCount"],
            "segmentCount": seg_count,
            "accent": f_sens or t_sens,
            "color": COLORS["edge"],
            "accentColor": COLORS["edgeAccent"],
        })
        total_segments += seg_count
        if include_segments:
            for a in range(f_layer["neuronCount"]):
                fn = neurons[f_layer["neuronStart"] + a]
                for b in range(t_layer["neuronCount"]):
                    tn = neurons[t_layer["neuronStart"] + b]
                    segments.append(
                        [fn["x"], fn["y"], fn["z"], tn["x"], tn["y"], tn["z"]]
                    )

    scene = {
        "modelId": arch["modelId"],
        "modelType": model_type,
        "constants": {
            "colSpacing": COL_SPACING,
            "rowSpacing": ROW_SPACING,
            "neuronRadius": NEURON_RADIUS,
            "sensitivityThreshold": SENSITIVITY_THRESHOLD,
            "colors": COLORS,
            "edgeOpacity": EDGE_OPACITY,
            "hoverScale": 1.3,
            "selectedScale": 1.7,
        },
        "layers": layers,
        "neurons": neurons,
        "edges": edges,
        "camera": _camera(neurons) if neurons else None,
        "counts": {
            "layers": len(layers),
            "neurons": len(neurons),
            "edges": len(edges),
            "segments": total_segments,
        },
    }
    if include_segments:
        scene["segments"] = segments
    return scene
