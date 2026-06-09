"""/architecture/scene + /pareto/scene — math parity with the SPA's renderers.

Expected values are recomputed with the exact formulas from
LayerGraph3D.tsx / ParetoFrontierPlot3D.tsx / AxesGrid.tsx / colorRamp.ts,
evaluated against REAL pipeline-produced models (no generated fixtures).
"""

from __future__ import annotations

import math

COL_SPACING = 1.55
ROW_SPACING = 0.22
SENS_THRESHOLD = 0.55
AXIS = 4.0


# ── architecture scene ──────────────────────────────────────────────────────

def _arch_pair(client, model_id):
    arch = client.get(f"/api/models/{model_id}/architecture").json()
    scene = client.get(f"/api/models/{model_id}/architecture/scene").json()
    return arch, scene


def test_arch_scene_neuron_positions_exact(client, real_model):
    """Every perceptron position must equal the LayerGraph3D formula:
    x = depth·1.55, y = col + (i-(w-1)/2)·0.22, z = zCol ?? 0."""
    arch, scene = _arch_pair(client, real_model["modelId"])
    assert scene["counts"]["neurons"] == len(scene["neurons"])
    cursor = 0
    for li, (node, layer) in enumerate(zip(arch["nodes"], scene["layers"])):
        w = layer["effectiveWidth"]
        assert layer["neuronStart"] == cursor
        assert layer["neuronCount"] == w
        for i in range(w):
            n = scene["neurons"][cursor + i]
            assert n["layerIndex"] == li
            assert n["layerId"] == node["id"]
            assert n["x"] == node["depth"] * COL_SPACING
            assert n["y"] == node["col"] + (i - (w - 1) / 2) * ROW_SPACING
            assert n["z"] == node.get("zCol", 0)
        cursor += w
    assert cursor == scene["counts"]["neurons"]


def test_arch_scene_width_fallback_table():
    """Real-pipeline nodes carry no width field — the kind-fallback table must
    match the LayerGraph3D widthFor switch exactly."""
    from app.services.viz.arch_scene import width_for

    table = {"input": 1, "output": 1, "embed": 18, "conv": 16, "attn": 16,
             "ffn": 20, "dense": 14, "lstm": 14, "bn": 10, "norm": 10,
             "relu": 8, "pool": 8, "softmax": 8, "upsample": 16}
    for kind, expected in table.items():
        assert width_for({"kind": kind}) == expected, kind
    assert width_for({"kind": "???"}) == 12
    assert width_for({"kind": "dense", "width": 7}) == 7  # explicit wins
    assert width_for({"kind": "dense", "width": 0}) == 14  # 0 → fallback


def test_arch_scene_edges_bipartite(client, real_model):
    arch, scene = _arch_pair(client, real_model["modelId"])
    assert len(scene["edges"]) == len(arch["edges"])
    total = 0
    layers = scene["layers"]
    for e_in, e_out in zip(arch["edges"], scene["edges"]):
        assert e_out["from"] == e_in["from"] and e_out["to"] == e_in["to"]
        f, t = layers[e_out["fromLayer"]], layers[e_out["toLayer"]]
        assert f["id"] == e_in["from"] and t["id"] == e_in["to"]
        assert e_out["segmentCount"] == f["neuronCount"] * t["neuronCount"]
        expected_accent = (
            f["sensitivity"] >= SENS_THRESHOLD or t["sensitivity"] >= SENS_THRESHOLD
        )
        assert e_out["accent"] == expected_accent
        total += e_out["segmentCount"]
    assert scene["counts"]["segments"] == total
    assert "segments" not in scene  # only inlined on request


def test_arch_scene_explicit_segments(client, real_model):
    scene = client.get(
        f"/api/models/{real_model['modelId']}/architecture/scene",
        params={"segments": "1"},
    ).json()
    segs = scene["segments"]
    assert len(segs) == scene["counts"]["segments"]
    # First edge's first segment connects neuron[fromStart] → neuron[toStart]
    e = scene["edges"][0]
    fn = scene["neurons"][e["fromNeuronStart"]]
    tn = scene["neurons"][e["toNeuronStart"]]
    assert segs[0] == [fn["x"], fn["y"], fn["z"], tn["x"], tn["y"], tn["z"]]


def test_arch_scene_camera_framing(client, real_model):
    """Camera must match the LayerGraph3D framing effect exactly."""
    _, scene = _arch_pair(client, real_model["modelId"])
    ns = scene["neurons"]
    min_x, max_x = min(n["x"] for n in ns), max(n["x"] for n in ns)
    min_y, max_y = min(n["y"] for n in ns), max(n["y"] for n in ns)
    min_z, max_z = min(n["z"] for n in ns), max(n["z"] for n in ns)
    x_c, y_c, z_c = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2
    x_span, y_span = max_x - min_x + 1.2, max_y - min_y + 1.2
    z_span = max(0.001, max_z - min_z)
    fov_rad = 38 * (math.pi / 180)
    z_for_y = y_span / 2 / math.tan(fov_rad / 2)
    z_for_x = x_span / 2 / (math.tan(fov_rad / 2) * 1.55)
    dist = max(10, max(z_for_x, z_for_y) + 1.5 + z_span * 0.6)
    cam = scene["camera"]
    assert cam["position"] == [
        x_c + min(dist * 0.18, x_span * 0.18),
        y_c + min(dist * 0.18, y_span * 0.55 + 1.2),
        z_c + dist,
    ]
    assert cam["target"] == [x_c, y_c, z_c]
    assert cam["fov"] == 38
    assert cam["bounds"]["min"] == [min_x, min_y, min_z]


def test_arch_scene_colors_and_sensitivity(client, real_model):
    _, scene = _arch_pair(client, real_model["modelId"])
    for layer in scene["layers"]:
        is_sens = layer["sensitivity"] >= SENS_THRESHOLD
        assert layer["isSensitive"] == is_sens
        assert layer["colors"]["base"] == "#a4a5a8"
        assert layer["colors"]["sensitivity"] == ("#7783e3" if is_sens else "#6a6b6e")
        assert layer["colors"]["viridis"].startswith("#")
    consts = scene["constants"]
    assert consts["colSpacing"] == 1.55
    assert consts["rowSpacing"] == 0.22
    assert consts["neuronRadius"] == 0.075
    assert consts["sensitivityThreshold"] == 0.55
    assert consts["colors"]["edgeAccent"] == "#4a55b8"
    assert consts["edgeOpacity"] == {"base": 0.18, "sensitivity": 0.3}


def test_viridis_parity_with_front_stops():
    """colorRamp.ts stops: t=0 → rgb(68,1,84); 0.5 → (33,144,141); 1 → (253,231,37);
    interpolation at 0.125 = midpoint of first segment (Math.round)."""
    from app.services.viz.arch_scene import viridis_hex

    assert viridis_hex(0.0) == "#440154"
    assert viridis_hex(0.5) == "#21908d"
    assert viridis_hex(1.0) == "#fde725"
    assert viridis_hex(-5) == "#440154"  # clamped
    assert viridis_hex(7) == "#fde725"
    # 0.125: k=0.5 → round(68+(59-68)*0.5)=round(63.5)=64, round(1+40.5)=42(JS round half-up)…
    r, g, b = (64, 42, 112)
    assert viridis_hex(0.125) == f"#{r:02x}{g:02x}{b:02x}"


def test_arch_scene_every_layer_described(client, real_model, statedict_model):
    for model in (real_model, statedict_model):
        _, scene = _arch_pair(client, model["modelId"])
        for layer in scene["layers"]:
            assert layer["description"]["title"], layer["id"]


def test_arch_scene_deterministic_and_404(client, real_model):
    a = client.get(f"/api/models/{real_model['modelId']}/architecture/scene").text
    b = client.get(f"/api/models/{real_model['modelId']}/architecture/scene").text
    assert a == b
    assert client.get("/api/models/m_missing/architecture/scene").status_code == 404


# ── pareto scene ────────────────────────────────────────────────────────────

def _pareto_pair(client, model_id, **params):
    exp = client.get(f"/api/models/{model_id}/pareto").json()
    scene = client.get(f"/api/models/{model_id}/pareto/scene", params=params).json()
    return exp, scene


def _map_range(v, lo, hi, out_lo=0.0, out_hi=AXIS):
    t = 0.5 if hi - lo == 0 else (v - lo) / (hi - lo)
    return out_lo + (out_hi - out_lo) * t


def test_pareto_scene_domains_and_positions_exact(client, real_model):
    exp, scene = _pareto_pair(client, real_model["modelId"])
    trials = exp["trials"]
    for key, axis in (("latency", "x"), ("accuracy", "y"), ("size", "z")):
        vals = [t[key] for t in trials]
        lo, hi = min(vals), max(vals)
        span = hi - lo
        assert scene["axis"][axis]["domain"] == [lo - span * 0.06, hi + span * 0.06]
    lat_d = scene["axis"]["x"]["domain"]
    acc_d = scene["axis"]["y"]["domain"]
    size_d = scene["axis"]["z"]["domain"]
    assert len(scene["points"]) == len(trials)
    for t, p in zip(trials, scene["points"]):
        assert p["id"] == t["id"]
        assert p["position"]["x"] == _map_range(t["latency"], *lat_d)
        assert p["position"]["y"] == _map_range(t["accuracy"], *acc_d)
        assert p["position"]["z"] == _map_range(t["size"], *size_d)
        assert 0 <= p["position"]["x"] <= AXIS
        assert p["worldPosition"]["x"] == p["position"]["x"] - AXIS / 2
        assert p["worldPosition"]["y"] == p["position"]["y"]
        assert p["worldPosition"]["z"] == p["position"]["z"] - AXIS / 2


def _expected_highlights(trials: list[dict]) -> dict[str, str]:
    out = {}
    for cat, trial in (
        ("latency", min(trials, key=lambda t: t["latency"])),
        ("size", min(trials, key=lambda t: t["size"])),
        ("accuracy", max(trials, key=lambda t: t["accuracy"])),
        ("best", max(trials, key=lambda t: t["score"])),
    ):
        out[trial["id"]] = cat
    return out


HL_COLORS = {"best": "#FFC857", "accuracy": "#40BF6B",
             "size": "#5EEAD4", "latency": "#F29926"}


def test_pareto_scene_colors_scales_frontier(client, real_model):
    exp, scene = _pareto_pair(client, real_model["modelId"])
    hl = _expected_highlights(exp["trials"])
    frontier = 0
    for t, p in zip(exp["trials"], scene["points"]):
        if t["onFrontier"]:
            frontier += 1
        cat = hl.get(t["id"])
        if cat is not None:
            assert p["highlight"] == cat
            assert p["color"] == HL_COLORS[cat] and p["scale"] == 1.8
            assert p["dimmed"] is False     # champions never dim
        elif t["onFrontier"]:
            assert p["color"] == "#E1FF6B" and p["scale"] == 1.4
        else:
            assert p["color"] == "#ADB4F3" and p["scale"] == 1.0
    assert scene["counts"]["frontier"] == frontier
    assert scene["constants"]["scales"] == {
        "base": 1.0, "frontier": 1.4, "hover": 2.3, "highlight": 1.8,
    }
    assert scene["constants"]["colors"]["dimFactor"] == 0.22
    assert scene["constants"]["colors"]["highlights"] == HL_COLORS
    assert scene["constants"]["pointRadius"] == 0.06


def test_pareto_scene_dimming_default_budget(client, real_model):
    """Default constraints = the experiment's own budget."""
    exp, scene = _pareto_pair(client, real_model["modelId"])
    budget = exp["budget"]
    assert scene["constraints"] == budget
    base = exp["baseAccuracy"]
    hl = _expected_highlights(exp["trials"])
    for t, p in zip(exp["trials"], scene["points"]):
        passes = (
            t["latency"] <= budget["maxLatency"]
            and (base - t["accuracy"]) <= budget["maxAccuracyDrop"]
            and t["size"] <= budget["maxSize"]
        )
        expected_dim = False if t["id"] in hl else (not passes)
        assert p["dimmed"] == expected_dim


def test_pareto_scene_constraint_query_overrides(client, real_model):
    _, scene = _pareto_pair(
        client, real_model["modelId"],
        maxLatency=99999, maxAccuracyDrop=99999, maxSize=99999,
    )
    assert scene["constraints"]["maxLatency"] == 99999
    assert not any(p["dimmed"] for p in scene["points"])


def test_pareto_scene_axis_ticks(client, real_model):
    _, scene = _pareto_pair(client, real_model["modelId"])
    for axis_key, direction in (("x", "down"), ("y", "up"), ("z", "down")):
        axis = scene["axis"][axis_key]
        assert axis["direction"] == direction
        assert len(axis["ticks"]) == 5  # divisions + 1
        lo, hi = axis["domain"]
        for i, tick in enumerate(axis["ticks"]):
            assert tick["position"] == (i * AXIS) / 4
            assert tick["value"] == lo + ((hi - lo) * i) / 4
    assert scene["axis"]["x"]["label"] == "Latency"
    assert scene["axis"]["y"]["label"] == "Accuracy"
    assert scene["axis"]["z"]["label"] == "Size"
    assert scene["axis"]["x"]["ticks"][-1]["label"].endswith(" ms")


def test_pareto_scene_tooltip_formats(client, real_model):
    exp, scene = _pareto_pair(client, real_model["modelId"])
    for t, p in zip(exp["trials"][:20], scene["points"][:20]):
        tip = p["tooltip"]
        assert tip["accuracy"].endswith("%") and len(tip["accuracy"].split(".")[-1]) == 3
        assert tip["latency"].endswith(" ms")
        assert tip["size"].endswith(" MB")
        assert tip["title"] == t["name"]
        assert tip["quant"] == t["quant"]
        assert tip["frontier"] == t["onFrontier"]


def test_pareto_scene_layout_constants(client, real_model):
    _, scene = _pareto_pair(client, real_model["modelId"])
    assert scene["groupOffset"] == {"x": -2.0, "y": 0.0, "z": -2.0}
    assert scene["camera"] == {"position": [8, 6, 8], "fov": 45, "near": 0.1, "far": 100}
    assert scene["axis"]["size"] == 4 and scene["axis"]["divisions"] == 4


def test_pareto_scene_deterministic_and_404(client, real_model):
    a = client.get(f"/api/models/{real_model['modelId']}/pareto/scene").text
    b = client.get(f"/api/models/{real_model['modelId']}/pareto/scene").text
    assert a == b
    assert client.get("/api/models/m_missing/pareto/scene").status_code == 404
