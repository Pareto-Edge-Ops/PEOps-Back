"""Hardware-aware telemetry — per-hardware inference speed, GPU/CPU resource
time-series, enriched fleet inventory, and the cost/efficiency lens.

The fleet simulator injects a believable multi-accelerator serving fleet (A10G,
T4, Apple CoreML, hosted x86 CPU) so these views have data on a box without a
GPU; the aggregation treats those rows exactly like real astra-sdk telemetry.
"""

from __future__ import annotations

from app.services import hardware


def _simulate_fleet(client, mid: str) -> dict:
    r = client.post(
        f"/api/models/{mid}/telemetry/simulate",
        json={"count": 480, "hours": 6, "incidents": False, "fleet": True},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_cost_model_is_single_stream_dollars_per_million():
    # 1M inferences at 5 ms single-stream on a $0.526/hr T4 ≈ $0.73.
    assert hardware.est_cost_per_million(5.0, 0.526) == round(0.526 * 5.0 / 3.6, 4)
    assert hardware.est_cost_per_million(0.0, 1.0) == 0.0      # no latency → no cost
    assert hardware.est_cost_per_million(10.0, 0.0) == 0.0     # on-device → no cost


def test_classify_accelerators():
    assert hardware.classify(
        {"gpuName": "NVIDIA T4", "activeProvider": "CUDAExecutionProvider",
         "gpuCount": 1})["accelerator"] == "gpu"
    assert hardware.classify(
        {"activeProvider": "CoreMLExecutionProvider", "cpuModel": "Apple M3",
         "arch": "arm64"})["accelerator"] == "coreml"
    cpu = hardware.classify({"activeProvider": "CPUExecutionProvider", "arch": "x86_64"})
    assert cpu["accelerator"] == "cpu" and cpu["hourlyUsd"] > 0


def test_fleet_simulate_summary(make_live_model, deploy_model, client):
    mid = make_live_model("hw-fleet.onnx")["modelId"]
    deploy_model(mid)
    summary = _simulate_fleet(client, mid)
    fleet = summary["fleet"]
    assert fleet["hosts"] == 4
    assert fleet["gpuHosts"] == 2
    assert fleet["events"] >= 400


def test_hardware_breakdown_groups_by_accelerator(make_live_model, deploy_model, client):
    mid = make_live_model("hw-breakdown.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)

    groups = client.get(f"/api/models/{mid}/telemetry/hardware").json()
    accels = {g["accelerator"] for g in groups}
    assert {"gpu", "coreml"} <= accels                # at least GPU + CoreML present
    assert len(groups) >= 4                           # 4 fleet hosts (+ maybe hosted)

    by_class = {g["deviceClass"]: g for g in groups}
    gpu_groups = [g for g in groups if g["accelerator"] == "gpu"]
    cpu_groups = [g for g in groups if g["accelerator"] in ("cpu", "hosted")]
    assert gpu_groups and cpu_groups

    # The core promise: the GPU serves the SAME artifact faster than the CPU.
    assert min(g["p95"] for g in gpu_groups) < max(g["p95"] for g in cpu_groups)
    # Every group carries a cost estimate and a throughput capacity proxy.
    for g in groups:
        assert g["throughputPerSec"] > 0
        assert g["estCostPer1M"] >= 0
        assert g["samples"] > 0
    # GPU groups expose live accelerator utilization.
    assert any(g["avgGpuUtilPct"] and g["avgGpuUtilPct"] > 0 for g in gpu_groups)
    assert "NVIDIA" in "".join(g["gpuName"] for g in gpu_groups)


def test_resource_series_has_gpu_when_fleet_present(make_live_model, deploy_model, client):
    mid = make_live_model("hw-resources.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)

    res = client.get(f"/api/models/{mid}/telemetry/resources").json()
    assert res["hasGpu"] is True
    assert res["points"], "expected bucketed resource points"
    # CPU% is always present; GPU util appears on at least one bucket.
    assert all("cpuPct" in p and "memMb" in p for p in res["points"])
    assert any(p.get("gpuUtilPct") for p in res["points"])


def test_clients_enriched_with_hardware(make_live_model, deploy_model, client):
    mid = make_live_model("hw-clients.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)

    clients = client.get(f"/api/models/{mid}/telemetry/clients").json()
    assert clients
    keys = set(clients[0])
    assert {"gpuName", "cpuModel", "cpuCores", "activeProvider", "gpuUtilPct"} <= keys
    # A GPU host reports its accelerator name + a live util sample.
    gpu_hosts = [c for c in clients if c["gpuName"]]
    assert gpu_hosts and any(c["gpuUtilPct"] for c in gpu_hosts)


def test_no_traffic_returns_empty_hardware_views(real_model, client):
    """A never-served model has no fleet — endpoints return empty, not errors."""
    mid = real_model["modelId"]
    assert client.get(f"/api/models/{mid}/telemetry/hardware").json() == []
    res = client.get(f"/api/models/{mid}/telemetry/resources").json()
    assert res == {"points": [], "hasGpu": False}
