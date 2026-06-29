"""Live-server smoke test — hits EVERY endpoint against a running uvicorn.

Usage:
    uvicorn app.main:app --port 8000   # in another terminal
    python3 scripts/smoke.py [--base http://localhost:8000] [--import-file name.onnx]

The DB contains ONLY real pipeline results (no fixtures), so the smoke first
provisions a real model through the live pipeline, then validates every JSON
response against the backend pydantic mirrors. It also uploads a raw
state_dict checkpoint to exercise the weight-only path. With the server in
real-pipeline mode (PEOPS_FAST_PIPELINE=0) this doubles as the real
Optuna/UOSA e2e check.
"""

from __future__ import annotations

import argparse
import io
import sys
import time

import httpx

sys.path.insert(0, ".")

from app.schemas import (  # noqa: E402
    Alert,
    Architecture,
    ComputeCost,
    DashboardRun,
    Deployment,
    ImportResponse,
    KpiSummary,
    ModelListItem,
    ParetoExperiment,
    ParetoSnapshot,
    Percentiles,
    Recipe,
    SdkSnippet,
    TelemetryKpi,
    TelemetryPoint,
    TopModel,
)
from app.schemas.dashboard import ActivityEvent  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(label: str, fn) -> object | None:
    try:
        out = fn()
        PASS.append(label)
        print(f"  ✓ {label}")
        return out
    except Exception as exc:  # noqa: BLE001
        FAIL.append(f"{label}: {exc}")
        print(f"  ✗ {label}: {exc}")
        return None


def _wait(c: httpx.Client, model_id: str, run_id: str, timeout: float) -> dict:
    t0 = time.time()
    status: dict = {}
    while time.time() - t0 < timeout:
        status = c.get(f"/api/models/{model_id}/ingestion/{run_id}").json()
        print(f"    … {status['status']} {status.get('progress', 0)}%", end="\r")
        if status["status"] != "streaming":
            break
        time.sleep(1.0)
    print()
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--import-file", default="smoke-attn-han.onnx")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    # httpx.Client keeps a cookie jar, so the session cookie from signup rides
    # every subsequent request automatically.
    c = httpx.Client(base_url=args.base, timeout=30.0)

    print(f"── smoke against {args.base}")
    check("healthz", lambda: c.get("/healthz").raise_for_status())

    # ── Authenticate (every /api route is now session-gated) ────────────────
    import uuid

    email = f"smoke_{uuid.uuid4().hex[:10]}@peops.dev"
    check("auth/signup", lambda: c.post("/api/auth/signup", json={
        "email": email, "password": "smoke-pass-1234", "name": "Smoke"}).raise_for_status())
    check("auth/me", lambda: c.get("/api/auth/me").raise_for_status())

    # ── Import → REAL pipeline → results (everything else depends on it) ────
    print(f"── import {args.import_file} (real pipeline)")
    imp = check("models/import", lambda: ImportResponse.model_validate(
        c.post("/api/models/import", json={"fileName": args.import_file})
        .raise_for_status().json()))
    model_id = None
    if imp:
        run_id, model_id = imp.runId, imp.modelId
        status = _wait(c, model_id, run_id, args.timeout)
        check("pipeline completed", lambda: (_ for _ in ()).throw(
            AssertionError(status)) if status["status"] != "completed" else status)
        check("ingestion logs", lambda: (
            lambda logs: logs if len(logs["logs"]) > 10 and logs["done"]
            else (_ for _ in ()).throw(AssertionError("too few logs"))
        )(c.get(f"/api/models/{model_id}/ingestion/{run_id}/logs").raise_for_status().json()))
        check("ingestion/complete", lambda: c.post(
            f"/api/models/{model_id}/ingestion/complete").raise_for_status().json())
        check("real architecture", lambda: Architecture.model_validate(
            c.get(f"/api/models/{model_id}/architecture").raise_for_status().json()))
        check("real pareto", lambda: ParetoExperiment.model_validate(
            c.get(f"/api/models/{model_id}/pareto").raise_for_status().json()))

        def _real_scenes():
            a = c.get(f"/api/models/{model_id}/architecture/scene").raise_for_status().json()
            assert a["counts"]["neurons"] > 0 and a["camera"]
            assert a["layers"][0]["description"]["title"]
            p = c.get(f"/api/models/{model_id}/pareto/scene").raise_for_status().json()
            assert p["counts"]["points"] >= 1
            assert len(p["axis"]["x"]["ticks"]) == 5
            assert all(0 <= pt["position"]["x"] <= 4 for pt in p["points"])
            return True

        def _scene_constraints():
            s = c.get(f"/api/models/{model_id}/pareto/scene"
                      "?maxLatency=99999&maxAccuracyDrop=99999&maxSize=99999") \
                .raise_for_status().json()
            assert not any(p["dimmed"] for p in s["points"])
            return s

        check("real scenes (arch+pareto)", _real_scenes)
        check("pareto/scene constraint override", _scene_constraints)
        check("compressed artifact", lambda: c.get(
            f"/api/models/{model_id}/artifact").raise_for_status())

        def _artifact_info():
            info = c.get(f"/api/models/{model_id}/artifact/info").raise_for_status().json()
            assert info["sizeBytes"] > 0 and len(info["sha256"]) == 64
            assert info["kind"] in ("onnx", "npz")
            return True

        check("artifact info", _artifact_info)

        # Telemetry — REAL benchmark measurements for this model
        check("telemetry kpi (real benchmark)", lambda: TelemetryKpi.model_validate(
            c.get(f"/api/models/{model_id}/telemetry/kpi").raise_for_status().json()))

        def _series():
            pts = [TelemetryPoint.model_validate(p) for p in
                   c.get(f"/api/models/{model_id}/telemetry/series")
                   .raise_for_status().json()]
            assert pts, "expected real benchmark buckets"
            return pts

        check("telemetry series (real buckets)", _series)
        check("telemetry percentiles", lambda: Percentiles.model_validate(
            c.get(f"/api/models/{model_id}/telemetry/percentiles").raise_for_status().json()))
        check("telemetry deployments (real only)", lambda: [Deployment.model_validate(d)
            for d in c.get(f"/api/models/{model_id}/telemetry/deployments")
            .raise_for_status().json()])
        check("telemetry alerts (real only)", lambda: [Alert.model_validate(a)
            for a in c.get(f"/api/models/{model_id}/telemetry/alerts")
            .raise_for_status().json()])

    # ── Dashboard (now backed by the real run) ──────────────────────────────
    check("dashboard/summary", lambda: KpiSummary.model_validate(
        c.get("/api/dashboard/summary").raise_for_status().json()))
    check("dashboard/runs", lambda: [DashboardRun.model_validate(r)
        for r in c.get("/api/dashboard/runs").raise_for_status().json()])
    check("dashboard/runs?status=running", lambda: [DashboardRun.model_validate(r)
        for r in c.get("/api/dashboard/runs?status=running").raise_for_status().json()])
    check("dashboard/pareto-snapshot", lambda: ParetoSnapshot.model_validate(
        c.get("/api/dashboard/pareto-snapshot").raise_for_status().json()))
    check("dashboard/top-models", lambda: [TopModel.model_validate(m)
        for m in c.get("/api/dashboard/top-models").raise_for_status().json()])
    check("dashboard/compute-cost", lambda: ComputeCost.model_validate(
        c.get("/api/dashboard/compute-cost").raise_for_status().json()))
    check("dashboard/activity", lambda: [ActivityEvent.model_validate(a)
        for a in c.get("/api/dashboard/activity?limit=10").raise_for_status().json()])

    # ── Models list/get ─────────────────────────────────────────────────────
    models = check("models list", lambda: [ModelListItem.model_validate(m)
        for m in c.get("/api/models").raise_for_status().json()])
    check("models q+sort", lambda: [ModelListItem.model_validate(m)
        for m in c.get("/api/models?q=smoke&sort=name:asc").raise_for_status().json()])
    if models:
        check(f"models/{models[0].id}", lambda: ModelListItem.model_validate(
            c.get(f"/api/models/{models[0].id}").raise_for_status().json()))

    # ── SDK hub (docs content + honest empty operational lists) ────────────
    check("sdk snippets (object!)", lambda: {k: SdkSnippet.model_validate(v)
        for k, v in c.get("/api/sdk/snippets").raise_for_status().json().items()})
    check("sdk recipes", lambda: [Recipe.model_validate(r)
        for r in c.get("/api/sdk/recipes").raise_for_status().json()])

    # ── Multipart upload of a real ONNX file ───────────────────────────────
    print("── multipart upload (real bytes)")

    def _upload():
        import tempfile

        from app.services.model_factory import synthesize

        with tempfile.TemporaryDirectory() as d:
            synth = synthesize("upload-cnn.onnx", out_dir=d, fast=False)
            data = open(synth.path, "rb").read()
        resp = c.post("/api/models/upload",
                      files={"file": ("upload-cnn.onnx", io.BytesIO(data))})
        return ImportResponse.model_validate(resp.raise_for_status().json())

    up = check("models/upload", _upload)
    if up:
        s = _wait(c, up.modelId, up.runId, args.timeout)
        check("upload pipeline completed", lambda: s if s["status"] == "completed"
              else (_ for _ in ()).throw(AssertionError(s)))

    # ── Raw state_dict upload → honest weight-only pipeline ────────────────
    print("── state_dict upload (weight-only pipeline)")

    def _upload_statedict():
        import tempfile
        from pathlib import Path

        import torch

        sd = {
            "module.backbone.0.weight": torch.randn(8, 3, 3, 3),
            "module.backbone.0.bias": torch.randn(8),
            "module.backbone.1.weight": torch.randn(8),
            "module.backbone.1.bias": torch.randn(8),
            "module.backbone.1.running_mean": torch.randn(8),
            "module.backbone.1.running_var": torch.rand(8) + 0.5,
            "module.head.weight": torch.randn(4, 16),
            "module.head.bias": torch.randn(4),
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "smoke-statedict.pth"
            torch.save(sd, str(path))
            data = path.read_bytes()
        resp = c.post("/api/models/upload",
                      files={"file": ("smoke-statedict.pth", io.BytesIO(data))})
        return ImportResponse.model_validate(resp.raise_for_status().json())

    sd_up = check("models/upload (state_dict)", _upload_statedict)
    if sd_up:
        s2 = _wait(c, sd_up.modelId, sd_up.runId, args.timeout)
        check("state_dict pipeline completed", lambda: s2 if s2["status"] == "completed"
              else (_ for _ in ()).throw(AssertionError(s2)))
        check("state_dict architecture (real inventory)", lambda: Architecture.model_validate(
            c.get(f"/api/models/{sd_up.modelId}/architecture").raise_for_status().json()))

        def _sd_honesty():
            m = c.get(f"/api/models/{sd_up.modelId}").json()
            assert m["bestAccuracy"] is None, "accuracy must never be invented"
            pr = c.get(f"/api/models/{sd_up.modelId}/pareto")
            assert pr.status_code == 404
            assert pr.json()["detail"]["code"] == "weights_only_checkpoint"
            tk = c.get(f"/api/models/{sd_up.modelId}/telemetry/kpi")
            assert tk.status_code == 404
            art = c.get(f"/api/models/{sd_up.modelId}/artifact")
            assert art.status_code == 200 and len(art.content) > 100
            return True

        check("state_dict honesty (no fabricated metrics)", _sd_honesty)

    print(f"\n{'═' * 60}\n  PASS {len(PASS)}  ·  FAIL {len(FAIL)}")
    for f in FAIL:
        print(f"  ✗ {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
