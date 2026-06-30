"""Test fixtures — fast pipeline, tmp SQLite, REAL pipeline-produced models.

The DB starts EMPTY (no demo fixtures exist anymore). `empty_state` snapshots
the zero-state responses before any model exists; `real_model` runs the real
fast pipeline once per session; `statedict_model` uploads a raw state_dict
checkpoint and exercises the weight-only pipeline.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


def wait_run(client: TestClient, model_id: str, run_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/models/{model_id}/ingestion/{run_id}").json()
        if status["status"] != "streaming":
            return status
        time.sleep(0.3)
    raise AssertionError(f"run {run_id} still streaming after {timeout}s")


def wait_model_terminal(client: TestClient, model_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    m: dict = {}
    while time.time() < deadline:
        m = client.get(f"/api/models/{model_id}").json()
        if m.get("status") in ("draft", "failed"):
            return m
        time.sleep(0.2)
    return m


@pytest.fixture(scope="session")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    tmp = tmp_path_factory.mktemp("astra")
    os.environ.update({
        "ASTRA_DB_PATH": str(tmp / "test.db"),
        "ASTRA_STORAGE_DIR": str(tmp / "storage"),
        "ASTRA_WORK_DIR": str(tmp / "work"),
        "ASTRA_FAST_PIPELINE": "1",
        "ASTRA_SEED": "42",
        "ASTRA_JOB_TIMEOUT_SEC": "120",
        # No Redis broker in the suite — run pipelines inline on daemon threads.
        "ASTRA_INLINE_JOBS": "1",
        # The suite fires many requests fast; don't rate-limit it.
        "ASTRA_RATE_LIMIT_ENABLED": "0",
        # TestClient speaks http:// — Secure cookies wouldn't be stored/sent,
        # so the session cookie must be non-Secure for the suite.
        "ASTRA_COOKIE_SECURE": "0",
        "ASTRA_JWT_SECRET": "test-secret-not-for-prod",
        # Pin Google OFF for the suite regardless of a developer's local .env
        # (env vars outrank .env). The default-disabled contract tests rely on
        # this; the enabled-path tests opt in via _enable_google().
        "ASTRA_GOOGLE_CLIENT_ID": "",
        "ASTRA_GOOGLE_CLIENT_SECRET": "",
        # Pin feedback→GitHub OFF for the suite regardless of a developer's local
        # .env (env vars outrank .env). Otherwise the feedback contract tests would
        # fire the background task against the REAL GitHub API and open live issues.
        "ASTRA_FEEDBACK_GITHUB_TOKEN": "",
        "ASTRA_FEEDBACK_GITHUB_REPO": "",
        # Enable the demo traffic generator for the telemetry tests. The inline
        # drift-monitor loop is pinned OFF (production default is ON) so passes
        # only run when a test calls them — keeping alert assertions deterministic.
        "ASTRA_TELEMETRY_SIM_ENABLED": "1",
        "ASTRA_MONITOR_INLINE_ENABLED": "0",
    })

    # Settings/engine/storage may have been cached by an earlier import — reset.
    from app.config import get_settings
    from app.db import reset_engine
    from app.services.jobs import reset_job_manager
    from app.services.storage import reset_storage

    get_settings.cache_clear()
    reset_engine()
    reset_job_manager()
    reset_storage()

    from app.main import create_app

    with TestClient(create_app()) as c:
        # Sign up the suite's user; httpx keeps the Set-Cookie in its jar so
        # every subsequent request rides authenticated. Existing contract tests
        # therefore need no changes — they hit the same authed client.
        r = c.post("/api/auth/signup", json={
            "email": "suite@astra.dev", "password": "suite-pass-1234", "name": "Suite",
        })
        assert r.status_code == 200, r.text
        yield c


@pytest.fixture(scope="session")
def empty_state(client: TestClient) -> dict:
    """Zero-state responses captured BEFORE any model exists.

    Eagerly evaluated by being the first session fixture every empty-DB test
    depends on; `real_model`/`statedict_model` depend on it so ordering is
    guaranteed regardless of which test runs first.
    """
    return {
        "models": client.get("/api/models").json(),
        "summary": client.get("/api/dashboard/summary").json(),
        "runs": client.get("/api/dashboard/runs").json(),
        "compression_map": client.get("/api/dashboard/compression-map").json(),
        "top_models": client.get("/api/dashboard/top-models").json(),
        "guarantee_coverage": client.get("/api/dashboard/guarantee-coverage").json(),
        "fleet_health": client.get("/api/dashboard/fleet-health").json(),
        "activity": client.get("/api/dashboard/activity?limit=50").json(),
    }


@pytest.fixture(scope="session")
def real_model(client: TestClient, empty_state: dict) -> dict:
    """One REAL fast-pipeline model (synthesized ONNX → UOSA → Optuna → DFCV
    → benchmark). Returns {"modelId", "runId"}."""
    r = client.post("/api/models/import", json={"fileName": "fixture-model.onnx"})
    assert r.status_code == 200
    body = r.json()
    status = wait_run(client, body["modelId"], body["runId"])
    assert status["status"] == "completed", status.get("error")
    client.post(f"/api/models/{body['modelId']}/ingestion/complete")
    wait_model_terminal(client, body["modelId"])
    return body


@pytest.fixture(scope="session")
def statedict_model(client: TestClient, empty_state: dict, tmp_path_factory) -> dict:
    """A raw state_dict (.pth) upload — exercises the weight-only pipeline."""
    import torch

    tmp = tmp_path_factory.mktemp("fixtures")
    sd = {
        "module.backbone.0.weight": torch.randn(8, 3, 3, 3),
        "module.backbone.0.bias": torch.randn(8),
        "module.backbone.1.weight": torch.randn(8),
        "module.backbone.1.bias": torch.randn(8),
        "module.backbone.1.running_mean": torch.randn(8),
        "module.backbone.1.running_var": torch.rand(8) + 0.5,
        "module.backbone.1.num_batches_tracked": torch.tensor(10),
        "module.embed.weight": torch.randn(32, 16),
        "module.head.weight": torch.randn(4, 16),
        "module.head.bias": torch.randn(4),
    }
    path = tmp / "fixture-statedict.pth"
    torch.save(sd, str(path))

    with open(path, "rb") as f:
        r = client.post(
            "/api/models/upload",
            files={"file": ("fixture-statedict.pth", f, "application/octet-stream")},
        )
    assert r.status_code == 200
    body = r.json()
    status = wait_run(client, body["modelId"], body["runId"])
    assert status["status"] == "completed", status.get("error")
    wait_model_terminal(client, body["modelId"])
    return body


@pytest.fixture(scope="session")
def make_live_model(client: TestClient, empty_state: dict):
    """Factory: import + complete a FRESH real model and return its body.

    Telemetry/deploy tests use this (not the shared `real_model`) so the events
    they generate stay isolated to their own model — `real_model` keeps zero
    traffic, preserving the benchmark-fallback contract tests."""
    def _make(file_name: str = "live-fixture.onnx") -> dict:
        r = client.post("/api/models/import", json={"fileName": file_name})
        assert r.status_code == 200, r.text
        body = r.json()
        status = wait_run(client, body["modelId"], body["runId"])
        assert status["status"] == "completed", status.get("error")
        client.post(f"/api/models/{body['modelId']}/ingestion/complete")
        wait_model_terminal(client, body["modelId"])
        return body

    return _make


@pytest.fixture
def deploy_model(client: TestClient):
    """Factory: deploy a model and return (deployment_id, plaintext_api_key)."""
    def _deploy(model_id: str, region: str = "ap-northeast-2") -> tuple[str, str]:
        r = client.post(f"/api/models/{model_id}/deployments", json={"region": region})
        assert r.status_code == 200, r.text
        data = r.json()
        return data["deployment"]["id"], data["apiKey"]

    return _deploy


@pytest.fixture(scope="session")
def failed_model(client: TestClient, empty_state: dict) -> dict:
    """A genuinely broken upload — pipeline must fail and raise a real alert."""
    r = client.post(
        "/api/models/upload",
        files={"file": ("broken.onnx", b"this is not a real onnx file",
                        "application/octet-stream")},
    )
    assert r.status_code == 200
    body = r.json()
    status = wait_run(client, body["modelId"], body["runId"])
    assert status["status"] == "failed"
    wait_model_terminal(client, body["modelId"])
    return body
