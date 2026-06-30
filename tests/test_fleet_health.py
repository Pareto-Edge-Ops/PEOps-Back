"""Fleet health rollup — workspace deployment health for the dashboard KPI strip.

The service is exercised over an ISOLATED in-memory DB so each health signal is
asserted deterministically (the shared session DB accumulates deployments/alerts
across tests, which would make precise counts flaky). One endpoint contract test
rides the real client to lock the response shape + status enum.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, create_engine

from app.dbmodels import AlertRow, DeploymentRow
from app.repositories import put_cached_result
from app.services import fleet

STATUSES = {"healthy", "warning", "danger", "idle"}


@pytest.fixture
def mem_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _dep(session, *, dep_id, model_id="m1", status="live", drift=0.0, user="u1"):
    session.add(DeploymentRow(
        id=dep_id, user_id=user, model_id=model_id,
        endpoint=f"https://x/{dep_id}", region="ap-northeast-2",
        status=status, accuracy_drift=drift,
    ))
    session.commit()


def _alert(session, *, alert_id, level, user="u1", model_id="m1"):
    session.add(AlertRow(
        id=alert_id, user_id=user, model_id=model_id,
        level=level, title="t", body="b", at="2026-06-30T00:00:00Z",
    ))
    session.commit()


# ── status derivation ──────────────────────────────────────────────────────


def test_idle_when_no_deployments(mem_session):
    h = fleet.workspace_fleet_health(mem_session, "u1")
    assert h == {
        "status": "idle", "liveDeployments": 0, "totalDeployments": 0,
        "driftingDeployments": 0, "openAlerts": 0, "dangerAlerts": 0,
        "warningAlerts": 0,
    }


def test_paused_only_is_idle(mem_session):
    _dep(mem_session, dep_id="d1", status="paused")
    h = fleet.workspace_fleet_health(mem_session, "u1")
    assert h["totalDeployments"] == 1
    assert h["liveDeployments"] == 0
    assert h["status"] == "idle"


def test_healthy_live_no_alerts(mem_session):
    _dep(mem_session, dep_id="d1", status="live")
    _dep(mem_session, dep_id="d2", status="canary")  # canary != paused → live
    h = fleet.workspace_fleet_health(mem_session, "u1")
    assert h["status"] == "healthy"
    assert h["liveDeployments"] == 2
    assert h["totalDeployments"] == 2
    assert h["openAlerts"] == 0


def test_warning_from_warning_alert(mem_session):
    _dep(mem_session, dep_id="d1")
    _alert(mem_session, alert_id="a1", level="warning")
    h = fleet.workspace_fleet_health(mem_session, "u1")
    assert h["status"] == "warning"
    assert h["warningAlerts"] == 1
    assert h["openAlerts"] == 1


def test_danger_from_danger_alert_outranks_warning(mem_session):
    _dep(mem_session, dep_id="d1")
    _alert(mem_session, alert_id="a1", level="warning")
    _alert(mem_session, alert_id="a2", level="danger")
    h = fleet.workspace_fleet_health(mem_session, "u1")
    assert h["status"] == "danger"
    assert h["dangerAlerts"] == 1
    assert h["warningAlerts"] == 1
    assert h["openAlerts"] == 2


# ── drift reuses the model's own tolerance budget ──────────────────────────


def test_drift_past_model_tolerance_is_danger(mem_session):
    put_cached_result(
        mem_session, "m1", "pareto", {"budget": {"maxAccuracyDrop": 2.0}}, "u1",
    )
    _dep(mem_session, dep_id="d1", model_id="m1", drift=5.0)  # 5 > 2 → drifting
    h = fleet.workspace_fleet_health(mem_session, "u1")
    assert h["driftingDeployments"] == 1
    assert h["status"] == "danger"  # drift alone escalates


def test_drift_within_tolerance_is_healthy(mem_session):
    put_cached_result(
        mem_session, "m1", "pareto", {"budget": {"maxAccuracyDrop": 2.0}}, "u1",
    )
    _dep(mem_session, dep_id="d1", model_id="m1", drift=1.0)  # 1 < 2 → fine
    h = fleet.workspace_fleet_health(mem_session, "u1")
    assert h["driftingDeployments"] == 0
    assert h["status"] == "healthy"


def test_drift_unknown_budget_never_flagged(mem_session):
    """No cached budget → we don't fabricate a threshold; never flag drift."""
    _dep(mem_session, dep_id="d1", model_id="m1", drift=99.0)
    h = fleet.workspace_fleet_health(mem_session, "u1")
    assert h["driftingDeployments"] == 0
    assert h["status"] == "healthy"


def test_paused_deployment_drift_ignored(mem_session):
    """Only LIVE deployments count toward drift — a paused one is not serving."""
    put_cached_result(
        mem_session, "m1", "pareto", {"budget": {"maxAccuracyDrop": 2.0}}, "u1",
    )
    _dep(mem_session, dep_id="d1", model_id="m1", status="paused", drift=9.0)
    h = fleet.workspace_fleet_health(mem_session, "u1")
    assert h["driftingDeployments"] == 0
    assert h["status"] == "idle"


def test_scoped_to_user(mem_session):
    _dep(mem_session, dep_id="d1", user="u1")
    _dep(mem_session, dep_id="d2", user="u2")
    _alert(mem_session, alert_id="a1", level="danger", user="u2")
    h = fleet.workspace_fleet_health(mem_session, "u1")
    assert h["totalDeployments"] == 1
    assert h["openAlerts"] == 0
    assert h["status"] == "healthy"


# ── endpoint contract (real client) ─────────────────────────────────────────


def test_fleet_health_endpoint_shape(client, make_live_model, deploy_model):
    mid = make_live_model("fleet-ep.onnx")["modelId"]
    deploy_model(mid)
    h = client.get("/api/dashboard/fleet-health").json()
    assert set(h) == {
        "status", "liveDeployments", "totalDeployments", "driftingDeployments",
        "openAlerts", "dangerAlerts", "warningAlerts",
    }
    assert h["status"] in STATUSES
    assert h["liveDeployments"] >= 1          # the deployment we just created
    assert h["totalDeployments"] >= h["liveDeployments"]
    assert h["dangerAlerts"] + h["warningAlerts"] <= h["openAlerts"]
