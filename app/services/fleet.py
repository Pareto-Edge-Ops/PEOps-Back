"""Fleet health lens — workspace deployment health rolled up for the dashboard.

Promotes "is anything broken right now?" into the KPI strip. Built entirely from
existing rows (``DeploymentRow`` live metrics + ``AlertRow``), so it adds a
signal without any new measurement.

Honesty rules (the platform's "measured, not guessed" brand):
  • "Drifting" is not a fresh threshold — a live deployment counts as drifting
    only when its measured ``accuracy_drift`` exceeds *its own model's* tolerance
    budget (``maxAccuracyDrop`` from the cached Pareto experiment), the same
    within-tolerance notion the compression map already plots. Models without a
    recorded budget are never flagged (no fabricated alarms).
  • Alerts have no resolved flag in the schema, so every ``AlertRow`` for the
    user is "open" — identical to how the telemetry stream snapshot counts them.
"""

from __future__ import annotations

from sqlmodel import Session, select

from app.dbmodels import AlertRow, DeploymentRow
from app.repositories import get_cached_result


def _model_tolerance(session: Session, model_id: str, user_id: str) -> float | None:
    """The model's own accuracy-drop budget (pts), if a Pareto experiment with a
    budget is cached. None when unknown — caller must not flag drift then."""
    pareto = get_cached_result(session, model_id, "pareto", user_id=user_id) or {}
    return (pareto.get("budget") or {}).get("maxAccuracyDrop")


def workspace_fleet_health(session: Session, user_id: str) -> dict:
    """Roll up deployment health across the workspace.

    status is the worst signal present: any danger alert or drifting deployment →
    "danger"; else any warning alert → "warning"; else live deployments exist →
    "healthy"; else no deployments → "idle"."""
    deployments = session.exec(
        select(DeploymentRow).where(DeploymentRow.user_id == user_id)
    ).all()
    alerts = session.exec(
        select(AlertRow).where(AlertRow.user_id == user_id)
    ).all()

    total = len(deployments)
    live = [d for d in deployments if d.status != "paused"]

    drifting = 0
    for dep in live:
        tol = _model_tolerance(session, dep.model_id, user_id)
        if tol is not None and dep.accuracy_drift > tol:
            drifting += 1

    danger_alerts = sum(1 for a in alerts if a.level == "danger")
    warning_alerts = sum(1 for a in alerts if a.level == "warning")
    open_alerts = len(alerts)

    if danger_alerts > 0 or drifting > 0:
        status = "danger"
    elif warning_alerts > 0:
        status = "warning"
    elif len(live) > 0:
        status = "healthy"
    else:
        status = "idle"

    return {
        "status": status,
        "liveDeployments": len(live),
        "totalDeployments": total,
        "driftingDeployments": drifting,
        "openAlerts": open_alerts,
        "dangerAlerts": danger_alerts,
        "warningAlerts": warning_alerts,
    }
