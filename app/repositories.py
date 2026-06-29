"""Query helpers shared by routers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.config import iso
from app.dbmodels import DeploymentRow, IngestionRunRow, ModelRow, ResultCacheRow
from app.schemas.models import ModelListItem

log = logging.getLogger("peops")

# Deployment statuses that actively route traffic. A model is "serving" iff it
# has ≥1 deployment in one of these; an all-"paused" model is deployed but not
# serving (drives the list badge's "Deployed · paused" state).
_SERVING_STATUSES = ("live", "canary")

# Whitelisted sort keys: frontend camelCase → ModelListItem attribute.
_SORT_KEYS = {
    "name", "typeFull", "typeShort", "format", "lastLearnedAt",
    "lastOptimizedAt", "status", "bestAccuracy", "isDeployed", "id",
}


def reconcile_deploy_status(session: Session) -> list[str]:
    """Self-heal the deploy badge invariant: a model with a live deployment
    (``is_deployed``) must read "deployed" in the AI Models list, which renders
    from ``status``. Rows written before the deploy→"deployed" transition fix can
    be stuck at "draft"; flip those to "deployed".

    Deliberately narrow — ONLY ``status == "draft"`` is touched, so an in-flight
    re-analysis ("analyzing"/"optimizing") or a "failed" model is never
    clobbered, and an already-"deployed" row is left untouched. Idempotent
    (no-op once aligned). Returns the names flipped (for logging + tests); the
    caller decides how to surface errors.
    """
    flipped: list[str] = []
    for m in session.exec(select(ModelRow)).all():
        if m.is_deployed and m.status == "draft":
            m.status = "deployed"
            session.add(m)
            flipped.append(m.name)
    if flipped:
        session.commit()
    return flipped


def reconcile_orphaned_runs(session: Session, max_age_sec: int) -> list[str]:
    """Fail ingestion runs stuck in 'streaming' with no live worker behind them —
    e.g. after a worker process died mid-pipeline. A native crash (SIGABRT) skips
    BOTH the cooperative deadline and arq's own timeout, so such a run would
    otherwise hang forever and its model would read 'analyzing' indefinitely.

    Age-based so it can NEVER reap a genuinely in-flight job: a real job cannot
    outlive ``job_timeout_sec``, so only runs whose ``started_at`` is older than
    ``max_age_sec`` (job timeout + margin) are touched. Idempotent. Returns the
    reaped run ids.
    """
    now = datetime.now(timezone.utc)
    reaped: list[str] = []
    streaming = session.exec(
        select(IngestionRunRow).where(IngestionRunRow.status == "streaming")
    ).all()
    for run in streaming:
        try:
            started = datetime.fromisoformat(run.started_at)
        except (ValueError, TypeError):
            continue  # unparseable timestamp — leave it untouched
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if (now - started).total_seconds() < max_age_sec:
            continue  # still within a plausible job lifetime — could be running
        run.status = "failed"
        run.error = run.error or "orphaned — worker stopped before the run finished"
        run.finished_at = iso(now)
        session.add(run)
        # Release the model from its stuck 'analyzing'/'optimizing' state so the
        # UI stops showing an endless analysis and the user can retry.
        model = session.get(ModelRow, run.model_id)
        if (model is not None and model.status in ("analyzing", "optimizing")
                and model.analysis_run_id == run.id):
            model.status = "failed"
            model.analysis_run_id = None
            session.add(model)
        reaped.append(run.id)
    if reaped:
        session.commit()
    return reaped


def model_is_serving(session: Session, model_id: str) -> bool:
    """True iff the model has ≥1 deployment actively routing traffic
    (live/canary). All-paused (or no) deployments → False. One bounded query."""
    hit = session.exec(
        select(DeploymentRow.id).where(
            DeploymentRow.model_id == model_id,
            DeploymentRow.status.in_(_SERVING_STATUSES),  # type: ignore[attr-defined]
        )
    ).first()
    return hit is not None


def model_row_to_item(row: ModelRow, *, serving: bool = False) -> ModelListItem:
    return ModelListItem(
        id=row.id,
        name=row.name,
        typeFull=row.type_full,
        typeShort=row.type_short,
        format=row.format,  # type: ignore[arg-type]
        lastLearnedAt=row.last_learned_at,
        lastOptimizedAt=row.last_optimized_at,
        status=row.status,  # type: ignore[arg-type]
        bestAccuracy=row.best_accuracy,
        isDeployed=row.is_deployed,
        isServing=serving,
        weightsOnly=row.weights_only,
        description=row.description,
        analysisRunId=row.analysis_run_id,
    )


def list_models(
    session: Session, q: str | None, only_deployed: bool, sort: str | None,
    user_id: str,
) -> list[ModelListItem]:
    """Replicates the front mock's observable list semantics
    (models/api/mockHandlers.ts): name-substring filter, isDeployed filter,
    `key:dir` sort comparing values as strings with nulls last. Scoped to owner."""
    rows = session.exec(select(ModelRow).where(ModelRow.user_id == user_id)).all()
    # One bounded query for every serving (live/canary) deployment the user owns,
    # so each row's `isServing` is computed without an N+1 per-model lookup.
    serving_ids = {
        d.model_id
        for d in session.exec(
            select(DeploymentRow).where(
                DeploymentRow.user_id == user_id,
                DeploymentRow.status.in_(_SERVING_STATUSES),  # type: ignore[attr-defined]
            )
        ).all()
    }
    items = [model_row_to_item(r, serving=r.id in serving_ids) for r in rows]

    needle = (q or "").lower()
    if needle:
        items = [m for m in items if needle in m.name.lower()]
    if only_deployed:
        items = [m for m in items if m.isDeployed]

    key, _, direction = (sort or "lastLearnedAt:desc").partition(":")
    if key not in _SORT_KEYS:
        key, direction = "lastLearnedAt", "desc"
    reverse = direction != "asc"

    def sort_key(m: ModelListItem):
        v = getattr(m, key, None)
        # Nulls sink to the bottom regardless of direction (mock: `av == null → 1`).
        return (v is None) != reverse, str(v) if v is not None else ""

    items.sort(key=sort_key, reverse=reverse)
    return items


def get_model(session: Session, model_id: str, user_id: str) -> ModelRow | None:
    row = session.get(ModelRow, model_id)
    if row is None or row.user_id != user_id:
        return None
    return row


def owned_model(session: Session, model_id: str, user_id: str) -> ModelRow:
    """Fetch a model the user owns, or 404 (never 403 — don't leak existence)."""
    from fastapi import HTTPException

    row = get_model(session, model_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="model not found")
    return row


def get_cached_result(
    session: Session, model_id: str, kind: str, user_id: str,
) -> dict | None:
    row = session.exec(
        select(ResultCacheRow).where(
            ResultCacheRow.model_id == model_id,
            ResultCacheRow.kind == kind,
            ResultCacheRow.user_id == user_id,
        )
    ).first()
    return json.loads(row.payload) if row else None


def user_artifact_metas(
    session: Session, user_id: str,
) -> list[tuple[ModelRow, dict]]:
    """Every optimized model owned by the user, paired with the served
    artifact's provenance (source / rung / sizeRatio / sizeBytes / accuracy).
    Empty until a pipeline has produced a compressed artifact. Shared by the
    dashboard (size-reduced / compression-map / guarantee-coverage) and the
    cost lens so both read the same portfolio of served picks."""
    rows = session.exec(
        select(ResultCacheRow).where(
            ResultCacheRow.kind == "artifact_meta",
            ResultCacheRow.user_id == user_id,
        )
    ).all()
    out: list[tuple[ModelRow, dict]] = []
    for row in rows:
        model = session.get(ModelRow, row.model_id)
        if model is None or model.user_id != user_id:
            continue
        meta = get_cached_result(session, row.model_id, "artifact_meta", user_id=user_id)
        if meta:
            out.append((model, meta))
    return out


def put_cached_result(
    session: Session, model_id: str, kind: str, payload: dict, user_id: str,
) -> None:
    existing = session.exec(
        select(ResultCacheRow).where(
            ResultCacheRow.model_id == model_id,
            ResultCacheRow.kind == kind,
            ResultCacheRow.user_id == user_id,
        )
    ).first()
    if existing:
        existing.payload = json.dumps(payload)
        session.add(existing)
    else:
        session.add(ResultCacheRow(
            model_id=model_id, kind=kind, payload=json.dumps(payload), user_id=user_id,
        ))
    session.commit()
