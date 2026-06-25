"""Query helpers shared by routers."""

from __future__ import annotations

import json

from sqlmodel import Session, select

from app.dbmodels import ModelRow, ResultCacheRow
from app.schemas.models import ModelListItem

# Whitelisted sort keys: frontend camelCase → ModelListItem attribute.
_SORT_KEYS = {
    "name", "typeFull", "typeShort", "format", "lastLearnedAt",
    "lastOptimizedAt", "status", "bestAccuracy", "isDeployed", "id",
}


def model_row_to_item(row: ModelRow) -> ModelListItem:
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
    items = [model_row_to_item(r) for r in rows]

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
