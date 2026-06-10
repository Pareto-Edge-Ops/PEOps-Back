"""Deployment API keys — hashed-at-rest bearer tokens for the inference endpoint.

The plaintext key is shown to the user exactly once (at creation / rotation);
only its sha256 hash is stored, so a leaked DB never yields a usable key. Lookups
hash the presented token and match against `key_hash` (indexed).
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.config import iso
from app.dbmodels import ApiKeyRow

KEY_PREFIX = "peops_sk_live_"


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def _display(plaintext: str) -> str:
    """Masked form for the UI: peops_sk_live_3i7c…b71c."""
    head = KEY_PREFIX + plaintext[len(KEY_PREFIX):len(KEY_PREFIX) + 4]
    return f"{head}…{plaintext[-4:]}"


def issue_key(session: Session, *, user_id: str, deployment_id: str) -> tuple[ApiKeyRow, str]:
    """Create a key for a deployment; returns (row, plaintext-shown-once)."""
    plaintext = KEY_PREFIX + secrets.token_hex(16)
    row = ApiKeyRow(
        id=f"key_{secrets.token_hex(6)}",
        user_id=user_id,
        deployment_id=deployment_id,
        key_hash=hash_key(plaintext),
        prefix=_display(plaintext),
        created_at=iso(datetime.now(timezone.utc)),
        revoked=False,
    )
    session.add(row)
    session.commit()
    return row, plaintext


def rotate_key(session: Session, *, user_id: str, deployment_id: str) -> tuple[ApiKeyRow, str]:
    """Revoke all existing keys for the deployment and issue a fresh one."""
    rows = session.exec(
        select(ApiKeyRow).where(ApiKeyRow.deployment_id == deployment_id)
    ).all()
    for r in rows:
        r.revoked = True
        session.add(r)
    session.commit()
    return issue_key(session, user_id=user_id, deployment_id=deployment_id)


def revoke_deployment_keys(session: Session, deployment_id: str) -> None:
    rows = session.exec(
        select(ApiKeyRow).where(ApiKeyRow.deployment_id == deployment_id)
    ).all()
    for r in rows:
        r.revoked = True
        session.add(r)
    if rows:
        session.commit()


def resolve_key(session: Session, plaintext: str | None) -> ApiKeyRow | None:
    """Return the live (non-revoked) key matching the token, or None."""
    if not plaintext:
        return None
    row = session.exec(
        select(ApiKeyRow).where(
            ApiKeyRow.key_hash == hash_key(plaintext),
            ApiKeyRow.revoked == False,  # noqa: E712 — SQL boolean compare
        )
    ).first()
    return row


def touch_key(session: Session, key: ApiKeyRow) -> None:
    """Best-effort last_used_at bump (skips writes more often than once/30s)."""
    now = datetime.now(timezone.utc)
    if key.last_used_at:
        try:
            prev = datetime.fromisoformat(key.last_used_at.replace("Z", "+00:00"))
            if (now - prev).total_seconds() < 30:
                return
        except ValueError:
            pass
    key.last_used_at = iso(now)
    session.add(key)
    session.commit()
