"""initial schema (built from SQLModel metadata)

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-08

The schema is the single source of truth in app/dbmodels.py. Building it via
SQLModel.metadata.create_all keeps Postgres column types exactly as the models
map them (ISO-8601 timestamps stay VARCHAR, the cached JSON `payload` stays
VARCHAR/TEXT, booleans become native BOOLEAN) — no hand-typed columns to drift.
"""
from __future__ import annotations

from alembic import op
from sqlmodel import SQLModel

import app.dbmodels  # noqa: F401 — register tables on the metadata

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    SQLModel.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    SQLModel.metadata.drop_all(bind=op.get_bind())
