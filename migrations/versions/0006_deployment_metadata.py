"""deployment metadata: user-editable description on deployments

Revision ID: 0006_deployment_metadata
Revises: 0005_hardware_telemetry
Create Date: 2026-06-29

Adds a free-text `description` column to deployments so users can annotate each
serving endpoint ("Seoul region prod mirror", "internal test box") and tell
multiple deployments of the same model apart on the Deployments panel. The
`name` column already exists (added in 0003); this only adds the note.

Idempotent and Postgres-only (SQLite/test DBs build the schema fresh from
SQLModel metadata via create_all, which already includes this column).
"""
from __future__ import annotations

from alembic import op

revision = "0006_deployment_metadata"
down_revision = "0005_hardware_telemetry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite path uses create_all, which already has this column.

    op.execute("ALTER TABLE deployments ADD COLUMN IF NOT EXISTS description VARCHAR NOT NULL DEFAULT ''")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE deployments DROP COLUMN IF EXISTS description")
