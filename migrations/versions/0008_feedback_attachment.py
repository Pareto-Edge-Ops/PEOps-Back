"""feedback attachment: optional screenshot on a feedback submission

Revision ID: 0008_feedback_attachment
Revises: 0007_feedback
Create Date: 2026-06-30

Adds the `attachment_key` / `attachment_name` columns to the `feedback` table so
a submission can carry an optional screenshot (stored in object storage, served
back from GET /api/feedback/{id}/attachment).

Idempotent and Postgres-only: a freshly-created SQLite/test DB already has these
columns because the schema is built from live SQLModel metadata via create_all,
so this migration is a no-op there while an already-migrated Postgres DB gets the
columns added with IF NOT EXISTS.
"""
from __future__ import annotations

from alembic import op

revision = "0008_feedback_attachment"
down_revision = "0007_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite path uses create_all, which already has these columns.

    op.execute("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS attachment_key VARCHAR")
    op.execute("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS attachment_name VARCHAR")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE feedback DROP COLUMN IF EXISTS attachment_name")
    op.execute("ALTER TABLE feedback DROP COLUMN IF EXISTS attachment_key")
