"""feedback: in-app feedback / feature-request intake

Revision ID: 0007_feedback
Revises: 0006_deployment_metadata
Create Date: 2026-06-30

Adds the `feedback` table backing the in-app feedback widget (sidebar → dialog →
POST /api/feedback). Each row is the durable record of a submission; when a
GitHub repo is configured the router also opens an issue and records its
number/url back here.

Idempotent and Postgres-only: a freshly-created SQLite/test DB already has this
because the schema is built from live SQLModel metadata via create_all, so this
migration is a no-op there while an already-migrated Postgres DB gets the table
added with IF NOT EXISTS.
"""
from __future__ import annotations

from alembic import op

revision = "0007_feedback"
down_revision = "0006_deployment_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite path uses create_all, which already has this table.

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id                  VARCHAR PRIMARY KEY,
            user_id             VARCHAR NOT NULL DEFAULT '',
            email               VARCHAR NOT NULL DEFAULT '',
            name                VARCHAR NOT NULL DEFAULT '',
            kind                VARCHAR NOT NULL DEFAULT 'feature',
            message             VARCHAR NOT NULL,
            page                VARCHAR,
            locale              VARCHAR,
            status              VARCHAR NOT NULL DEFAULT 'open',
            github_issue_number INTEGER,
            github_issue_url    VARCHAR,
            created_at          VARCHAR NOT NULL DEFAULT ''
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_feedback_user_id ON feedback (user_id)")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS feedback")
