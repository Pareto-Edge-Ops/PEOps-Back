"""google oauth columns on users

Revision ID: 0002_google_oauth
Revises: 0001_initial
Create Date: 2026-06-08

Adds auth_provider + google_sub and relaxes password_hash to nullable (OAuth-only
accounts have no password). Idempotent: 0001 builds the schema from the live
SQLModel metadata via create_all, so a freshly-created DB already has these
columns — IF NOT EXISTS / DROP NOT NULL make this a safe no-op there, while an
already-migrated DB gets the new columns added.
"""
from __future__ import annotations

from alembic import op

revision = "0002_google_oauth"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite path uses create_all, which already has these columns.
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider VARCHAR NOT NULL DEFAULT 'password'")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_sub VARCHAR")
    op.execute("ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL")
    op.execute("CREATE INDEX IF NOT EXISTS ix_users_google_sub ON users (google_sub)")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_users_google_sub")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS google_sub")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS auth_provider")
