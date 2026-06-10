"""real telemetry: inference_events, api_keys, telemetry_rollup + deployments cols

Revision ID: 0003_telemetry
Revises: 0002_google_oauth
Create Date: 2026-06-09

Adds the live-telemetry tables (raw inference events, deployment API keys, and
per-minute rollups) plus the new deployments columns the drift monitor writes.
Idempotent and Postgres-only here: a freshly-created SQLite/test DB already has
these because 0001 builds the schema from live SQLModel metadata via create_all,
so this migration is a no-op there while an already-migrated Postgres DB gets the
new tables/columns added with IF NOT EXISTS.
"""
from __future__ import annotations

from alembic import op

revision = "0003_telemetry"
down_revision = "0002_google_oauth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite path uses create_all, which already has all of this.

    # --- new columns on deployments (live metrics + business id) ---
    op.execute("ALTER TABLE deployments ADD COLUMN IF NOT EXISTS id VARCHAR NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE deployments ADD COLUMN IF NOT EXISTS name VARCHAR NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE deployments ADD COLUMN IF NOT EXISTS accuracy_drift DOUBLE PRECISION NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE deployments ADD COLUMN IF NOT EXISTS created_at VARCHAR NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE deployments ADD COLUMN IF NOT EXISTS last_event_at VARCHAR")
    op.execute("ALTER TABLE deployments ALTER COLUMN qps SET DEFAULT 0")
    op.execute("ALTER TABLE deployments ALTER COLUMN p95 SET DEFAULT 0")
    op.execute("ALTER TABLE deployments ALTER COLUMN errors_pct SET DEFAULT 0")
    op.execute("CREATE INDEX IF NOT EXISTS ix_deployments_id ON deployments (id)")

    # --- inference_events (raw telemetry facts) ---
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_events (
            pk            SERIAL PRIMARY KEY,
            user_id       VARCHAR NOT NULL DEFAULT '',
            model_id      VARCHAR NOT NULL,
            deployment_id VARCHAR NOT NULL DEFAULT '',
            ts            VARCHAR NOT NULL,
            latency_ms    DOUBLE PRECISION NOT NULL,
            success       BOOLEAN NOT NULL DEFAULT TRUE,
            error_code    VARCHAR,
            batch_size    INTEGER NOT NULL DEFAULT 1,
            region        VARCHAR NOT NULL DEFAULT ''
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_inference_events_user_id ON inference_events (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_inference_events_model_id ON inference_events (model_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_inference_events_deployment_id ON inference_events (deployment_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_inference_events_ts ON inference_events (ts)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_infer_model_ts ON inference_events (model_id, ts)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_infer_deployment_ts ON inference_events (deployment_id, ts)")

    # --- api_keys (hashed bearer keys for inference endpoints) ---
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id            VARCHAR PRIMARY KEY,
            user_id       VARCHAR NOT NULL DEFAULT '',
            deployment_id VARCHAR NOT NULL DEFAULT '',
            key_hash      VARCHAR NOT NULL,
            prefix        VARCHAR NOT NULL,
            created_at    VARCHAR NOT NULL,
            last_used_at  VARCHAR,
            revoked       BOOLEAN NOT NULL DEFAULT FALSE
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_api_keys_user_id ON api_keys (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_api_keys_deployment_id ON api_keys (deployment_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_api_keys_key_hash ON api_keys (key_hash)")

    # --- telemetry_rollup (per-deployment, per-minute pre-aggregate) ---
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_rollup (
            pk            SERIAL PRIMARY KEY,
            deployment_id VARCHAR NOT NULL DEFAULT '',
            bucket_ts     VARCHAR NOT NULL,
            count         INTEGER NOT NULL DEFAULT 0,
            errors        INTEGER NOT NULL DEFAULT 0,
            sum_latency   DOUBLE PRECISION NOT NULL DEFAULT 0,
            p50           DOUBLE PRECISION NOT NULL DEFAULT 0,
            p95           DOUBLE PRECISION NOT NULL DEFAULT 0,
            p99           DOUBLE PRECISION NOT NULL DEFAULT 0,
            CONSTRAINT uq_rollup_dep_bucket UNIQUE (deployment_id, bucket_ts)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_rollup_deployment_id ON telemetry_rollup (deployment_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_rollup_bucket_ts ON telemetry_rollup (bucket_ts)")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS telemetry_rollup")
    op.execute("DROP TABLE IF EXISTS api_keys")
    op.execute("DROP TABLE IF EXISTS inference_events")
    op.execute("DROP INDEX IF EXISTS ix_deployments_id")
    for col in ("last_event_at", "created_at", "accuracy_drift", "name", "id"):
        op.execute(f"ALTER TABLE deployments DROP COLUMN IF EXISTS {col}")
