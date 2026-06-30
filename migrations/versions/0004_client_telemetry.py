"""client telemetry: inference_events source cols + snapshot/window-stats tables

Revision ID: 0004_client_telemetry
Revises: 0003_telemetry
Create Date: 2026-06-11

The astra-sdk pip package serves deployed artifacts locally and ships telemetry
batches to /api/v1/telemetry/{deployment_id}/batch. Request events land in the
existing inference_events table (new `source` discriminator + client latency
breakdown), while periodic system snapshots and windowed input/output stats get
their own tables — the drift monitor reads those for prediction/input drift.

Idempotent and Postgres-only (SQLite/test DBs build the schema fresh from
SQLModel metadata via create_all).
"""
from __future__ import annotations

from alembic import op

revision = "0004_client_telemetry"
down_revision = "0003_telemetry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite path uses create_all, which already has all of this.

    # --- inference_events: client-source discriminator + latency breakdown ---
    op.execute("ALTER TABLE inference_events ADD COLUMN IF NOT EXISTS source VARCHAR NOT NULL DEFAULT 'server'")
    op.execute("ALTER TABLE inference_events ADD COLUMN IF NOT EXISTS latency_pre_ms DOUBLE PRECISION")
    op.execute("ALTER TABLE inference_events ADD COLUMN IF NOT EXISTS latency_post_ms DOUBLE PRECISION")
    op.execute("ALTER TABLE inference_events ADD COLUMN IF NOT EXISTS client_id VARCHAR")
    op.execute("ALTER TABLE inference_events ADD COLUMN IF NOT EXISTS input_sig VARCHAR")
    op.execute("CREATE INDEX IF NOT EXISTS ix_inference_events_source ON inference_events (source)")

    # --- telemetry_snapshots (periodic client system snapshots) ---
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_snapshots (
            pk             SERIAL PRIMARY KEY,
            user_id        VARCHAR NOT NULL DEFAULT '',
            model_id       VARCHAR NOT NULL DEFAULT '',
            deployment_id  VARCHAR NOT NULL DEFAULT '',
            client_id      VARCHAR NOT NULL DEFAULT '',
            ts             VARCHAR NOT NULL,
            cpu_pct        DOUBLE PRECISION NOT NULL DEFAULT 0,
            rss_mb         DOUBLE PRECISION NOT NULL DEFAULT 0,
            throughput_rpm DOUBLE PRECISION NOT NULL DEFAULT 0,
            dropped_events INTEGER NOT NULL DEFAULT 0,
            sdk_version    VARCHAR NOT NULL DEFAULT '',
            runtime_json   VARCHAR NOT NULL DEFAULT '{}'
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_snapshots_user_id ON telemetry_snapshots (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_snapshots_model_id ON telemetry_snapshots (model_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_snapshots_deployment_id ON telemetry_snapshots (deployment_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_snapshots_client_id ON telemetry_snapshots (client_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_snapshots_ts ON telemetry_snapshots (ts)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_snap_dep_ts ON telemetry_snapshots (deployment_id, ts)")

    # --- telemetry_window_stats (windowed input/output distributions) ---
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_window_stats (
            pk               SERIAL PRIMARY KEY,
            user_id          VARCHAR NOT NULL DEFAULT '',
            model_id         VARCHAR NOT NULL DEFAULT '',
            deployment_id    VARCHAR NOT NULL DEFAULT '',
            client_id        VARCHAR NOT NULL DEFAULT '',
            window_start     VARCHAR NOT NULL,
            window_end       VARCHAR NOT NULL DEFAULT '',
            n                INTEGER NOT NULL DEFAULT 0,
            input_stats_json VARCHAR NOT NULL DEFAULT '{}',
            output_json      VARCHAR NOT NULL DEFAULT '{}'
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_window_stats_user_id ON telemetry_window_stats (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_window_stats_model_id ON telemetry_window_stats (model_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_window_stats_deployment_id ON telemetry_window_stats (deployment_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_window_stats_client_id ON telemetry_window_stats (client_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_telemetry_window_stats_window_start ON telemetry_window_stats (window_start)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_winstats_dep_start ON telemetry_window_stats (deployment_id, window_start)")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS telemetry_window_stats")
    op.execute("DROP TABLE IF EXISTS telemetry_snapshots")
    op.execute("ALTER TABLE inference_events DROP COLUMN IF EXISTS client_id")
    op.execute("ALTER TABLE inference_events DROP COLUMN IF EXISTS latency_post_ms")
    op.execute("ALTER TABLE inference_events DROP COLUMN IF EXISTS latency_pre_ms")
    op.execute("ALTER TABLE inference_events DROP COLUMN IF EXISTS source")
