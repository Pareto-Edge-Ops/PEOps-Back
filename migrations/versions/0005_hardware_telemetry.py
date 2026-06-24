"""hardware telemetry: GPU sample columns on telemetry_snapshots

Revision ID: 0005_hardware_telemetry
Revises: 0004_client_telemetry
Create Date: 2026-06-23

Adds the dynamic accelerator sample (GPU util%, GPU mem used, GPU temperature)
to telemetry_snapshots so the dashboard can chart GPU/CPU resource utilization
over time and attribute inference speed to real hardware. Static GPU/CPU
identity (gpuName, cpuModel, cores, providers, …) rides in the existing
runtime_json blob, so no column is needed for it.

All three columns are nullable — a host with no NVIDIA GPU simply leaves them
NULL, which the aggregation reads as "no GPU on this host".

Idempotent and Postgres-only (SQLite/test DBs build the schema fresh from
SQLModel metadata via create_all, which already includes these columns).
"""
from __future__ import annotations

from alembic import op

revision = "0005_hardware_telemetry"
down_revision = "0004_client_telemetry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite path uses create_all, which already has all of this.

    op.execute("ALTER TABLE telemetry_snapshots ADD COLUMN IF NOT EXISTS gpu_util_pct DOUBLE PRECISION")
    op.execute("ALTER TABLE telemetry_snapshots ADD COLUMN IF NOT EXISTS gpu_mem_used_mb DOUBLE PRECISION")
    op.execute("ALTER TABLE telemetry_snapshots ADD COLUMN IF NOT EXISTS gpu_temp_c DOUBLE PRECISION")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE telemetry_snapshots DROP COLUMN IF EXISTS gpu_temp_c")
    op.execute("ALTER TABLE telemetry_snapshots DROP COLUMN IF EXISTS gpu_mem_used_mb")
    op.execute("ALTER TABLE telemetry_snapshots DROP COLUMN IF EXISTS gpu_util_pct")
