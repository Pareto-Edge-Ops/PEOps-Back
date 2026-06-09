"""Alembic environment — drives migrations against the app's configured engine.

Target metadata is SQLModel.metadata (the same tables the app defines), so the
schema is authored once in app/dbmodels.py and never drifts from the migrations.
"""

from __future__ import annotations

from alembic import context
from sqlmodel import SQLModel

import app.dbmodels  # noqa: F401 — register all tables on the metadata
from app.config import get_settings
from app.db import _normalized_url, get_engine

target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    url = _normalized_url(get_settings().effective_database_url)
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = get_engine()
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
