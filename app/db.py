"""Database engine + session helpers — SQLite (dev/tests) or PostgreSQL (prod).

The dialect is chosen from ASTRA_DATABASE_URL (Postgres) or ASTRA_DB_PATH
(SQLite fallback). SQLite-only tuning (WAL pragma, check_same_thread) is guarded
by dialect; Postgres gets a real connection pool for many API/worker processes.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings

_engine = None


def _normalized_url(raw: str) -> str:
    # Use psycopg3 for Postgres. Accept bare postgres:// / postgresql:// too.
    if raw.startswith("postgresql+"):
        return raw
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw[len("postgresql://"):]
    if raw.startswith("postgres://"):
        return "postgresql+psycopg://" + raw[len("postgres://"):]
    return raw


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        url = _normalized_url(settings.effective_database_url)

        if url.startswith("sqlite"):
            _engine = create_engine(
                url, connect_args={"check_same_thread": False, "timeout": 30},
            )

            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_connection, _record):  # pragma: no cover
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()
        else:
            # Postgres: pool sized for multiple API workers + the job worker.
            _engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                pool_recycle=1800,
            )

    return _engine


def reset_engine() -> None:
    """Dispose the cached engine (tests swap ASTRA_DB_PATH between runs)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def init_db() -> None:
    import app.dbmodels  # noqa: F401 — register tables

    SQLModel.metadata.create_all(get_engine())


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session


def open_session() -> Session:
    """Non-dependency session for background workers."""
    return Session(get_engine())
