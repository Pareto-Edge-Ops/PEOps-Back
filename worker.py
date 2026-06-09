"""PEOps pipeline worker entrypoint.

Run with either:
    arq app.services.queue.WorkerSettings
or:
    python worker.py

Consumes compression jobs from Redis and runs the real engine. Scale by running
multiple instances (each bounded by PEOPS_JOB_WORKERS concurrent jobs).
"""

from __future__ import annotations

from arq import run_worker

from app.services.queue import WorkerSettings

if __name__ == "__main__":
    run_worker(WorkerSettings)  # type: ignore[arg-type]
