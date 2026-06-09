"""Structured request logging + X-Request-ID propagation (API and worker share
the logging config)."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import Settings

_access = logging.getLogger("peops.access")


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler()
    if settings.log_json:
        from pythonjsonlogger import jsonlogger

        handler.setFormatter(jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
        ))
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
        ))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    # uvicorn duplicates access logs — let our middleware own them.
    logging.getLogger("uvicorn.access").handlers = []


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration = (time.perf_counter() - start) * 1000
            _access.exception(
                "request_error",
                extra={"request_id": rid, "method": request.method,
                       "path": request.url.path, "duration_ms": round(duration, 1)},
            )
            raise
        duration = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = rid
        # Skip the health probes' chatter at INFO.
        if request.url.path not in ("/healthz", "/readyz"):
            _access.info(
                "request",
                extra={"request_id": rid, "method": request.method,
                       "path": request.url.path, "status": response.status_code,
                       "duration_ms": round(duration, 1)},
            )
        return response
