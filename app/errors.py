"""Global exception handlers — every error leaves as {detail: {code, message}}.

The SPA's apiErrorCode() reads detail.code, so unhandled 500s, validation
failures and rate-limit rejections are normalized to that shape. Existing
HTTPExceptions (string or {code} detail) are left to FastAPI's own handler, so
the contract the SPA already relies on is untouched.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

log = logging.getLogger("astra.error")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {"loc": ".".join(str(p) for p in e.get("loc", [])), "msg": e.get("msg", "")}
            for e in exc.errors()[:10]
        ]
        return JSONResponse(
            status_code=422,
            content={"detail": {
                "code": "validation_error",
                "message": "The request was invalid.",
                "fields": fields,
            }},
        )

    @app.exception_handler(RateLimitExceeded)
    async def _on_rate_limit(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": {
                "code": "rate_limited",
                "message": "Too many requests — please slow down and try again shortly.",
            }},
        )

    @app.exception_handler(Exception)
    async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        log.exception("unhandled error on %s %s (request_id=%s)",
                      request.method, request.url.path, rid)
        return JSONResponse(
            status_code=500,
            content={"detail": {
                "code": "internal_error",
                "message": "Something went wrong on our end. Please try again.",
                **({"requestId": rid} if rid else {}),
            }},
        )
