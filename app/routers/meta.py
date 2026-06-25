"""Public metadata: static facts the SPA needs before (and without) a session.

`/meta/format-capabilities` is the single source of truth the upload UI and the
docs read to label, per format, what the pipeline actually delivers (full
guarantee vs weight-only vs LLM caveat). It mounts outside the cookie gate so the
marketing/import surface can show it pre-login; it exposes no user data.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.capabilities import FORMAT_CAPABILITIES

router = APIRouter(prefix="/meta", tags=["meta"])


@router.get("/format-capabilities")
def format_capabilities() -> JSONResponse:
    """Per-format capability matrix (the routing the worker actually performs)."""
    return JSONResponse([c.model_dump() for c in FORMAT_CAPABILITIES])
