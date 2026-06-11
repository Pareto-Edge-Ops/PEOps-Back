"""GET /api/sdk/* — snippets is an OBJECT keyed by language (zod Record).

API keys / webhooks endpoints were removed: this is a local tool with no auth
or delivery infrastructure, and serving empty husks for them added nothing.
The SDK Hub now centers on the REAL compressed artifact
(`/api/models/{id}/artifact/info` + `/api/models/{id}/sdk/usage`).
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.db import get_session
from app.dbmodels import RecipeRow, SdkSnippetRow
from app.schemas.sdk import Recipe, SdkSnippet

router = APIRouter(prefix="/sdk", tags=["sdk"])


@router.get("/snippets")
def snippets(session: Session = Depends(get_session)) -> dict[str, SdkSnippet]:
    rows = session.exec(select(SdkSnippetRow)).all()
    return {
        r.language: SdkSnippet(language=r.language, filename=r.filename, code=r.code)  # type: ignore[arg-type]
        for r in rows
        if not r.language.startswith("_")  # "_meta" is the docs version marker
    }


@router.get("/recipes")
def recipes(session: Session = Depends(get_session)) -> list[Recipe]:
    rows = session.exec(select(RecipeRow)).all()
    return [
        Recipe(
            id=r.id, title=r.title, description=r.description,
            language=r.language, steps=json.loads(r.steps_json),  # type: ignore[arg-type]
        )
        for r in rows
    ]
