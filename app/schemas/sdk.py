"""Mirrors Astra-Front/src/features/sdk-hub/types.ts.

`GET /api/sdk/snippets` returns an OBJECT keyed by language — not an array.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.common import SdkLanguage


class Recipe(BaseModel):
    id: str
    title: str
    description: str
    language: SdkLanguage
    steps: list[str]


class SdkSnippet(BaseModel):
    language: SdkLanguage
    filename: str
    code: str
