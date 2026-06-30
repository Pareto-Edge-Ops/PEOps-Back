"""In-app feedback / feature-request DTOs (mirrors Astra-Front/src/features/feedback)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# The submit endpoint takes multipart Form fields (so an optional screenshot can
# ride along), not a JSON body — so there's no CreateFeedbackRequest model; the
# router declares kind/message/page/locale as Form params directly.
FeedbackKind = Literal["feature", "bug", "question", "other"]


class FeedbackResult(BaseModel):
    id: str
    status: str
    # Set only once the background task has opened the GitHub issue (and the row
    # is re-read). The POST returns before that, so this is normally null.
    githubIssueUrl: str | None = None
