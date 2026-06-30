"""In-app feedback → GitHub Issues bridge.

A single side-effect function the feedback router schedules as a background task.
It is a no-op (the DB row stays the record) when PEOPS_FEEDBACK_GITHUB_TOKEN /
PEOPS_FEEDBACK_GITHUB_REPO are unset; otherwise it opens an issue in the target
repo (recommend a PRIVATE repo) and records the issue number/url back on the
feedback row. Reuses the app's existing httpx dependency — same shape as
app/auth/google.py.
"""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings
from app.db import open_session
from app.dbmodels import FeedbackRow

log = logging.getLogger("peops")

_GITHUB_API = "https://api.github.com"
# Kind → extra label (always alongside the base "feedback" label). GitHub's
# create-issue endpoint auto-creates any label that doesn't exist yet.
_KIND_LABEL = {
    "feature": "feature-request",
    "bug": "bug",
    "question": "question",
    "other": None,
}


def _issue_payload(row: FeedbackRow, base_url: str | None = None) -> dict:
    stripped = row.message.strip()
    first_line = stripped.splitlines()[0] if stripped else "(no message)"
    who = f"{row.name} <{row.email}>" if row.email else (row.name or row.user_id or "unknown")
    body = (
        f"{row.message}\n\n"
        "---\n"
        f"- **From:** {who}\n"
        f"- **Kind:** {row.kind}\n"
        f"- **Page:** {row.page or '—'}\n"
        f"- **Locale:** {row.locale or '—'}\n"
        f"- **Submitted:** {row.created_at}\n"
        f"- **Feedback id:** `{row.id}`\n"
    )
    # The attachment is served behind the app's session gate, so GitHub can't
    # proxy-render it inline (`![]()` would 401). Link to it instead — a developer
    # signed into PEOps in the same browser can open it.
    if row.attachment_key and base_url:
        url = f"{base_url}/api/feedback/{row.id}/attachment"
        body += f"- **Attachment:** [{row.attachment_name or 'screenshot'}]({url})\n"
    labels = ["feedback"]
    extra = _KIND_LABEL.get(row.kind)
    if extra:
        labels.append(extra)
    return {"title": f"[{row.kind}] {first_line[:80]}", "body": body, "labels": labels}


def create_issue_for_feedback(feedback_id: str, base_url: str | None = None) -> None:
    """Background task: open a GitHub issue for one feedback row and record the
    result. Disabled config → silent no-op. Any failure is logged and swallowed:
    the DB row is the source of truth, so a GitHub outage never loses feedback.

    base_url is the app's absolute origin, used to build the attachment link."""
    settings = get_settings()
    if not settings.github_feedback_enabled:
        return
    try:
        with open_session() as session:
            row = session.get(FeedbackRow, feedback_id)
            if row is None:
                log.warning("feedback %s vanished before issue creation", feedback_id)
                return
            resp = httpx.post(
                f"{_GITHUB_API}/repos/{settings.feedback_github_repo}/issues",
                headers={
                    "Authorization": f"Bearer {settings.feedback_github_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json=_issue_payload(row, base_url),
                timeout=15.0,
            )
            if resp.status_code not in (200, 201):
                log.warning(
                    "GitHub issue creation failed for feedback %s: HTTP %s %s",
                    feedback_id, resp.status_code, resp.text[:200],
                )
                return
            data = resp.json()
            row.github_issue_number = data.get("number")
            row.github_issue_url = data.get("html_url")
            session.add(row)
            session.commit()
            log.info("feedback %s → GitHub issue %s", feedback_id, row.github_issue_url)
    except Exception:  # noqa: BLE001 — a side effect must never break feedback intake
        log.exception("GitHub issue creation errored for feedback %s", feedback_id)
