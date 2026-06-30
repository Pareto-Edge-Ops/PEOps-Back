"""Thin HTTP client for the Astra inference endpoint."""

from __future__ import annotations

from typing import Any

import httpx

from ._http import resolve_base_url


class InferenceError(Exception):
    """Raised when the inference endpoint returns a non-2xx response."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"[{status}] {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class AstraClient:
    """Minimal client for POST /api/v1/infer/{deployment_id}."""

    def __init__(
        self,
        deployment_id: str,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = resolve_base_url(base_url)
        self.deployment_id = deployment_id
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    @property
    def _url(self) -> str:
        return f"{self.base_url}/api/v1/infer/{self.deployment_id}"

    def infer(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        region: str | None = None,
        batch: int | None = None,
    ) -> dict[str, Any]:
        """Run one inference. `inputs` maps input name → nested list; pass None to
        let the server synthesize a valid random probe (handy for smoke tests)."""
        body: dict[str, Any] = {"inputs": inputs}
        if region is not None:
            body["region"] = region
        if batch is not None:
            body["batch"] = batch
        resp = self._client.post(
            self._url, json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if resp.status_code >= 400:
            detail = {}
            try:
                detail = resp.json().get("detail", {}) or {}
            except ValueError:
                pass
            raise InferenceError(
                resp.status_code,
                detail.get("code", "error"),
                detail.get("message", resp.text),
            )
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AstraClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
