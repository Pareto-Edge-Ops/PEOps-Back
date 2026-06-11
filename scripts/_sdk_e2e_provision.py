#!/usr/bin/env python3
"""SDK e2e step 3: sign up, import a model, deploy, write a handoff file."""

from __future__ import annotations

import argparse
import json
import time
import uuid

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    c = httpx.Client(base_url=args.base.rstrip("/"), timeout=60.0)
    email = f"sdk-e2e+{uuid.uuid4().hex[:8]}@peops.dev"
    r = c.post("/api/auth/signup", json={
        "email": email, "password": "sdk-e2e-pass-1234", "name": "SDK E2E"})
    r.raise_for_status()

    r = c.post("/api/models/import", json={"fileName": "sdk-e2e-model.onnx"})
    r.raise_for_status()
    body = r.json()
    mid, rid = body["modelId"], body["runId"]

    deadline = time.time() + 180
    status = c.get(f"/api/models/{mid}/ingestion/{rid}").json()
    while status.get("status") == "streaming" and time.time() < deadline:
        time.sleep(0.5)
        status = c.get(f"/api/models/{mid}/ingestion/{rid}").json()
    assert status["status"] == "completed", status
    c.post(f"/api/models/{mid}/ingestion/complete")

    r = c.post(f"/api/models/{mid}/deployments", json={"region": "ap-northeast-2"})
    r.raise_for_status()
    dep = r.json()

    cookies = dict(c.cookies)
    with open(args.out, "w") as f:
        json.dump({
            "baseUrl": args.base.rstrip("/"),
            "modelId": mid,
            "deploymentId": dep["deployment"]["id"],
            "apiKey": dep["apiKey"],
            "cookies": cookies,
        }, f, indent=2)
    print(f"   model={mid} deployment={dep['deployment']['id']}")


if __name__ == "__main__":
    main()
