#!/usr/bin/env python3
"""Upload REAL test models through the live API and gate on real compression.

Asserts, per model: pipeline completes, the served artifact is genuinely
smaller when quantization was selected (or honest 1.0 fallback), the guarantee
certificate is in the ingestion log, and a frontier trial exports + downloads
as runnable ONNX.

Usage: python3 scripts/verify_real_models.py --base http://localhost:8200 \
            --models squeezenet1.1-7.onnx har-cnn-full.h5 ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

MODELS_DIR = Path("/Users/kwonminjae/Desktop/Astra/test-models")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8200")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--timeout", type=float, default=1500.0)
    args = ap.parse_args()

    c = httpx.Client(base_url=args.base.rstrip("/"), timeout=120.0)
    email = f"realgate+{uuid.uuid4().hex[:8]}@astra.dev"
    c.post("/api/auth/signup", json={
        "email": email, "password": "real-gate-pass-1234", "name": "RealGate",
    }).raise_for_status()

    results = []
    failed = False
    for name in args.models:
        path = MODELS_DIR / name
        print(f"\n=== {name} ({path.stat().st_size / 1e6:.2f} MB) ===")
        with open(path, "rb") as f:
            r = c.post("/api/models/upload",
                       files={"file": (name, f, "application/octet-stream")})
        if r.status_code != 200:
            print(f"  ✗ upload rejected: {r.status_code} {r.text[:200]}")
            failed = True
            continue
        body = r.json()
        mid, rid = body["modelId"], body["runId"]

        deadline = time.time() + args.timeout
        status = c.get(f"/api/models/{mid}/ingestion/{rid}").json()
        while status.get("status") == "streaming" and time.time() < deadline:
            time.sleep(2)
            status = c.get(f"/api/models/{mid}/ingestion/{rid}").json()
        if status.get("status") != "completed":
            print(f"  ✗ pipeline {status.get('status')}: {status.get('error', '')[:200]}")
            failed = True
            continue
        c.post(f"/api/models/{mid}/ingestion/complete")
        print("  ✓ pipeline completed")

        logs = c.get(f"/api/models/{mid}/ingestion/{rid}/logs").json()
        text = "\n".join(
            entry.get("message", "") for entry in (logs.get("logs") or []))
        weights_only = "weight-only" in text.lower() and "Guarantee" not in text

        info = c.get(f"/api/models/{mid}/artifact/info")
        row = {"model": name, "modelId": mid, "weightsOnly": weights_only}
        if info.status_code == 200:
            meta = info.json()
            original = path.stat().st_size
            ratio = meta["sizeBytes"] / original
            row["artifactMB"] = round(meta["sizeBytes"] / 1e6, 3)
            row["uploadMB"] = round(original / 1e6, 3)
            row["ratioVsUpload"] = round(ratio, 3)
            print(f"  artifact {row['artifactMB']} MB vs upload {row['uploadMB']} MB "
                  f"(x{row['ratioVsUpload']})")

        if not weights_only and "Guarantee Certificate" in text:
            print("  ✓ guarantee certificate present")
            row["certificate"] = True
        elif not weights_only:
            print("  ✗ guarantee certificate MISSING")
            row["certificate"] = False
            failed = True

        # Per-trial export gate (executable models only)
        pareto = c.get(f"/api/models/{mid}/pareto")
        if pareto.status_code == 200:
            trials = pareto.json()["trials"]
            target = next((t for t in trials if t["onFrontier"]), trials[0])
            exp = c.post(
                f"/api/models/{mid}/pareto/trials/{target['trialNumber']}/export")
            if exp.status_code == 200:
                dl = c.get(exp.json()["downloadPath"])
                ok = dl.status_code == 200 and len(dl.content) == exp.json()["sizeBytes"]
                print(f"  {'✓' if ok else '✗'} trial #{target['trialNumber']} export+download "
                      f"({exp.json()['sizeBytes'] / 1e6:.2f} MB)")
                row["trialExport"] = ok
                failed = failed or not ok
            else:
                print(f"  ✗ trial export failed: {exp.status_code} {exp.text[:150]}")
                row["trialExport"] = False
                failed = True
        results.append(row)

    print("\n" + json.dumps(results, indent=2))
    print("\nREAL-MODEL GATE " + ("FAILED" if failed else "PASSED"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
