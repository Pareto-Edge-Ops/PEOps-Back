"""PEOps SDK — call a deployed, PEOps-compressed model from your app.

    from peops_sdk import PeopsClient

    client = PeopsClient(
        base_url="https://app.example.com",
        deployment_id="dep_ab12cd34ef",
        api_key="peops_sk_live_…",
    )
    out = client.infer({"input": [[0.1, 0.2, ...]]})   # or infer() for a random probe
    print(out["latencyMs"], out["outputs"])

The endpoint is the same one the dashboard's Telemetry tab measures, so every
call your app makes shows up as live telemetry.
"""

from __future__ import annotations

from .client import InferenceError, PeopsClient

__all__ = ["PeopsClient", "InferenceError"]
__version__ = "0.1.0"
