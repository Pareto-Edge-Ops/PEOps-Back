"""Astra SDK — serve Astra-compressed models anywhere, with telemetry built in.

Hosted inference (server-side telemetry, zero extra deps):

    from astra_sdk import AstraClient

    client = AstraClient(deployment_id, api_key)   # base_url defaults to the hosted origin
    out = client.infer({"input": [[0.1, 0.2, ...]]})

Local serving (pip install 'astra-sdk[serve]') — pulls the compressed artifact
and runs it on YOUR hardware while the dashboard keeps monitoring it:

    from astra_sdk import LocalRunner

    runner = LocalRunner.from_deployment(deployment_id, api_key)
    out = runner.run({"input": my_array})
    runner.close()

Every locally-served request ships latency breakdown, system snapshots and
windowed input/output stats to Astra — powering the live Telemetry tab and
prediction/input drift alerts. Opt out: report_telemetry=False or
ASTRA_SDK_TELEMETRY=0.
"""

from __future__ import annotations

from ._http import ApiError
from .client import InferenceError, AstraClient
from .runner import LocalRunner, RunnerError, pull_artifact
from .telemetry import TelemetryReporter

__all__ = [
    "ApiError",
    "InferenceError",
    "LocalRunner",
    "AstraClient",
    "RunnerError",
    "TelemetryReporter",
    "pull_artifact",
]
__version__ = "0.2.0"
