"""PEOps SDK — serve PEOps-compressed models anywhere, with telemetry built in.

Hosted inference (server-side telemetry, zero extra deps):

    from peops_sdk import PeopsClient

    client = PeopsClient(base_url, deployment_id, api_key)
    out = client.infer({"input": [[0.1, 0.2, ...]]})

Local serving (pip install 'peops-sdk[serve]') — pulls the compressed artifact
and runs it on YOUR hardware while the dashboard keeps monitoring it:

    from peops_sdk import LocalRunner

    runner = LocalRunner.from_deployment(base_url, deployment_id, api_key)
    out = runner.run({"input": my_array})
    runner.close()

Every locally-served request ships latency breakdown, system snapshots and
windowed input/output stats to PEOps — powering the live Telemetry tab and
prediction/input drift alerts. Opt out: report_telemetry=False or
PEOPS_SDK_TELEMETRY=0.
"""

from __future__ import annotations

from ._http import ApiError
from .client import InferenceError, PeopsClient
from .runner import LocalRunner, RunnerError, pull_artifact
from .telemetry import TelemetryReporter

__all__ = [
    "ApiError",
    "InferenceError",
    "LocalRunner",
    "PeopsClient",
    "RunnerError",
    "TelemetryReporter",
    "pull_artifact",
]
__version__ = "0.2.0"
