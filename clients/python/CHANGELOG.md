# Changelog

## 0.2.0 — 2026-06-11

- **Local serving**: `LocalRunner.from_deployment()` pulls the deployed,
  PEOps-compressed artifact (sha256/ETag-cached on disk) and serves it with
  onnxruntime on your own hardware (`pip install 'peops-sdk[serve]'`).
- **Built-in telemetry**: every local inference is measured (latency
  breakdown, batch, input signature) and shipped in fault-tolerant background
  batches to the PEOps dashboard — plus periodic system snapshots (CPU/RSS/
  throughput/runtime fingerprint) and windowed input/output distribution
  stats that power prediction/input drift alerts. Opt out with
  `report_telemetry=False` or `PEOPS_SDK_TELEMETRY=0`.
- **CLI**: `peops pull | serve | bench`.
- Retries with exponential backoff on transient HTTP failures; telemetry can
  never raise into your serving path (drop-oldest queue, atexit flush).
- PEP 561 (`py.typed`), MIT license, full PyPI metadata.

## 0.1.0

- Initial release: `PeopsClient.infer()` against the hosted
  `POST /api/v1/infer/{deployment_id}` endpoint.
