# Changelog

Versions are kept in **lockstep** with the Python `astra-ai-sdk` so the dashboard's
single `GET /api/sdk/version` describes both clients truthfully.

## 0.2.0 — 2026-06-29

- Initial Node release — API parity with the Python `astra-ai-sdk` 0.2.0.
- **No base URL in your code**: `baseUrl` is optional everywhere — it defaults to
  the hosted Astra origin (override with `ASTRA_BASE_URL` or the `baseUrl`
  option). SDK code only needs the deployment id + API key. `AstraClient` is now
  `new AstraClient(deploymentId, apiKey, { baseUrl? })`.
- **Hosted inference**: `AstraClient.infer()` against
  `POST /api/v1/infer/{deployment_id}` (zero runtime dependencies — uses the
  global `fetch`).
- **Local serving**: `LocalRunner.fromDeployment()` pulls the deployed,
  Astra-compressed artifact (sha256-cached on disk) and runs it with
  onnxruntime-node **inside your own code** — no server to stand up.
- **Run a local file**: `LocalRunner.fromFile("model.onnx")` serves an artifact
  you already have (e.g. the SDK Hub "Download Artifact" file) — no deployment
  needed; telemetry off unless you pass `{ deploymentId, apiKey }`.
- **Built-in telemetry**: every local inference is measured (latency breakdown,
  batch, input signature) and shipped in fault-tolerant background batches to
  the Astra dashboard, plus periodic system snapshots and windowed input/output
  distribution stats that power prediction/input drift alerts. Opt out with
  `reportTelemetry: false` or `ASTRA_SDK_TELEMETRY=0`.
- **CLI**: `astra pull | serve | bench`.
- Retries with exponential backoff on transient HTTP failures; telemetry can
  never raise into your serving path (drop-oldest queue, `beforeExit` flush).
- `onnxruntime-node` is an optional dependency; ESM-only; ships `.d.ts` types.
