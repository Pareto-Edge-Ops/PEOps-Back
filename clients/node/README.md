# astra-ai-sdk (Node)

Serve **Astra-compressed models** anywhere — and keep the Astra dashboard
monitoring them while they run on your hardware. The Node client mirrors the
[Python `astra-ai-sdk`](../python) API.

```bash
npm i astra-ai-sdk                  # hosted inference client (zero deps)
npm i astra-ai-sdk onnxruntime-node # + local ONNX serving on your hardware
```

Requires **Node ≥ 18.17** (global `fetch`). `onnxruntime-node` is an optional
dependency — install it only if you use `LocalRunner` / `astra serve`.

## Hosted inference

Calls the Astra-hosted endpoint; telemetry is recorded server-side.

```ts
import { AstraClient } from "astra-ai-sdk";

// baseUrl defaults to the hosted Astra origin (override with ASTRA_BASE_URL).
const client = new AstraClient("dep_ab12cd34ef", "astra_sk_live_...");
const out = await client.infer({ input: [[0.1, 0.2, 0.3]] });
console.log(out.latencyMs, out.outputs);
```

## Local serving (the headline)

Pulls the deployed, compressed artifact once (sha256-cached under
`~/.cache/astra`) and runs it with onnxruntime **inside your own code** — no
server to stand up:

```ts
import { LocalRunner } from "astra-ai-sdk";

// baseUrl defaults to the hosted Astra origin (override with ASTRA_BASE_URL).
const runner = await LocalRunner.fromDeployment({
  deploymentId: "dep_ab12cd34ef",
  apiKey: "astra_sk_live_...",
});

// Inputs are name → { data, dims, type? }. type defaults to "float32".
const out = await runner.run({
  input: { data: new Float32Array(1 * 128), dims: [1, 128] },
});
console.log(out.latencyMs, out.outputs);

await runner.close(); // final telemetry flush
```

### Run a file you already have

Downloaded the artifact (SDK Hub → **Download Artifact**) or have an `.onnx` on
disk? Skip the deployment — serve the file directly:

```ts
import { LocalRunner } from "astra-ai-sdk";

const runner = await LocalRunner.fromFile("compressed.onnx");
const out = await runner.run({ data: { data: myFloats, dims: [1, 3, 224, 224] } });
await runner.close();
```

Telemetry is off for a bare file; pass `{ deploymentId, apiKey }` to still report
local runs to that deployment.

### What gets reported to the dashboard

A background reporter batches telemetry to Astra (never blocks or breaks your
serving path; bounded queue with drop-oldest under pressure):

| Stream | Cadence | Fields |
|---|---|---|
| **Request events** | per inference | timestamp, latency breakdown (pre / inference / post ms), success / error code, batch size, region tag, input shape signature |
| **System snapshots** | ~30 s | CPU %, RSS MB, throughput req/min, dropped-event count, SDK / onnxruntime versions, OS, arch, execution provider, hostname |
| **Window stats** | ~60 s or 200 requests | per-input tensor mean/std/min/max/NaN%, output class distribution (top-10), 16-bin confidence histogram, mean entropy, mean top-1 confidence |

Window stats power the dashboard's **prediction drift** and **input
distribution shift** alerts.

Opt out any time: `LocalRunner.fromDeployment({ ..., reportTelemetry: false })`
or `ASTRA_SDK_TELEMETRY=0`.

## CLI

```bash
astra pull  --deployment dep_x --api-key KEY
astra serve --deployment dep_x --api-key KEY --port 8765
astra bench --deployment dep_x --api-key KEY -n 200
```

`astra serve` is an optional zero-code convenience that wraps `LocalRunner` in a
`POST /infer` endpoint on `127.0.0.1`. Options can also come from
`ASTRA_BASE_URL`, `ASTRA_DEPLOYMENT_ID`, `ASTRA_API_KEY`.

## Supported platforms

`onnxruntime-node` ships prebuilt binaries for common platforms (macOS/Linux/
Windows on x64/arm64). On platforms it doesn't cover, `npm i astra-ai-sdk` still
succeeds (it's an optional dependency) — `AstraClient` and `astra pull` work,
and `LocalRunner` raises a clear error until onnxruntime-node is available.
