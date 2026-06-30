# astra-sdk

Serve **Astra-compressed models** anywhere — and keep the Astra dashboard
monitoring them while they run on your hardware.

```bash
pip install astra-sdk                # hosted inference client
pip install 'astra-sdk[serve]'       # + local ONNX serving (onnxruntime, numpy)
pip install 'astra-sdk[serve,system]'  # + precise CPU/RSS metrics (psutil)
```

## Hosted inference

Calls the Astra-hosted endpoint; telemetry is recorded server-side.

```python
from astra_sdk import AstraClient

# base_url defaults to the hosted Astra origin (override with ASTRA_BASE_URL).
client = AstraClient("dep_ab12cd34ef", "astra_sk_live_...")
out = client.infer({"input": [[0.1, 0.2, 0.3]]})
print(out["latencyMs"], out["outputs"])
```

## Local serving (the headline)

Pulls the deployed, compressed artifact once (sha256-cached under
`~/.cache/astra`) and serves it with onnxruntime in your process:

```python
from astra_sdk import LocalRunner

# base_url defaults to the hosted Astra origin (override with ASTRA_BASE_URL).
runner = LocalRunner.from_deployment("dep_ab12cd34ef", "astra_sk_live_...")
out = runner.run({"input": my_numpy_array})   # local inference
print(out["latencyMs"], out["raw"][0].shape)
runner.close()
```

### Run a file you already have

Downloaded the artifact (SDK Hub → **Download Artifact**) or have an `.onnx` on
disk? Skip the deployment — serve the file directly:

```python
from astra_sdk import LocalRunner

runner = LocalRunner.from_file("compressed.onnx")
out = runner.run({"input": my_numpy_array})
runner.close()
```

Telemetry is off for a bare file; pass `deployment_id=` + `api_key=` to still
report local runs to that deployment.

### What gets reported to the dashboard

A background thread batches telemetry to Astra (never blocks or breaks your
serving path; bounded queue with drop-oldest under pressure):

| Stream | Cadence | Fields |
|---|---|---|
| **Request events** | per inference | timestamp, latency breakdown (preprocess / inference / postprocess ms), success / error code, batch size, region tag, input shape signature |
| **System snapshots** | ~30 s | CPU %, RSS MB, throughput req/min, dropped-event count, SDK / Python / onnxruntime versions, OS, arch, execution provider, hostname |
| **Window stats** | ~60 s or 200 requests | per-input tensor mean/std/min/max/NaN%, output class distribution (top-10), 16-bin confidence histogram, mean entropy, mean top-1 confidence |

Window stats power the dashboard\'s **prediction drift** (PSI vs the
deployment\'s reference distribution) and **input distribution shift** alerts.

Opt out any time: `LocalRunner.from_deployment(..., report_telemetry=False)`
or `ASTRA_SDK_TELEMETRY=0`.

## CLI

```bash
astra pull  --deployment dep_x --api-key KEY
astra serve --deployment dep_x --api-key KEY --port 8765
astra bench --deployment dep_x --api-key KEY -n 200
```

Options can also come from `ASTRA_BASE_URL`, `ASTRA_DEPLOYMENT_ID`,
`ASTRA_API_KEY`.
