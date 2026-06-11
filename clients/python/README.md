# peops-sdk

Serve **PEOps-compressed models** anywhere — and keep the PEOps dashboard
monitoring them while they run on your hardware.

```bash
pip install peops-sdk                # hosted inference client
pip install 'peops-sdk[serve]'       # + local ONNX serving (onnxruntime, numpy)
pip install 'peops-sdk[serve,system]'  # + precise CPU/RSS metrics (psutil)
```

## Hosted inference

Calls the PEOps-hosted endpoint; telemetry is recorded server-side.

```python
from peops_sdk import PeopsClient

client = PeopsClient(
    base_url="https://app.example.com",
    deployment_id="dep_ab12cd34ef",
    api_key="peops_sk_live_...",
)
out = client.infer({"input": [[0.1, 0.2, 0.3]]})
print(out["latencyMs"], out["outputs"])
```

## Local serving (the headline)

Pulls the deployed, compressed artifact once (sha256-cached under
`~/.cache/peops`) and serves it with onnxruntime in your process:

```python
from peops_sdk import LocalRunner

runner = LocalRunner.from_deployment(
    base_url="https://app.example.com",
    deployment_id="dep_ab12cd34ef",
    api_key="peops_sk_live_...",
)
out = runner.run({"input": my_numpy_array})   # local inference
print(out["latencyMs"], out["raw"][0].shape)
runner.close()
```

### What gets reported to the dashboard

A background thread batches telemetry to PEOps (never blocks or breaks your
serving path; bounded queue with drop-oldest under pressure):

| Stream | Cadence | Fields |
|---|---|---|
| **Request events** | per inference | timestamp, latency breakdown (preprocess / inference / postprocess ms), success / error code, batch size, region tag, input shape signature |
| **System snapshots** | ~30 s | CPU %, RSS MB, throughput req/min, dropped-event count, SDK / Python / onnxruntime versions, OS, arch, execution provider, hostname |
| **Window stats** | ~60 s or 200 requests | per-input tensor mean/std/min/max/NaN%, output class distribution (top-10), 16-bin confidence histogram, mean entropy, mean top-1 confidence |

Window stats power the dashboard\'s **prediction drift** (PSI vs the
deployment\'s reference distribution) and **input distribution shift** alerts.

Opt out any time: `LocalRunner.from_deployment(..., report_telemetry=False)`
or `PEOPS_SDK_TELEMETRY=0`.

## CLI

```bash
peops pull  --base-url https://app.example.com --deployment dep_x --api-key KEY
peops serve --base-url ... --deployment dep_x --api-key KEY --port 8765
peops bench --base-url ... --deployment dep_x --api-key KEY -n 200
```

Options can also come from `PEOPS_BASE_URL`, `PEOPS_DEPLOYMENT_ID`,
`PEOPS_API_KEY`.
