# peops-sdk

Python client for calling a deployed, **PEOps-compressed** model.

```bash
pip install -e .   # from this directory
```

```python
from peops_sdk import PeopsClient

client = PeopsClient(
    base_url="https://app.example.com",   # your PEOps origin
    deployment_id="dep_ab12cd34ef",        # from the Deployments tab
    api_key="peops_sk_live_…",             # shown once when you deploy
)

# Real inference (inputs: name -> nested list). Pass None for a random probe.
result = client.infer({"input": [[0.1, 0.2, 0.3]]})
print(result["latencyMs"], result["outputs"])
```

Every call is measured by the deployment and shows up live on the model's
**Telemetry Dashboard** (requests/min, p95 latency, error rate, drift alerts).
