"""First-boot seed — documentation content ONLY.

No demo models, runs, deployments, alerts, activity, API keys or webhooks are
ever fabricated: every operational row in the DB comes from real uploads and
real pipeline events. The only seeded rows are SDK documentation (quick-start
snippets and recipes), which are static docs content, not operational state.
"""

from __future__ import annotations

import json

from sqlmodel import Session, select

from app.dbmodels import RecipeRow, SdkSnippetRow

_SNIPPETS: dict[str, dict] = {
    "python": {
        "language": "python", "filename": "quickstart.py",
        "code": '''# Install
pip install peops

from peops import Client, Budget

client = Client(api_key="peops_sk_live_...")

# Upload a model and walk the Pareto frontier
model = client.models.upload("./ggee-han.onnx", name="ggee-han-v3")
run = client.pareto.run(model.id, budget=Budget(latency_ms=20, size_mb=50))

# Deploy the winning configuration to Seoul edge
deployment = client.deploy(run.best(), target="edge-seoul")''',
    },
    "node": {
        "language": "node", "filename": "quickstart.ts",
        "code": '''// Install
// npm install peops

import { Client, Budget } from "peops";

const client = new Client({ apiKey: "peops_sk_live_..." });

// Upload a model and walk the Pareto frontier
const model = await client.models.upload("./ggee-han.onnx", { name: "ggee-han-v3" });
const run = await client.pareto.run(model.id, {
  budget: new Budget({ latencyMs: 20, sizeMb: 50 }),
});

// Deploy the winning configuration to Seoul edge
const deployment = await client.deploy(run.best(), { target: "edge-seoul" });''',
    },
    "cli": {
        "language": "cli", "filename": "quickstart.sh",
        "code": '''# Install
brew install peops

peops auth login --workspace kwon5700-lab
peops models upload ./ggee-han.onnx --name ggee-han-v3
peops pareto run ggee-han-v3 --latency-ms 20 --size-mb 50
peops deploy --target edge-seoul''',
    },
    "curl": {
        "language": "curl", "filename": "quickstart.sh",
        "code": '''# Upload a model
curl https://api.peops.dev/v1/models/upload \\
  -H "Authorization: Bearer $PEOPS_API_KEY" \\
  -F "file=@./ggee-han.onnx" -F "name=ggee-han-v3"

# Run a Pareto search
curl https://api.peops.dev/v1/pareto/run \\
  -H "Authorization: Bearer $PEOPS_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model_id":"m_...","budget":{"latency_ms":20,"size_mb":50}}\'''',
    },
}


def seed_if_empty(session: Session) -> bool:
    """Insert documentation content on first boot. Returns True when it ran."""
    if session.exec(select(SdkSnippetRow).limit(1)).first() is not None:
        return False

    session.add(RecipeRow(id="r1", title="Sensitivity batch analysis",
                          description="Score 10 models in parallel and export the importance maps.",
                          language="python", steps_json=json.dumps([])))
    session.add(RecipeRow(id="r2", title="Canary deploy with auto-rollback",
                          description="Promote a Pareto candidate gradually based on live telemetry.",
                          language="node", steps_json=json.dumps([])))
    session.add(RecipeRow(id="r3", title="Offline export for CoreML",
                          description="Compile the winning artifact to .mlpackage for on-device inference.",
                          language="cli", steps_json=json.dumps([])))

    for lang, snip in _SNIPPETS.items():
        session.add(SdkSnippetRow(language=lang, filename=snip["filename"], code=snip["code"]))

    session.commit()
    return True
