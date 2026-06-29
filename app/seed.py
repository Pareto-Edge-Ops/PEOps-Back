"""First-boot seed — documentation content ONLY.

No demo models, runs, deployments, alerts, activity, API keys or webhooks are
ever fabricated: every operational row in the DB comes from real uploads and
real pipeline events. The only seeded rows are SDK documentation (quick-start
snippets and recipes), which are static docs content, not operational state.

Every snippet below targets APIs that actually exist: the peops-sdk pip
package (PeopsClient / LocalRunner / `peops` CLI) and the real REST endpoints
(/api/v1/infer, /api/v1/artifacts, /api/v1/telemetry). A version marker row
lets upgrades replace stale docs rows in already-seeded databases.
"""

from __future__ import annotations

import json

from sqlmodel import Session, select

from app.dbmodels import RecipeRow, SdkSnippetRow

# Bump when snippets/recipes change — seeded DBs are upgraded in place.
_DOCS_VERSION = "5"

# Copy-paste "deploy the compressed model on YOUR server" guide. Every snippet
# self-hosts a `POST /infer` endpoint on the user's own hardware (not a call to
# the hosted API), using only APIs that exist: the peops-sdk LocalRunner / `peops
# serve` CLI, and onnxruntime-node for the Node path.
_SNIPPETS: dict[str, dict] = {
    "python": {
        "language": "python", "filename": "serve.py",
        "code": '''# Serve the compressed model as an HTTP endpoint on your own server.
# pip install 'peops-sdk[serve]' fastapi uvicorn
from fastapi import FastAPI
from peops_sdk import LocalRunner

BASE_URL = "__PEOPS_ORIGIN__"
DEPLOYMENT_ID = "dep_..."                 # Deployments tab -> deployment id
API_KEY = "peops_sk_live_..."            # shown once when the key is minted

# Pulls + caches the compressed artifact, runs onnxruntime locally, and ships
# latency / system / drift telemetry back to this dashboard.
runner = LocalRunner.from_deployment(BASE_URL, DEPLOYMENT_ID, API_KEY)

app = FastAPI()

@app.post("/infer")
def infer(payload: dict):
    out = runner.run(payload.get("inputs"))   # {"inputs": null} -> random probe
    return {"latencyMs": out["latencyMs"], "outputs": out["outputs"]}

# Run it:  uvicorn serve:app --host 0.0.0.0 --port 8765
# Zero-code alternative:  peops serve --port 8765   (see the CLI tab)''',
    },
    "node": {
        "language": "node", "filename": "serve.ts",
        "code": '''// Self-host the compressed ONNX from a Node server (no Node SDK needed).
// 1) get model.onnx: run `peops pull` (CLI tab) or the Download Artifact panel
// 2) npm i express onnxruntime-node
import express from "express";
import * as ort from "onnxruntime-node";

const session = await ort.InferenceSession.create("model.onnx");
const app = express();
app.use(express.json());

app.post("/infer", async (req, res) => {
  // body: { <input_name>: { data: number[], dims: number[] } } per model input
  const feeds = Object.fromEntries(
    Object.entries(req.body).map(([name, t]: any) =>
      [name, new ort.Tensor("float32", Float32Array.from(t.data), t.dims)]),
  );
  const out = await session.run(feeds);
  res.json(Object.fromEntries(
    Object.entries(out).map(([name, t]) => [name, { dims: t.dims }])));
});

app.listen(8765, () => console.log("POST /infer live on http://localhost:8765"));''',
    },
    "cli": {
        "language": "cli", "filename": "serve.sh",
        "code": '''# One command turns the compressed model into a live /infer endpoint.
pip install 'peops-sdk[serve]'      # or: pipx install 'peops-sdk[serve]'

export PEOPS_BASE_URL=__PEOPS_ORIGIN__
export PEOPS_DEPLOYMENT_ID=dep_...         # Deployments tab -> deployment id
export PEOPS_API_KEY=peops_sk_live_...     # shown once when the key is minted

peops pull                          # cache the compressed artifact
peops serve --port 8765             # POST /infer now live at http://127.0.0.1:8765
# Binds localhost — front it with your reverse proxy / container to expose it.''',
    },
}

_RECIPES: list[dict] = [
    {
        "id": "r1", "language": "python",
        "title": "Deploy + mint an API key",
        "description": "Import a model, wait for optimization, create a "
                       "deployment on the Deployments tab and copy the key — "
                       "it authenticates /api/v1/infer, /artifacts and "
                       "/telemetry for that deployment.",
    },
    {
        "id": "r2", "language": "python",
        "title": "Local serving with live drift alerts",
        "description": "LocalRunner.from_deployment() serves the compressed "
                       "ONNX on your hardware while shipping latency, system "
                       "and input/output stats — prediction/input drift "
                       "alerts appear on the Telemetry tab automatically.",
    },
    {
        "id": "r3", "language": "cli",
        "title": "Download any Pareto trial as ONNX",
        "description": "Pick a point in Pareto 3D Search and Export ONNX — or "
                       "POST /api/models/{id}/pareto/trials/{n}/export and "
                       "fetch the artifact for the exact accuracy/size/latency "
                       "trade-off you need.",
    },
]


def seed_if_empty(session: Session) -> bool:
    """Insert documentation content on first boot, and upgrade stale docs rows
    in place when _DOCS_VERSION changes. Returns True when anything ran."""
    marker = session.exec(
        select(SdkSnippetRow).where(SdkSnippetRow.language == "_meta")
    ).first()
    if marker is not None and marker.filename == _DOCS_VERSION:
        return False

    # Replace docs rows wholesale (docs only — never operational state).
    for row in session.exec(select(SdkSnippetRow)).all():
        session.delete(row)
    for row in session.exec(select(RecipeRow)).all():
        session.delete(row)

    for recipe in _RECIPES:
        session.add(RecipeRow(
            id=recipe["id"], title=recipe["title"],
            description=recipe["description"],
            language=recipe["language"], steps_json=json.dumps([]),
        ))
    for lang, snip in _SNIPPETS.items():
        session.add(SdkSnippetRow(
            language=lang, filename=snip["filename"], code=snip["code"]))
    session.add(SdkSnippetRow(language="_meta", filename=_DOCS_VERSION, code=""))

    session.commit()
    return True
