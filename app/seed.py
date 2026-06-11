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
_DOCS_VERSION = "2"

_SNIPPETS: dict[str, dict] = {
    "python": {
        "language": "python", "filename": "quickstart.py",
        "code": '''# Install (local serving needs the [serve] extra)
# pip install 'peops-sdk[serve]'

from peops_sdk import LocalRunner, PeopsClient

BASE_URL = "http://localhost:8080"        # your PEOps origin
DEPLOYMENT_ID = "dep_..."                  # Deployments tab -> deployment id
API_KEY = "peops_sk_live_..."              # shown once when the key is minted

# Option A - hosted inference (telemetry recorded server-side)
client = PeopsClient(BASE_URL, DEPLOYMENT_ID, API_KEY)
out = client.infer({"input": [[0.1, 0.2, 0.3]]})
print(out["latencyMs"], out["outputs"])

# Option B - local serving: pulls the compressed artifact, runs it on YOUR
# hardware, and ships telemetry (latency breakdown, system snapshots,
# input/output stats) back to this dashboard.
runner = LocalRunner.from_deployment(BASE_URL, DEPLOYMENT_ID, API_KEY)
result = runner.run(None)                  # None -> random valid probe
print(result["latencyMs"], result["outputs"])
runner.close()''',
    },
    "node": {
        "language": "node", "filename": "quickstart.ts",
        "code": '''// No npm package needed - the inference endpoint is plain HTTP.
const BASE_URL = "http://localhost:8080";   // your PEOps origin
const DEPLOYMENT_ID = "dep_...";            // Deployments tab -> deployment id
const API_KEY = "peops_sk_live_...";        // shown once at key creation

const res = await fetch(`${BASE_URL}/api/v1/infer/${DEPLOYMENT_ID}`, {
  method: "POST",
  headers: {
    Authorization: `Bearer ${API_KEY}`,
    "Content-Type": "application/json",
  },
  // inputs: { input_name: nested arrays }; null lets the server synthesize
  // a valid random probe (handy for smoke tests).
  body: JSON.stringify({ inputs: null, batch: 1 }),
});
const out = await res.json();
console.log(out.latencyMs, out.outputs);''',
    },
    "cli": {
        "language": "cli", "filename": "quickstart.sh",
        "code": '''# Install the SDK + CLI
pipx install 'peops-sdk[serve]'    # or: pip install 'peops-sdk[serve]'

export PEOPS_BASE_URL=http://localhost:8080
export PEOPS_DEPLOYMENT_ID=dep_...
export PEOPS_API_KEY=peops_sk_live_...

peops pull                          # download the compressed artifact
peops bench -n 200                  # local p50/p95 benchmark (reports telemetry)
peops serve --port 8765             # local HTTP endpoint: POST /infer''',
    },
    "curl": {
        "language": "curl", "filename": "quickstart.sh",
        "code": '''# One real inference against your deployment
curl -s $PEOPS_BASE_URL/api/v1/infer/$PEOPS_DEPLOYMENT_ID \\
  -H "Authorization: Bearer $PEOPS_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"inputs": null, "batch": 1}'

# Pull the deployed (compressed) artifact with the same key
curl -sL $PEOPS_BASE_URL/api/v1/artifacts/$PEOPS_DEPLOYMENT_ID \\
  -H "Authorization: Bearer $PEOPS_API_KEY" \\
  -o model.onnx''',
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
