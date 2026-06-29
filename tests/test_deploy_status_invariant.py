"""The deploy-status invariant: a model with a live deployment reads "deployed",
and that is a STABLE terminal state — the only way out is deleting the model's
last deployment (→ "draft"). Locks both the reconcile self-heal helper and the
create/pause/resume/delete lifecycle so the badge can never silently regress."""

from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.dbmodels import ModelRow
from app.repositories import reconcile_deploy_status


def _model(mid: str, status: str, is_deployed: bool) -> ModelRow:
    return ModelRow(
        id=mid, user_id="u1", name=mid, type_full="CNN", type_short="CNN",
        format="ONNX", last_learned_at="2026-01-01T00:00:00Z",
        status=status, is_deployed=is_deployed,
    )


def test_reconcile_only_flips_deployed_drafts():
    """reconcile_deploy_status flips ONLY is_deployed+draft rows, leaving
    already-deployed and in-flight/failed rows untouched, and is idempotent."""
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        s.add_all([
            _model("m_stale", "draft", True),          # the bug case → must flip
            _model("m_ok", "deployed", True),          # already correct → untouched
            _model("m_draft", "draft", False),         # no deployment → untouched
            _model("m_analyzing", "analyzing", True),  # in-flight → NOT clobbered
            _model("m_optimizing", "optimizing", True),
            _model("m_failed", "failed", True),
        ])
        s.commit()

        assert reconcile_deploy_status(s) == ["m_stale"]

        got = {m.id: m.status for m in s.exec(select(ModelRow)).all()}
        assert got == {
            "m_stale": "deployed",       # flipped
            "m_ok": "deployed",          # unchanged
            "m_draft": "draft",          # not deployed → untouched
            "m_analyzing": "analyzing",  # in-flight → not clobbered
            "m_optimizing": "optimizing",
            "m_failed": "failed",
        }
        # Idempotent: a second pass changes nothing.
        assert reconcile_deploy_status(s) == []


def test_deployed_status_stable_through_pause_resume(make_live_model, deploy_model, client):
    """Pausing/resuming a deployment never changes the model's "deployed" badge —
    the model still HAS a deployment."""
    mid = make_live_model("invariant-a.onnx")["modelId"]
    dep_id, _ = deploy_model(mid)
    assert client.get(f"/api/models/{mid}").json()["status"] == "deployed"

    assert client.post(f"/api/deployments/{dep_id}/pause").json()["status"] == "paused"
    assert client.get(f"/api/models/{mid}").json()["status"] == "deployed"
    assert client.post(f"/api/deployments/{dep_id}/resume").json()["status"] == "live"
    assert client.get(f"/api/models/{mid}").json()["status"] == "deployed"


def test_deployed_reverts_only_when_last_deployment_deleted(make_live_model, deploy_model, client):
    """With two deployments, deleting one keeps "deployed"; deleting the LAST one
    reverts to "draft". This is the single sanctioned exit from "deployed"."""
    mid = make_live_model("invariant-b.onnx")["modelId"]
    dep1, _ = deploy_model(mid)
    dep2, _ = deploy_model(mid)
    assert client.get(f"/api/models/{mid}").json()["status"] == "deployed"

    # delete one of two → still deployed
    assert client.delete(f"/api/deployments/{dep1}").status_code == 200
    m = client.get(f"/api/models/{mid}").json()
    assert m["status"] == "deployed" and m["isDeployed"] is True

    # delete the last → reverts to draft (the only exit)
    assert client.delete(f"/api/deployments/{dep2}").status_code == 200
    m = client.get(f"/api/models/{mid}").json()
    assert m["status"] == "draft" and m["isDeployed"] is False


def _list_item(client, mid: str) -> dict:
    """The model's row in the list endpoint — the badge's real data source."""
    return next(m for m in client.get("/api/models").json() if m["id"] == mid)


def test_isserving_tracks_pause_resume_without_touching_status(
    make_live_model, deploy_model, client,
):
    """`isServing` flips with pause/resume while `status` stays "deployed": the
    model still HAS a deployment (status), it just isn't routing traffic
    (isServing). Both the single-GET and the list endpoints agree — this is what
    lets the badge read "Deployed · paused" without regressing the lifecycle."""
    mid = make_live_model("serving-a.onnx")["modelId"]

    # not deployed yet → not serving
    before = client.get(f"/api/models/{mid}").json()
    assert before["isDeployed"] is False and before["isServing"] is False

    dep_id, _ = deploy_model(mid)
    live = client.get(f"/api/models/{mid}").json()
    assert live["status"] == "deployed" and live["isServing"] is True
    assert _list_item(client, mid)["isServing"] is True

    # pause → STILL deployed, but no longer serving (the fix)
    assert client.post(f"/api/deployments/{dep_id}/pause").json()["status"] == "paused"
    paused = client.get(f"/api/models/{mid}").json()
    assert paused["status"] == "deployed" and paused["isDeployed"] is True
    assert paused["isServing"] is False
    assert _list_item(client, mid)["isServing"] is False

    # resume → serving again
    assert client.post(f"/api/deployments/{dep_id}/resume").json()["status"] == "live"
    assert client.get(f"/api/models/{mid}").json()["isServing"] is True


def test_isserving_true_when_any_deployment_live(make_live_model, deploy_model, client):
    """With two deployments the model serves while ≥1 is live; pausing BOTH makes
    it deployed-but-not-serving."""
    mid = make_live_model("serving-b.onnx")["modelId"]
    dep1, _ = deploy_model(mid)
    dep2, _ = deploy_model(mid)

    # pause one of two → still serving via the other
    client.post(f"/api/deployments/{dep1}/pause")
    m = client.get(f"/api/models/{mid}").json()
    assert m["status"] == "deployed" and m["isServing"] is True

    # pause the last live one → deployed, not serving
    client.post(f"/api/deployments/{dep2}/pause")
    m = client.get(f"/api/models/{mid}").json()
    assert m["status"] == "deployed" and m["isDeployed"] is True
    assert m["isServing"] is False


def test_canary_counts_as_serving(make_live_model, client):
    """A canary deployment routes (a fraction of) traffic → isServing True."""
    mid = make_live_model("serving-c.onnx")["modelId"]
    r = client.post(f"/api/models/{mid}/deployments", json={"status": "canary"})
    assert r.status_code == 200, r.text
    assert r.json()["deployment"]["status"] == "canary"
    assert client.get(f"/api/models/{mid}").json()["isServing"] is True


def test_weights_only_model_cannot_deploy(statedict_model, client):
    """A weights-only (.npz) model has no executable artifact → deploy is
    rejected (409 not_servable) and the model stays "draft": it can be
    compressed but not served, so its badge never becomes "deployed"."""
    mid = statedict_model["modelId"]
    assert client.get(f"/api/models/{mid}").json()["status"] == "draft"

    r = client.post(f"/api/models/{mid}/deployments", json={})
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "not_servable"

    # status unchanged — still draft, never flipped to deployed
    m = client.get(f"/api/models/{mid}").json()
    assert m["status"] == "draft" and m["isDeployed"] is False
