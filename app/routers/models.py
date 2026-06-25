"""Models registry: list/get/import/upload + the ingestion/complete handshake."""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import text
from sqlmodel import Session

from app.auth.dependencies import CurrentUser
from app.config import get_settings, iso
from app.db import get_session
from app.dbmodels import (
    ActivityRow,
    AlertRow,
    DeploymentRow,
    IngestionLogRow,
    IngestionRunRow,
    ModelRow,
    ResultCacheRow,
    RunRow,
)
from app.repositories import get_cached_result, list_models, model_row_to_item, owned_model
from app.schemas.common import OkResponse
from app.schemas.models import ImportRequest, ImportResponse, RenameRequest
from app.services.cancel import request_cancel
from app.services.family import family_from_file_name
from app.services.formats import display_name, infer_format
from app.services.limits import enforce_size, limiter, validate_upload
from app.services.queue import enqueue_pipeline
from app.services.storage import (
    StorageError,
    artifact_prefix,
    get_storage,
)
from app.services.storage import (
    source_key as make_source_key,
)

router = APIRouter(prefix="/models", tags=["models"])


def _now() -> str:
    return iso(datetime.now(timezone.utc))


@router.get("")
def models_list(
    current_user: CurrentUser,
    q: str | None = Query(default=None),
    onlyDeployed: str | None = Query(default=None),
    sort: str | None = Query(default=None),
    session: Session = Depends(get_session),
) -> JSONResponse:
    items = list_models(session, q, onlyDeployed == "1", sort, user_id=current_user.id)
    return JSONResponse([m.to_response() for m in items])


@router.get("/{model_id}")
def model_get(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> JSONResponse:
    model = owned_model(session, model_id, current_user.id)
    return JSONResponse(model_row_to_item(model).to_response())


async def _start_ingestion(
    session: Session,
    *,
    user_id: str,
    file_name: str,
    source_key: str | None,
) -> ImportResponse:
    token = uuid.uuid4().hex[:8]
    run_id = f"ing_{token}"
    model_id = f"m_uploaded_{token}"
    fmt, type_full, type_short = infer_format(file_name)
    settings = get_settings()
    now = _now()

    session.add(ModelRow(
        id=model_id,
        user_id=user_id,
        name=display_name(file_name),
        type_full=type_full,
        type_short=type_short,
        format=fmt,
        last_learned_at=now,
        last_optimized_at=None,
        status="analyzing",
        best_accuracy=None,
        is_deployed=False,
        description=f"Uploaded {file_name} · awaiting sensitivity analysis",
        analysis_run_id=run_id,
        family=family_from_file_name(file_name, fmt),
        source="pipeline",
        source_key=source_key,
    ))
    session.add(IngestionRunRow(
        id=run_id, user_id=user_id, model_id=model_id, file_name=file_name,
        started_at=now, status="streaming", progress=0,
    ))
    n_trials = 4 if settings.fast_pipeline else settings.pareto_trials
    session.add(RunRow(
        # Honest initial state: the job is enqueued, not yet picked up. The worker
        # flips this to "running" when it actually starts processing (see
        # jobs.py). This makes the dashboard "Queued" tab + activeRuns KPI real
        # instead of every run appearing "running" the instant it's created.
        id=f"run_{run_id}", user_id=user_id, model_id=model_id, name=display_name(file_name),
        status="queued", progress_pct=0, iter=f"0 / {n_trials}",
        best_acc=0, delta_acc=0, created_at=now,
    ))
    session.add(ActivityRow(
        id=f"act_up_{token}", user_id=user_id, kind="model_uploaded",
        text=f"Model uploaded — {file_name}", timestamp=now,
    ))
    session.add(ActivityRow(
        id=f"act_start_{token}", user_id=user_id, kind="run_started",
        text=f"Run started — run_{run_id} · {display_name(file_name)}", timestamp=now,
    ))
    session.commit()

    await enqueue_pipeline(
        run_id=run_id,
        model_id=model_id,
        user_id=user_id,
        model_name=display_name(file_name),
        file_name=file_name,
        source_key=source_key,
        input_shape=None,
        declared_format=fmt,
    )
    return ImportResponse(runId=run_id, modelId=model_id, fileName=file_name)


@router.post("/import")
@limiter.limit(get_settings().rate_limit_import)
async def models_import(
    request: Request,
    current_user: CurrentUser,
    body: ImportRequest | None = None,
    session: Session = Depends(get_session),
) -> ImportResponse:
    file_name = (body.fileName if body and body.fileName else "uploaded-model.onnx")
    # No bytes in this flow — the job synthesizes a real model for the format.
    return await _start_ingestion(
        session, user_id=current_user.id, file_name=file_name, source_key=None,
    )


@router.post("/upload")
@limiter.limit(get_settings().rate_limit_upload)
async def models_upload(
    request: Request,
    file: UploadFile,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> ImportResponse:
    """Real multipart upload — stores the source in object storage; the worker
    pulls it down to run the pipeline on the actual bytes."""
    validate_upload(file)
    file_name = file.filename or "uploaded-model.onnx"
    key = make_source_key(uuid.uuid4().hex[:8], file_name)
    # Stage to a temp file (counting bytes for the size cap), then hand off to
    # the storage backend (local or S3). The temp file is always cleaned up,
    # even if the size cap trips mid-stream.
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    total = 0
    try:
        while chunk := await file.read(1 << 20):
            total += len(chunk)
            enforce_size(total)
            tmp.write(chunk)
        tmp.close()
        get_storage().upload_file(tmp_path, key)
    finally:
        tmp.close()
        Path(tmp_path).unlink(missing_ok=True)
    return await _start_ingestion(
        session, user_id=current_user.id, file_name=file_name, source_key=key,
    )


@router.post("/{model_id}/ingestion/complete")
def ingestion_complete(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> OkResponse:
    model = owned_model(session, model_id, current_user.id)
    if model.status != "analyzing":
        return OkResponse()  # idempotent — mirrors the front mock

    # Single guarded UPDATE so a worker finishing concurrently can't be
    # overwritten (lost-update): "optimizing" is only entered while the run is
    # still streaming, atomically with that check. The worker's final commit
    # (run→completed + model→draft) flips "optimizing" → "draft" afterwards.
    if model.analysis_run_id:
        result = session.execute(
            text(
                "UPDATE models SET status = 'optimizing' "
                "WHERE id = :mid AND status = 'analyzing' "
                "AND (SELECT status FROM ingestion_runs WHERE id = :rid) = 'streaming'"
            ),
            {"mid": model_id, "rid": model.analysis_run_id},
        )
        session.commit()
        if result.rowcount:
            return OkResponse()  # worker finalizes to draft when it completes

    session.refresh(model)
    if model.status != "analyzing":
        return OkResponse()  # the worker finalized in the meantime

    run = session.get(IngestionRunRow, model.analysis_run_id) if model.analysis_run_id else None
    if run is not None and run.status == "failed":
        model.status = "failed"
        model.analysis_run_id = None
        session.add(model)
        session.commit()
        return OkResponse()

    # Run finished (or there was never a run) but the model wasn't finalized —
    # complete it with the REAL result when one exists; otherwise accuracy
    # honestly stays unset (it was never measured).
    cached = get_cached_result(session, model_id, "pareto", user_id=current_user.id)
    model.status = "draft"
    if cached and cached.get("trials"):
        model.best_accuracy = round(max(t["accuracy"] for t in cached["trials"]), 1)
    model.description = (
        f"{model.description or ''} · sensitivity analysis complete"
    ).strip(" ·")
    model.analysis_run_id = None
    model.last_optimized_at = None
    session.add(model)
    session.commit()
    return OkResponse()


@router.patch("/{model_id}")
def model_rename(
    model_id: str,
    body: RenameRequest,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> JSONResponse:
    model = owned_model(session, model_id, current_user.id)
    name = body.name.strip()
    if not 1 <= len(name) <= 80:
        raise HTTPException(status_code=400, detail="name must be 1–80 characters")
    model.name = name
    session.add(model)
    # Dashboard run rows carry a display-name copy — keep them consistent.
    from sqlmodel import select

    for run in session.exec(select(RunRow).where(RunRow.model_id == model_id)).all():
        run.name = name
        session.add(run)
    session.commit()
    return JSONResponse(model_row_to_item(model).to_response())


@router.delete("/{model_id}")
def model_delete(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> OkResponse:
    model = owned_model(session, model_id, current_user.id)
    # Stop a live pipeline first so the worker can't resurrect rows mid-delete.
    if model.analysis_run_id:
        request_cancel(model.analysis_run_id)

    from sqlmodel import delete as sql_delete
    from sqlmodel import select

    run_ids = [
        r.id for r in session.exec(
            select(IngestionRunRow).where(IngestionRunRow.model_id == model_id)
        ).all()
    ]
    if run_ids:
        session.execute(sql_delete(IngestionLogRow).where(IngestionLogRow.run_id.in_(run_ids)))  # type: ignore[attr-defined]
    for table in (IngestionRunRow, RunRow, ResultCacheRow, AlertRow, DeploymentRow):
        session.execute(sql_delete(table).where(table.model_id == model_id))  # type: ignore[attr-defined]
    source_key_to_drop = model.source_key
    session.delete(model)
    session.commit()

    # Compressed artifact + uploaded source go from object storage too.
    storage = get_storage()
    storage.delete_prefix(artifact_prefix(model_id))
    if source_key_to_drop:
        storage.delete(source_key_to_drop)
    return OkResponse()


@router.get("/{model_id}/artifact/info")
def model_artifact_info(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> dict:
    """REAL metadata of the compressed artifact — size, checksum, IO spec."""
    import hashlib

    model = owned_model(session, model_id, current_user.id)
    key = model.artifact_key
    if not key or not get_storage().exists(key):
        raise HTTPException(status_code=404, detail={
            "code": "no_artifact",
            "message": "No compressed artifact yet — it is produced when the "
                       "ingestion pipeline completes.",
        })
    data = get_storage().read_bytes(key)
    name = Path(key).name
    suffix = Path(key).suffix
    info: dict = {
        "fileName": name,
        "sizeBytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "kind": "onnx" if suffix == ".onnx" else "npz",
        "downloadPath": f"/api/models/{model_id}/artifact",
    }
    if suffix == ".onnx":
        try:
            import onnx

            m = onnx.load_from_string(data)
            initializers = {init.name for init in m.graph.initializer}
            info["inputs"] = [
                {
                    "name": i.name,
                    "shape": [d.dim_value if d.dim_value > 0 else 1
                              for d in i.type.tensor_type.shape.dim],
                    "dtype": onnx.TensorProto.DataType.Name(
                        i.type.tensor_type.elem_type
                    ).lower(),
                }
                for i in m.graph.input if i.name not in initializers
            ]
        except Exception:  # noqa: BLE001 — IO spec is best-effort metadata
            pass
    # Provenance lets the SDK Hub label exactly which candidate this artifact is
    # (Pareto pick / guarantee-ladder rung) so its size reconciles with the
    # Pareto Studio. Absent for models optimized before this was recorded.
    meta = get_cached_result(session, model_id, "artifact_meta", user_id=current_user.id)
    if meta:
        info["provenance"] = meta
    return info


@router.get("/{model_id}/sdk/usage")
def model_sdk_usage(
    model_id: str,
    request: Request,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> dict:
    """Per-model, RUNNABLE usage snippets for the real compressed artifact."""
    info = model_artifact_info(model_id, current_user, session)
    # Prefer an explicitly configured public origin; otherwise derive it from the
    # incoming request so copy-pasted snippets point at the host the user is on.
    base = (get_settings().public_origin or str(request.base_url)).rstrip("/")
    download = f"{base}{info['downloadPath']}"
    if info["kind"] == "onnx":
        inp = (info.get("inputs") or [{}])[0]
        shape = inp.get("shape", [1])
        name = inp.get("name", "input")
        python = f'''# Run the PEOps-compressed model with onnxruntime
# pip install onnxruntime numpy requests
import numpy as np
import onnxruntime as ort
import requests

# 1. Download the compressed artifact ({info["sizeBytes"] / 1e6:.2f} MB)
data = requests.get("{download}").content
open("{info["fileName"]}", "wb").write(data)

# 2. Real inference
session = ort.InferenceSession("{info["fileName"]}")
x = np.random.rand(*{shape}).astype(np.float32)   # input "{name}" {shape}
outputs = session.run(None, {{"{name}": x}})
print(outputs[0].shape)'''
    else:
        python = f'''# Load the PEOps weight-only compressed checkpoint (.npz)
# Weights are INT8-quantized; each "<key>.__scale__" entry restores FP32:
#     w_fp32 = w_int8.astype(np.float32) * scale
import numpy as np
import requests

data = requests.get("{download}").content
open("{info["fileName"]}", "wb").write(data)

archive = np.load("{info["fileName"]}")
for key in archive.files:
    if key.endswith(".__scale__"):
        continue
    w = archive[key]
    scale_key = key + ".__scale__"
    if scale_key in archive.files:
        w = w.astype(np.float32) * float(archive[scale_key])
    print(key, w.shape, w.dtype)'''
    curl = f'''# Download the compressed artifact and verify its real checksum
curl -L -o {info["fileName"]} {download}
echo "{info["sha256"]}  {info["fileName"]}" | shasum -a 256 -c -'''
    # The /sdk/usage contract is a flat record of language-keyed snippets (the SDK
    # Hub renders each key as a code tab). The LIVE inference snippet + deploy CTA
    # are built on the Deployments tab from the deployment's real endpoint, so
    # this stays {python, curl} and the artifact-download story is unchanged.
    return {
        "python": {"language": "python", "filename": "use_compressed.py", "code": python},
        "curl": {"language": "curl", "filename": "download.sh", "code": curl},
    }


@router.get("/{model_id}/artifact")
def model_artifact(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    model = owned_model(session, model_id, current_user.id)
    # ONNX for executable models; quantized tensor archive for weights-only ones.
    key = model.artifact_key
    if not key:
        raise HTTPException(status_code=404, detail="no compressed artifact for this model")
    try:
        stream, size = get_storage().open_stream(key)
    except StorageError:
        raise HTTPException(
            status_code=404, detail="no compressed artifact for this model"
        ) from None
    name = Path(key).name
    return StreamingResponse(
        stream,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Content-Length": str(size),
        },
    )
