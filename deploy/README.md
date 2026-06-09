# PEOps — Deployment (single-URL Docker stack)

This folder holds the deployment glue for the full PEOps product: a single
`docker compose` that serves the SPA and the API from one origin (Caddy on
:8080), backed by PostgreSQL, Redis (arq queue) and MinIO (S3 object storage).

```
┌────────────┐      ┌──────────────┐      ┌──────────────┐
│  Caddy     │────▶ │  API (FastAPI)│────▶ │ PostgreSQL    │
│  (SPA +    │      │  cookie auth, │      │ Redis (arq)   │
│  /api proxy)│◀──── │  per-user     │────▶ │ S3 / MinIO    │
└────────────┘      └──────────────┘      └──────────────┘
        single origin          enqueue ▼         ▲ artifacts
                              ┌──────────────┐    │
                              │ Worker (arq) │────┘  runs the 6-phase
                              │  peops engine│       compression pipeline
                              └──────────────┘
```

- **API / Worker** build from this repo (`PEOps-Back`). The compression engine
  (`peops/`) is **vendored into the repo**, so no extra copy step is needed.
- **Caddy** builds the frontend from a **sibling checkout of `PEOps-Front`**
  (i.e. `PEOps-Front` lives next to `PEOps-Back`):

  ```
  <workspace>/
  ├── PEOps-Back/    ← you are here (PEOps-Back/deploy)
  └── PEOps-Front/   ← required for the `caddy` build
  ```

## Deploy

```bash
cd PEOps-Back/deploy

# 1. Configure secrets:
cp .env.example .env
#    edit .env — set PEOPS_JWT_SECRET, POSTGRES_PASSWORD, MINIO_ROOT_PASSWORD

# 2. Launch:
docker compose up -d --build
```

Open **http://localhost:8080**, sign up, and upload a model.

Services: `caddy` (single origin) · `api` · `worker` · `postgres` · `redis` ·
`minio` (+ `minio-init`). The API runs `alembic upgrade head` on start.

### Scaling

```bash
docker compose up -d --scale api=3 --scale worker=3
```

Job state lives in Postgres and the queue in Redis, so any API replica serves
progress for a job any worker is running. Run multiple `api`/`worker` containers
to scale out — **do not** add uvicorn `--workers N` to a single container (the
in-process job registry assumes one process per container). Concurrency per
worker is capped by `PEOPS_JOB_WORKERS`.

### Google sign-in (optional)

Users can sign in with Google in addition to email/password. Enable it by setting
`PEOPS_GOOGLE_CLIENT_ID` / `PEOPS_GOOGLE_CLIENT_SECRET` in `.env` (leave blank to
disable — the Google button auto-hides).

In Google Cloud Console → **APIs & Services → Credentials → OAuth client ID (Web)**:
- **Authorized redirect URIs** must include the callback exactly:
  - `http://localhost:8080/api/auth/google/callback` (docker single-origin)
  - `http://localhost:5173/api/auth/google/callback` (local `pnpm dev`)
  - your real `https://<domain>/api/auth/google/callback` in production
- Set `PEOPS_GOOGLE_REDIRECT_URI` to match the origin you serve.

Accounts are linked by verified email: signing in with Google for an email that
already has a password account logs into the same account.

### Production notes

- **HTTPS**: put this behind a TLS terminator (or give Caddy a domain) and set
  `PEOPS_COOKIE_SECURE=1` so the session cookie is Secure.
- **Real object storage**: point `PEOPS_S3_*` at AWS S3 (drop
  `PEOPS_S3_ENDPOINT_URL`, set `PEOPS_S3_FORCE_PATH_STYLE=0`) instead of MinIO.
- **Backups**: persist the `pgdata` and `miniodata` volumes.
- **TensorFlow converters** (`.h5`/`.pb`/`.tflite` → ONNX) are optional; without
  TensorFlow installed those uploads fall back to honest weight-only analysis.
- **Password reset by email** is out of scope for v1; signed-in users can change
  their password in Settings.

## Configuration

All settings are `PEOPS_*` environment variables (see `../app/config.py` and
`.env.example`). Key ones: `PEOPS_DATABASE_URL`, `PEOPS_REDIS_URL`,
`PEOPS_STORAGE_BACKEND`, `PEOPS_S3_*`, `PEOPS_JWT_SECRET`, `PEOPS_PUBLIC_ORIGIN`,
`PEOPS_COOKIE_SECURE`, `PEOPS_JOB_WORKERS`, `PEOPS_MAX_UPLOAD_MB`,
`PEOPS_RATE_LIMIT_*`.
