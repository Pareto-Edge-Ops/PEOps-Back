# PEOps — Backend deployment (independently deployable)

This folder deploys **only the backend**: the FastAPI API, the arq worker, and
their infrastructure — PostgreSQL, Redis (arq queue) and MinIO (S3 object
storage). The frontend is deployed separately from the **PEOps-Front** repo; its
Caddy reverse-proxies `/api` back to this API, so the browser still talks to a
single origin and the httpOnly session cookie works.

```
            (browser → frontend's Caddy, single origin)
                              │  /api/*
                              ▼
┌──────────────┐      ┌──────────────┐
│  API (FastAPI)│────▶ │ PostgreSQL    │
│  cookie auth, │      │ Redis (arq)   │
│  per-user     │────▶ │ S3 / MinIO    │
└──────────────┘      └──────────────┘
        enqueue ▼            ▲ artifacts
      ┌──────────────┐       │
      │ Worker (arq) │───────┘  runs the 6-phase compression pipeline
      │  peops engine│
      └──────────────┘
```

The API/worker image builds from this repo (`PEOps-Back`). The compression
engine (`peops/`) is **vendored into the repo**, so the image is self-contained —
no sibling checkout or copy step is required.

## Deploy

```bash
cd PEOps-Back/deploy

# 1. Configure secrets:
cp .env.example .env
#    edit .env — set PEOPS_JWT_SECRET, POSTGRES_PASSWORD, MINIO_ROOT_PASSWORD,
#    and PEOPS_PUBLIC_ORIGIN (the frontend's public URL — used for CORS).

# 2. Launch:
docker compose up -d --build
```

The API is then reachable at **http://localhost:8000** (override with
`PEOPS_API_PORT`). Smoke-test:

```bash
curl -fsS http://localhost:8000/healthz   # {"status":"ok"}
curl -fsS http://localhost:8000/readyz    # checks DB / Redis / S3
```

Services: `api` · `worker` · `postgres` · `redis` · `minio` (+ `minio-init`).
The API runs `alembic upgrade head` on start.

### Reaching the API from the frontend stack

The frontend's Caddy needs a network path to this API (`BACKEND_UPSTREAM` in
`PEOps-Front/deploy`):

- **Same host, two compose stacks** — set `PEOPS_API_BIND=0.0.0.0` here so the
  API is reachable on the host, then point the frontend at
  `BACKEND_UPSTREAM=host.docker.internal:8000`.
- **Production** — keep the API on a private network (don't bind it to a public
  `0.0.0.0`), front it with TLS, and set `BACKEND_UPSTREAM` to its internal
  address (e.g. `https://api.internal.example.com`).

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
  - `http://localhost:8080/api/auth/google/callback` (local docker)
  - `http://localhost:5173/api/auth/google/callback` (local `pnpm dev`)
  - your real `https://<domain>/api/auth/google/callback` in production
- Set `PEOPS_GOOGLE_REDIRECT_URI` to match the origin you serve, and point it at
  the **frontend** origin (Caddy proxies the callback through to this API).

Accounts are linked by verified email: signing in with Google for an email that
already has a password account logs into the same account.

### Production notes

- **HTTPS**: terminate TLS at the frontend's Caddy (or a load balancer) and set
  `PEOPS_COOKIE_SECURE=1` so the session cookie is Secure.
- **Real object storage**: point `PEOPS_S3_*` at AWS S3 (drop
  `PEOPS_S3_ENDPOINT_URL`, set `PEOPS_S3_FORCE_PATH_STYLE=0`) instead of MinIO.
- **Backups**: persist the `pgdata` and `miniodata` volumes.
- **TensorFlow converters** (`.h5`/`.pb`/`.tflite` → ONNX) are baked into the
  image; uploads in those formats take the real conversion path.

## Configuration

All settings are `PEOPS_*` environment variables (see `../app/config.py` and
`.env.example`). Key ones: `PEOPS_DATABASE_URL`, `PEOPS_REDIS_URL`,
`PEOPS_STORAGE_BACKEND`, `PEOPS_S3_*`, `PEOPS_JWT_SECRET`, `PEOPS_PUBLIC_ORIGIN`,
`PEOPS_COOKIE_SECURE`, `PEOPS_JOB_WORKERS`, `PEOPS_MAX_UPLOAD_MB`,
`PEOPS_RATE_LIMIT_*`.
