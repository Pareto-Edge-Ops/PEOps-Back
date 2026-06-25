#!/usr/bin/env bash
# Real, end-to-end proof of the COST & SAVINGS lens:
#
#   boot backend → provision a deployment → inject a multi-accelerator fleet →
#   assert /telemetry/cost shows live per-hardware $/1M with the GPU cheaper than
#   the CPU, the original-vs-compressed counterfactual + monthly bill from the
#   measured QPS, a labeled projection, and a reconciling workspace rollup.
#
# Lighter than verify_hardware_telemetry.sh — it drives traffic through the gated
# fleet simulator rather than a separate `peops serve`, so no wheel/venv is built.
# Self-contained: starts its own backend and tears it down.
# Usage: scripts/verify_cost_savings.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8123
BASE_URL="http://127.0.0.1:${PORT}"
WORK=/tmp/peops-cost-e2e
RUNDIR=/tmp/peops-cost-run

BACKEND_PID=""
cleanup() {
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "── 1. boot backend ($BASE_URL) with sim + inline monitor + fast pipeline"
rm -rf "$RUNDIR" && mkdir -p "$RUNDIR"
( cd "$ROOT" && exec env \
  PEOPS_DB_PATH="$RUNDIR/db.sqlite" PEOPS_STORAGE_DIR="$RUNDIR/storage" \
  PEOPS_WORK_DIR="$RUNDIR/work" PEOPS_FAST_PIPELINE=1 PEOPS_INLINE_JOBS=1 \
  PEOPS_MONITOR_INLINE_ENABLED=1 PEOPS_MONITOR_INTERVAL_SEC=5 \
  PEOPS_TELEMETRY_SIM_ENABLED=1 PEOPS_COOKIE_SECURE=0 PEOPS_RATE_LIMIT_ENABLED=0 \
  python3 -m uvicorn app.main:app --port "$PORT" --log-level warning ) \
  > "$RUNDIR/backend.log" 2>&1 &
BACKEND_PID=$!

for _ in $(seq 1 60); do
  curl -fsS "$BASE_URL/healthz" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -fsS "$BASE_URL/healthz" >/dev/null || { echo "backend never became healthy"; cat "$RUNDIR/backend.log"; exit 1; }
echo "   backend up (pid $BACKEND_PID)"

echo "── 2. provision a deployment (real fast pipeline → benchmark + artifact)"
rm -rf "$WORK" && mkdir -p "$WORK"
python3 "$ROOT/scripts/_sdk_e2e_provision.py" --base "$BASE_URL" --out "$WORK/handoff.json"

echo "── 3. assert the cost & savings lens (live + projection + workspace rollup)"
python3 "$ROOT/scripts/_cost_e2e_assert.py" --base "$BASE_URL" --handoff "$WORK/handoff.json"
