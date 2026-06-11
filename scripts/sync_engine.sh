#!/usr/bin/env bash
# Sync the peops engine from its source of truth (the PEOps-PoC research repo)
# into the two vendored copies. All engine edits land in the PoC first; this
# script is the only sanctioned way bytes flow into the backend.
set -euo pipefail

SRC="${PEOPS_POC_DIR:-$HOME/Desktop/PEOps-PoC}/peops"
BACK="$(cd "$(dirname "$0")/.." && pwd)/peops"
MINI="$(cd "$(dirname "$0")/../.." && pwd)/PEOps-PoC/peops"

if [ ! -d "$SRC" ]; then
  echo "source engine not found: $SRC" >&2
  exit 1
fi

for DST in "$BACK" "$MINI"; do
  [ -d "$(dirname "$DST")" ] || continue
  rsync -a --delete --exclude '__pycache__' "$SRC/" "$DST/"
  echo "synced -> $DST"
done

for DST in "$BACK" "$MINI"; do
  [ -d "$DST" ] || continue
  if diff -rq -x __pycache__ "$SRC" "$DST" > /dev/null; then
    echo "verified identical: $DST"
  else
    echo "DRIFT REMAINS after sync: $DST" >&2
    exit 1
  fi
done
