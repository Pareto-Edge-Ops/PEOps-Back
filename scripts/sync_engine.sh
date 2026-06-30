#!/usr/bin/env bash
# Sync the astra engine from its source of truth (the Astra-PoC research repo)
# into the two vendored copies. All engine edits land in the PoC first; this
# script is the only sanctioned way bytes flow into the backend.
set -euo pipefail

SRC="${ASTRA_POC_DIR:-$HOME/Desktop/Astra-PoC}/astra"
BACK="$(cd "$(dirname "$0")/.." && pwd)/astra"
MINI="$(cd "$(dirname "$0")/../.." && pwd)/Astra-PoC/astra"

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
