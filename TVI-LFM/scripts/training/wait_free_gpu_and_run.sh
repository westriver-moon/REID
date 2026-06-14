#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

THRESHOLD_MB="${THRESHOLD_MB:-1000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

while true; do
  GPU="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v threshold="${THRESHOLD_MB}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if (($2 + 0) < threshold) {print $1; exit}}'
  )"
  if [ -n "${GPU}" ]; then
    echo "[wait-gpu] $(date '+%F %T') selected gpu=${GPU} threshold_mb=${THRESHOLD_MB}"
    exec "$@" "${GPU}"
  fi
  echo "[wait-gpu] $(date '+%F %T') no free gpu under ${THRESHOLD_MB} MiB; sleeping ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

