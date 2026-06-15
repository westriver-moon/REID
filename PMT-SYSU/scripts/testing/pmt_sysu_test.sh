#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-/home/cgv841/datasets/SYSU-MM01}"
WEIGHTS="${WEIGHTS:-outputs/pmt_sysu/official_reproduction/checkpoints/best.pth}"
MODE="${MODE:-all}"
GALLERY_MODE="${GALLERY_MODE:-single}"
TRIALS="${TRIALS:-10}"

python -m pmt_sysu.test \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root "$DATA_ROOT" \
  --weights "$WEIGHTS" \
  --mode "$MODE" \
  --gallery-mode "$GALLERY_MODE" \
  --trials "$TRIALS" \
  --device "cuda:${GPU}"

