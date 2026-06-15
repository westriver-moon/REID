#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-/home/cgv841/datasets/SYSU-MM01}"
PRETRAIN="${PRETRAIN:-pretrained/jx_vit_base_p16_224-80ecf9dd.pth}"
OUTPUT="${OUTPUT:-outputs/pmt_sysu/official_reproduction}"

python -m pmt_sysu.train \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root "$DATA_ROOT" \
  --pretrained "$PRETRAIN" \
  --output "$OUTPUT" \
  --device "cuda:${GPU}"

