#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

gpu_id="${1:-0}"
output="${OUTPUT_DIR:-logs/sysu_ir_vcm_ir/a32_vcm_id03_tri05}"
config="${CONFIG:-project/sysumm01/configs/sysu_ir_vcm_ir_a32_vcm_id03_tri05.yaml}"

CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n clipreid python -u project/sysumm01/engine/train.py \
  --config "${config}" \
  --output "${output}" \
  --device cuda \
  --print-freq 50
