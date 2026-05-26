#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

gpu_id="${1:-0}"
output="${OUTPUT_DIR:-logs/sysu_ir_vcm_ir/a3_source_aware_loss}"
config="${CONFIG:-project/sysumm01/configs/sysu_ir_vcm_ir_a3_source_aware_loss.yaml}"

CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n clipreid python -u project/sysumm01/engine/train.py \
  --config "${config}" \
  --output "${output}" \
  --device cuda \
  --print-freq 50
