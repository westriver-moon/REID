#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

gpu_id="${1:-0}"
output="${OUTPUT_DIR:-logs/sysumm01/ir_finetune_from_vcm_k2_preserve_cls_ep12}"
config="${CONFIG:-project/sysumm01/configs/sysumm01_ir_finetune_from_vcm_k2_preserve_cls_ep12.yaml}"

CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n clipreid python -u project/sysumm01/engine/train.py \
  --config "${config}" \
  --output "${output}" \
  --device cuda \
  --print-freq 50
