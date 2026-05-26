#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

gpu_id="${1:-0}"
stage1_output="${STAGE1_OUTPUT:-logs/sysu_ir_vcm_ir/stage1_singleframe_ep30}"
stage2_output="${STAGE2_OUTPUT:-logs/sysumm01/ir_finetune_from_vcm_stage1_ep15}"
stage1_config="${STAGE1_CONFIG:-project/sysumm01/configs/sysu_ir_vcm_ir_stage1_ep30.yaml}"
stage2_config="${STAGE2_CONFIG:-project/sysumm01/configs/sysumm01_ir_lastvit_finetune_ep15.yaml}"

CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n clipreid python -u project/sysumm01/engine/train.py \
  --config "${stage1_config}" \
  --output "${stage1_output}" \
  --device cuda \
  --print-freq 50

CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n clipreid python -u project/sysumm01/engine/train.py \
  --config "${stage2_config}" \
  --output "${stage2_output}" \
  --device cuda \
  --print-freq 50 \
  train.init_checkpoint="${stage1_output}/checkpoints/best.pth"
