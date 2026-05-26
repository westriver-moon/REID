#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

trial="${1:-1}"
physical_gpu="${2:-0}"

conda run -n clipreid python main.py \
  --mode train \
  --dataset regdb \
  --regdb_data_path /home/cgv841/datasets/RegDB/ \
  --trial "${trial}" \
  --pretrain_choice LASTVIT_ORI \
  --lastvit_pretrained /home/cgv841/ybj/pretrained/ViT_190k.pth \
  --lastvit_pretrained_rgb /home/cgv841/ybj/pretrained/external/rgb_sysumm01_vitb_best_timm.pth \
  --lastvit_pretrained_ir /home/cgv841/ybj/pretrained/external/ir_sysumm01_vitb_best_timm.pth \
  --training_mode RGB_IR \
  --loss_names pair_clip \
  --clip_use_aug_rgb \
  --clip_loss_weight 1.0 \
  --batch-size 16 \
  --num_workers 4 \
  --total_train_epoch 10 \
  --milestones 4 8 \
  --warmup_epochs 1 \
  --checkpoint_epoch 1 \
  --test_modality IR \
  --eval_start_epoch 9999 \
  --eval_epoch 100 \
  --output_path logs/regdb_pair_clip_warmup_b16/ \
  --gpu_id 0 \
  --CUDA_VISIBLE_DEVICES "${physical_gpu}"
