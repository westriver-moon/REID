#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

conda run -n clipreid python main.py \
  --mode train \
  --dataset sysu \
  --sysu_data_path /home/cgv841/datasets/SYSU-MM01/ \
  --pretrain_choice LASTVIT_ORI \
  --lastvit_pretrained /home/cgv841/ybj/pretrained/ViT_190k.pth \
  --lastvit_pretrained_rgb /home/cgv841/ybj/pretrained/external/rgb_sysumm01_vitb_best_timm.pth \
  --lastvit_pretrained_ir /home/cgv841/ybj/pretrained/external/ir_sysumm01_vitb_best_timm.pth \
  --training_mode RGB_IR \
  --loss_names id,wrt,clip,proto \
  --enable_rgb_ir_clip \
  --enable_proto_align \
  --clip_use_aug_rgb \
  --clip_loss_weight 1.0 \
  --proto_loss_weight 0.05 \
  --proto_momentum 0.9 \
  --batch-size 32 \
  --num_workers 8 \
  --total_train_epoch 1 \
  --test_modality IR \
  --train_max_iter 20 \
  --eval_start_epoch 9999 \
  --eval_epoch 100 \
  --output_path logs/clip_align_smoke/ \
  --gpu_id 0 \
  --CUDA_VISIBLE_DEVICES 0
