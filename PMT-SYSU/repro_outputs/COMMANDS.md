# Commands

## Environment

```bash
cd /home/cgv841/ybj/PMT-SYSU
/home/cgv841/anaconda3/envs/reid/bin/python -m pip install 'pytest>=7.0'
```

## Static Import Check

```bash
cd /home/cgv841/ybj/PMT-SYSU
python -m compileall -q pmt_sysu
```

## Unit Tests

```bash
cd /home/cgv841/ybj/PMT-SYSU
/home/cgv841/anaconda3/envs/reid/bin/python -m pytest pmt_sysu/tests -q
```

## Download Weights

```bash
cd /home/cgv841/ybj/PMT-SYSU
/home/cgv841/anaconda3/envs/reid/bin/python -m pmt_sysu.tools.download_weights --imagenet
/home/cgv841/anaconda3/envs/reid/bin/python -m pmt_sysu.tools.download_weights --official
```

## Preflight

```bash
cd /home/cgv841/ybj/PMT-SYSU
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m pmt_sysu.tools.preflight \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --device cuda:0
```

## Smoke Training

```bash
cd /home/cgv841/ybj/PMT-SYSU
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m pmt_sysu.train \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output outputs/pmt_sysu/smoke \
  --device cuda:0 \
  --smoke-batches 1
```

```bash
cd /home/cgv841/ybj/PMT-SYSU
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m pmt_sysu.train \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output outputs/pmt_sysu/smoke_epoch7 \
  --device cuda:0 \
  --smoke-batches 1 \
  --override train.start_epoch=7
```

## Official Checkpoint 1-Trial Evaluation

```bash
cd /home/cgv841/ybj/PMT-SYSU
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m pmt_sysu.test \
  --config pmt_sysu/config/sysu_pmt.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --weights pretrained/pmt_sysu_vit_official.pth \
  --mode all \
  --gallery-mode single \
  --trials 1 \
  --device cuda:0 \
  --output outputs/pmt_sysu/official_weight_trial1
```

## Full Training Command

```bash
cd /home/cgv841/ybj/PMT-SYSU
GPU=0 DATA_ROOT=/home/cgv841/datasets/SYSU-MM01 \
  PRETRAIN=pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  OUTPUT=outputs/pmt_sysu/official_reproduction \
  bash scripts/training/pmt_sysu_train.sh
```

## Full 10-Trial Test Command

```bash
cd /home/cgv841/ybj/PMT-SYSU
GPU=0 DATA_ROOT=/home/cgv841/datasets/SYSU-MM01 \
  WEIGHTS=outputs/pmt_sysu/official_reproduction/checkpoints/best.pth \
  TRIALS=10 \
  bash scripts/testing/pmt_sysu_test.sh
```

