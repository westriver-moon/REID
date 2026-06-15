# Commands

## Static Check

```bash
cd /home/cgv841/ybj/PMT-SYSU
python -m compileall -q pmt_sysu
```

## Unit Tests

```bash
cd /home/cgv841/ybj/PMT-SYSU
/home/cgv841/anaconda3/envs/reid/bin/python -m pytest pmt_sysu/tests -q
```

## Preflight

```bash
cd /home/cgv841/ybj/PMT-SYSU
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m pmt_sysu.tools.preflight \
  --config pmt_sysu/config/sysu_pmt_mbpatch.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --device cuda:0
```

## Epoch 1 Smoke

```bash
cd /home/cgv841/ybj/PMT-SYSU
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m pmt_sysu.train \
  --config pmt_sysu/config/sysu_pmt_mbpatch.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output outputs/pmt_sysu/mbpatch_smoke_ep1 \
  --device cuda:0 \
  --smoke-batches 1 \
  --override data.num_workers=0
```

## Epoch 7 Smoke

```bash
cd /home/cgv841/ybj/PMT-SYSU
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m pmt_sysu.train \
  --config pmt_sysu/config/sysu_pmt_mbpatch.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output outputs/pmt_sysu/mbpatch_smoke_ep7 \
  --device cuda:0 \
  --smoke-batches 1 \
  --override data.num_workers=0 \
  --override train.start_epoch=7
```

## Full Training Candidate

```bash
cd /home/cgv841/ybj/PMT-SYSU
CUDA_VISIBLE_DEVICES=0 /home/cgv841/anaconda3/envs/reid/bin/python -m pmt_sysu.train \
  --config pmt_sysu/config/sysu_pmt_mbpatch.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --pretrained pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  --output outputs/pmt_sysu/mbpatch_reproduction \
  --device cuda:0
```

