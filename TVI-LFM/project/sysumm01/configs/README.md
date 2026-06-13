# SYSU-MM01 当前配置入口

更新时间：2026-06-13

## 当前主线

1. VCM-only RGB-IR tracklet 预训练：

```bash
project/sysumm01/configs/external_pretrain_vcm_only.yaml
```

2. SYSU-MM01 RGB-IR fine-tune：

```bash
project/sysumm01/configs/sysu_finetune_from_vcm_only.yaml
```

## 数据约定

- LLCM 已从当前外部预训练主线移除。
- VCM 索引使用 `project/sysumm01/datasets/vcm_index/vcm_train_tracklets.json`。
- VCM 路径中的 `data/` 是 `/home/cgv841/datasets/HITSZ-VCM/Train` 的符号链接，不是第二份数据。
- 当前 VCM 安全索引已过滤原始 `track_train_info.txt` 中 39 个 source range 的混入帧。
- 训练中间评估默认使用 `eval.selection_split: val`；最终 test 评估使用 `eval.final_seed: 0`，即 SYSU 常用 trial 0-9。
- 开发期如不希望每轮实验结束自动读取 test，可在配置中设置 `eval.run_final_test: false`。

## 旧实验配置

目录中仍保留若干历史消融配置，例如 SCHP、adapter、shared head、A/B/C patch 消融等。它们只用于复查旧实验，不是当前默认训练入口。
