# SYSU-MM01 当前配置入口

更新时间：2026-06-14

## 当前主线

当前只保留两条单模态训练入口。

1. RGB 侧：MSMT17 + SYSU RGB 混合训练

```bash
project/sysumm01/configs/mixed_msmt17_sysumm01_rgb_lastvit.yaml
```

2. IR 侧：SYSU IR + VCM IR SCHP quality sampling

```bash
project/sysumm01/configs/sysu_ir_vcm_ir_schp_full_s2_steps800.yaml
```

## 数据约定

- RGB 训练使用 SYSU RGB train+val 与 MSMT17 train+val。
- IR 训练使用 SYSU IR train+val 与 VCM IR filtered tracklet index。
- VCM IR 索引使用 `project/sysumm01/datasets/vcm_index/vcm_train_tracklets_schp_filtered.json`。
- VCM IR SCHP 质量分数使用 `data/schp_quality/vcm_ir_quality_full.json`。
- 原始数据集目录不由配置目录管理，不在本次清理范围内。

## 已移除内容

2026-06-14 起，RGB-IR 双模态、adapter、shared head、external RGB-IR pretrain、VCM RGB-IR tracklet pretrain、RegDB / LLCM 相关实验配置均已从默认仓库配置中移除。
