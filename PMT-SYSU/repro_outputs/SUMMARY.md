# PMT-SYSU 复现摘要

目标：在 `/home/cgv841/ybj/PMT-SYSU` 新建独立 PMT 基线，用于 SYSU-MM01 上复现 AAAI 2023 PMT。

结果：已完成独立工程实现，并完成静态导入、pytest、真实 ImageNet 权重 preflight、epoch 1/epoch 7 单 batch 训练 smoke、官方 PMT checkpoint 加载和 1-trial SYSU 评测验证。

本次没有启动完整 24 epoch 训练。

关键验证结果：

- SYSU 训练缓存存在：RGB 22258 张，IR 11909 张，标签 0..394，共 395 类。
- ImageNet ViT-B/16 权重加载：missing keys 0，unexpected keys 0。
- 官方 PMT SYSU checkpoint 加载：missing keys 0，unexpected keys 0。
- preflight epoch 1：`stage=gray_ir`，特征形状 `[64, 768]`，MSEL/DCL 为 0。
- preflight epoch 7：`stage=rgb_ir`，特征形状 `[64, 768]`，MSEL/DCL 已启用且 finite。
- 训练 smoke epoch 1：`stage=gray_ir`，单 batch 成功。
- 训练 smoke epoch 7：`stage=rgb_ir`，单 batch 成功。
- 官方权重 1-trial all-search single-shot：Rank-1 70.55%，mAP 67.71%，mINP 54.96%。

注意：上述 1-trial 不是最终论文指标。正式对比应运行 10 trials。

