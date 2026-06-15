# Scientific Changelog

本次实现目标是忠实复现 PMT 基线，不引入新算法。

保留的科学语义：

- PMT 官方 TransReID-style ViT-B/16。
- 输入尺寸 256x128。
- overlapping patch embedding，patch 16x16，stride 12x12。
- 12 层 Transformer，768 维 CLS feature。
- BNNeck 后无偏置 classifier。
- batch 排列为 `[visible, ir]`，不是 TVI-LFM 的三分支排列。
- epoch 1-6 使用 Gray-IR，仅 ID + 模态内 Triplet。
- epoch 7-24 使用 RGB-IR，启用 global Triplet + MSEL + DCL。
- SYSU query 为 IR，gallery 为 RGB。
- all-search single-shot 支持 10 trials。

工程兼容性改动：

- 路径改为 YAML/CLI。
- `.cuda()` 改为 `.to(device)`。
- `torch.load` 使用 `map_location`。
- 移除 `Variable`。
- 旧 `addmm_` 写法改为现代 PyTorch 签名。
- 增加 layout、shape、finite loss 和 gradient 断言。
- 增加 checkpoint resume 信息。

这些改动不改变 PMT 模型结构、训练阶段、损失定义或评测协议。

