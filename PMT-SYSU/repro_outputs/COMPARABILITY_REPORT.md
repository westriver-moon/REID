# Comparability Report

当前实现可与 PMT 官方 SYSU 设置比较，前提是使用相同数据、ImageNet ViT-B/16 初始化、24 epoch 训练和 10-trial all-search single-shot 评测。

已验证：

- 官方 ImageNet ViT-B/16 权重可加载到本实现，missing/unexpected 均为 0。
- 官方 PMT SYSU checkpoint 可加载到本实现，missing/unexpected 均为 0。
- 官方 checkpoint 的 1-trial 评测结果处于参考指标附近：Rank-1 70.55%，mAP 67.71%，mINP 54.96%。

尚未完成：

- 未运行完整 24 epoch 训练。
- 未运行官方 checkpoint 的完整 10-trial 平均评测。
- 未对自训练 checkpoint 做最终 10-trial 评估。

因此当前证据证明代码路径、权重加载、数据协议和基础评测可用，但不能声称已经完整复现论文训练结果。

