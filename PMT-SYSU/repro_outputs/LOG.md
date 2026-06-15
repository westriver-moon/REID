# Log

1. 读取用户提供的 PMT 复现方案，并按用户最新要求将目标目录从 `TVI-LFM/project/pmt_sysu` 调整为独立目录 `/home/cgv841/ybj/PMT-SYSU`。
2. 读取本地 RigorPilot skills：`ai-research-reproduction`、`repo-intake-and-plan`、`env-and-assets-bootstrap`、`minimal-run-and-audit`、`run-train`。
3. 克隆官方 PMT 到 `/tmp/PMT_official` 作为只读参考。
4. 核查 SYSU-MM01 数据缓存：
   - `train_rgb_resized_img.npy`：22258 张。
   - `train_ir_resized_img.npy`：11909 张。
   - label 范围：0..394。
5. 实现独立 PMT-SYSU 工程。
6. 使用 `reid` conda 环境执行验证。该环境原本缺少 pytest，已安装 `pytest>=7.0`。
7. 运行 pytest：7 个测试全部通过。
8. 下载 ImageNet ViT-B/16 权重和官方 PMT SYSU checkpoint。
9. 运行真实 preflight，检查 Gray-IR 与 RGB-IR 两阶段。
10. 运行 epoch 1 和 epoch 7 单 batch 训练 smoke。
11. 验证官方 checkpoint 能 strict-compatible 加载，missing/unexpected 均为 0。
12. 用官方 checkpoint 执行 1-trial SYSU all-search single-shot 评测。

