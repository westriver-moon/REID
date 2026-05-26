# ybj 上传清单

更新时间：2026-05-26

## 结论

可以上传，但不建议直接在当前状态下执行一次无差别的 `git add .`。

当前工作区有两个先处理的点：

1. 根目录 `.git/` 是空目录，Git 现在不会把 `/home/cgv841/ybj` 识别为有效仓库。
2. `TVI-LFM/.git/` 是独立仓库。如果目标是把整个 `ybj` 上传到 `https://github.com/westriver-moon/REID.git`，需要先把这个嵌套仓库的元数据备份或移开，否则外层 Git 会把 `TVI-LFM` 当成嵌入仓库处理。

这里的排除原则不是“保密”，而是三件事：

- 普通 Git 是否适合长期追踪
- GitHub 单文件和仓库体积是否容易失控
- 文件是否属于可复现源码，还是训练产物、权重或数据副本

## 推荐直接上传到 Git 的内容

### 根目录

- `README.md`
- `.gitignore`
- `UPLOAD_MANIFEST_ZH.md`
- `server_configuration_inventory_20260523.md`（可选；如果你想保留服务器基线记录，可以一并上传）

### `manual_weight_upload/`

- `manual_weight_upload/README_ZH.md`
- `manual_weight_upload/ingest_official_weights.sh`

这部分是脚本和文档，本身轻量，适合入库。

### `non_research/`

- `non_research/check_lastvit_preflight.sh`
- `non_research/training_preflight_guide_zh.md`
- `non_research/yes.md`
- `non_research/codex_proxy/check-proxy.sh`（可选）
- `non_research/codex_proxy/check-vscode-env.sh`（可选）
- `non_research/codex_proxy/install-vscode-code-server-hook.sh`（可选）
- `non_research/codex_proxy/install-vscode-server-env.sh`（可选）
- `non_research/codex_proxy/proxy-env.sh`（可选）

这些文件是脚本或说明文档；`codex_proxy` 更偏环境运维，如果你只想保存研究代码，也可以不上传。

### `TVI-LFM/` 中建议保留的源码与配置

- `TVI-LFM/LICENSE`
- `TVI-LFM/README.md`
- `TVI-LFM/main.py`
- `TVI-LFM/requirements.txt`
- `TVI-LFM/config/`
- `TVI-LFM/core/`
- `TVI-LFM/data_loader/`
- `TVI-LFM/generators/`
- `TVI-LFM/network/`（排除 `network/clip_model/data/` 和 `network/clip_model/checkpoint.pth.tar`）
- `TVI-LFM/project/`（排除 `project/sysumm01/datasets/vcm_index/`）
- `TVI-LFM/scripts/`
- `TVI-LFM/solver/`
- `TVI-LFM/tools/`

这些目录主要是源码、配置和训练入口，适合放进 Git。

## 不要直接上传到普通 Git 的内容

### 权重、检查点和二进制模型

- `pretrained/`
- `manual_weight_upload/inbox/`
- `TVI-LFM/base_model/`
- `TVI-LFM/logs/**/checkpoints/`
- `TVI-LFM/outputs/**/checkpoints/`
- `TVI-LFM/network/clip_model/checkpoint.pth.tar`
- 所有 `*.pth`、`*.pt`、`*.ckpt`、`*.onnx`、`*.safetensors`、`*.h5`

这些文件要么已经超过或接近 GitHub 的 100MB 单文件限制，要么会让仓库历史迅速膨胀。

当前工作区里已经发现的典型大文件包括：

- `pretrained/ViT_190k.pth`
- `pretrained/official/transreid_market_official.pth`
- `pretrained/official/vitb16_ics_official.pth`
- `manual_weight_upload/inbox/transreid_market_official.pth`
- `manual_weight_upload/inbox/vitb16_ics_official.pth`
- `TVI-LFM/logs/sysu_ir_vcm_ir/a34_top5_reeval/checkpoints/top_epoch_034_map_0.837851.pth`

### 数据集、派生数组和索引

- `TVI-LFM/datasets/`
- `TVI-LFM/project/sysumm01/datasets/vcm_index/`
- 所有 `*.npy`、`*.npz`

这些内容技术上不一定全部超限，但它们本质上是数据副本或派生产物，不适合长期作为普通 Git 历史的一部分。

### 训练日志、运行输出和本地状态

- `TVI-LFM/logs/`
- `TVI-LFM/outputs/`
- `manual_weight_upload/logs/`
- `non_research/codex_proxy/login-device.log`
- `non_research/codex_proxy/login-device-live.log`
- 所有 `*.log`、`*.out`

### 本地工具运行时和临时目录

- `codex-local/`
- `tmp/`
- `.codex/`
- `.agents/`
- `non_research/codex_proxy/backups/`

这里面包含本地二进制、缓存、临时 clone 和运行态文件，不适合跟研究代码绑定在一起。

## 可以保留，但更适合放到 Git LFS / Release / 云盘

- `TVI-LFM/datasets/*/Text/`
- `TVI-LFM/project/sysumm01/datasets/vcm_index/`

如果你确实想保留这些派生文本或索引，建议用 Git LFS、GitHub Release、对象存储或网盘，而不是普通 Git 提交。

## 建议的上传步骤

### 方案 A：把整个 `ybj` 作为一个私有仓库上传

1. 备份 `TVI-LFM` 的嵌套仓库元数据，避免外层 Git 把它当成嵌入仓库：

   `mv TVI-LFM/.git TVI-LFM/.git.backup-20260526`

2. 处理根目录空 `.git/`。如果确认这个目录没有任何有价值内容，可以先删除空目录再初始化：

   `rmdir .git`

3. 在 `/home/cgv841/ybj` 初始化根仓库并绑定远端：

   `git init`

   `git branch -M main`

   `git remote add origin https://github.com/westriver-moon/REID.git`

4. 优先显式添加轻量内容，不要第一步就 `git add .`：

   `git add .gitignore README.md UPLOAD_MANIFEST_ZH.md server_configuration_inventory_20260523.md`

   `git add manual_weight_upload/README_ZH.md manual_weight_upload/ingest_official_weights.sh`

   `git add non_research/check_lastvit_preflight.sh non_research/training_preflight_guide_zh.md non_research/yes.md non_research/codex_proxy/*.sh`

   `git add TVI-LFM/LICENSE TVI-LFM/README.md TVI-LFM/main.py TVI-LFM/requirements.txt`

   `git add TVI-LFM/config TVI-LFM/core TVI-LFM/data_loader TVI-LFM/generators TVI-LFM/network TVI-LFM/project TVI-LFM/scripts TVI-LFM/solver TVI-LFM/tools`

5. 检查暂存结果，确认没有把不该跟踪的内容带进去：

   `git status --short`

   `git ls-files | rg '(^pretrained/|^codex-local/|^tmp/|^TVI-LFM/logs/|^TVI-LFM/outputs/|\.pth$|\.npy$)'`

   第二条命令理想情况下应当没有输出。

6. 提交并推送：

   `git commit -m "Initialize private REID workspace repository"`

   `git push -u origin main`

### 方案 B：只上传 `TVI-LFM` 代码

如果你最后发现真正想保存的是实验代码而不是整个工作区，可以保留 `TVI-LFM/.git/`，单独整理 `TVI-LFM` 仓库，然后把 `manual_weight_upload/` 与 `non_research/` 作为附加文档再决定是否迁入。

这个方案更简单，但不能完整保存整个 `ybj` 工作区。

## 当前工作区体量概览

- `TVI-LFM` 约 26G，其中 `logs/` 约 24G，`outputs/` 约 2.0G，`datasets/` 约 386M
- `pretrained/` 约 2.9G
- `manual_weight_upload/` 约 661M
- `codex-local/` 约 264M
- `tmp/` 约 117M

这也是为什么需要先按源码与产物分层，再执行上传。