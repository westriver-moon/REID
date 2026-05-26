# LAST-ViT 训练前防卡死 / 防假写入说明

这份说明是给当前这个工作区准备的，目标只有两个：

1. 不要再因为终端目录不对、输出目录不可写、关键文件没同步而把训练卡在半路。
2. 不要再出现“编辑器里看起来已经改了，但终端真正运行时没看到”的情况。

对应的自检脚本在：

`non_research/check_lastvit_preflight.sh`

## 一、先说结论

这类问题里，最危险的不是训练报错，而是**你以为自己在正确状态下启动了训练，实际上没有**。这会导致：

- 相对路径指到错误位置。
- 输出写到意料之外的目录。
- 终端侧读到旧文件，训练用的不是你刚编辑过的版本。
- 训练已经开始，但日志没有落到你以为会落的地方。

所以建议把下面这条规则固定下来：

**每次正式训练前，先跑一遍预检脚本。预检不过，不要启动训练。**

## 二、这个脚本能检查什么

脚本会检查下面几类问题：

1. 当前终端目录是否位于正确仓库内。
2. `--repo` 指向的目录是否真的是 git 仓库根目录。
3. 关键文件是否存在。
4. 关键文件里是否真的出现了你期望的标记字符串。
5. 训练输出目录是否可创建、可写。
6. 可选地扫描同名仓库目录，辅助排查“是不是进错仓库”。

## 三、这个脚本不能检查什么

这点要说清楚。

这个脚本运行在**终端侧真实文件系统**里，所以它不能直接读取 VS Code 编辑器内部的那层视图，也不能数学上证明“编辑器侧和终端侧 100% 同步”。

它能做的是一件更实用的事：

如果你刚刚改过某个关键文件，那么你把“这次改动一定会出现的关键字”通过 `--must-contain` 传给脚本；只要终端侧没有看到这个关键字，脚本就会立刻失败。这就足以把“假写入”拦在训练启动之前。

换句话说：

- 它不能证明两边永远一致。
- 但它能在训练前快速发现“终端现在看到的不是你以为那份代码”。

## 四、推荐用法

### 1. 最小推荐命令

```bash
cd /home/cgv841/ybj/LAST-ViT
bash ../non_research/check_lastvit_preflight.sh \
  --repo /home/cgv841/ybj/LAST-ViT \
  --strict-cwd \
  --output-dir runs/exp_sysumm01_check \
  --must-contain 'project/sysumm01/engine/train.py::install_stream_tee' \
  --must-contain 'project/sysumm01/engine/train.py::original_stdout' \
  --must-contain 'project/sysumm01/engine/train.py::log_handle.close()'
```

这个命令的含义是：

- 强制要求当前终端目录处在正确仓库里。
- 明确指定真正的训练仓库路径。
- 检查输出目录是否可写。
- 检查终端侧的 `train.py` 是否真的已经包含我们最近加进去的日志回退逻辑。

### 2. 如果你怀疑自己进错仓库

```bash
cd /home/cgv841/ybj/LAST-ViT
bash ../non_research/check_lastvit_preflight.sh \
  --repo /home/cgv841/ybj/LAST-ViT \
  --scan-duplicates
```

这个选项会额外扫描同名仓库目录，用来辅助判断当前机器上是不是有多个 `LAST-ViT`。

### 3. 如果你刚刚改了别的文件

例如你刚改了配置文件，想确认终端已经看到新的字段，可以继续追加：

```bash
cd /home/cgv841/ybj/LAST-ViT
bash ../non_research/check_lastvit_preflight.sh \
  --repo /home/cgv841/ybj/LAST-ViT \
  --output-dir runs/exp_new \
  --must-contain 'project/sysumm01/configs/sysumm01_vitb_lastvit.yaml::topk:'
```

原则很简单：

- 你改了什么，就把那个改动对应的关键字喂给脚本。
- 脚本看不到，就说明终端侧还不能安全开跑。

## 五、如何解读脚本输出

- `[ OK ]`：这项检查已经通过。
- `[WARN]`：不是立即致命，但你应该人工确认。
- `[FAIL]`：不要继续训练，先处理问题。

最常见的几种失败及处理方式：

### 1. 当前目录不在仓库里

处理：

```bash
cd /home/cgv841/ybj/LAST-ViT
```

然后重新跑预检。

### 2. 关键字未命中

这通常表示下面几种情况之一：

1. 你以为已经改了文件，但终端侧没看到。
2. 你改的是另一份仓库或另一层视图。
3. 你传入的关键字写错了。

处理顺序建议是：

1. 先在终端里直接 `sed -n` 或 `grep -n` 读取该文件。
2. 如果还是看不到改动，停止训练，不要赌。
3. 新开一个终端，重新 `cd /home/cgv841/ybj/LAST-ViT`。
4. 再次确认文件内容后再继续。

### 3. 输出目录不可写

处理：

1. 换一个新的输出目录。
2. 或先检查父目录权限。

建议始终给每次实验一个新的目录，不要复用旧目录覆盖结果。

## 六、建议固化成日常流程

每次正式训练，建议固定按下面顺序执行：

1. 新开终端。
2. 显式进入正确目录：`cd /home/cgv841/ybj/LAST-ViT`。
3. 跑一次预检脚本。
4. 预检通过后，再启动训练。
5. 训练输出目录使用全新的名字。

## 六点五、长实验必须拆成多个小步执行

如果这次任务同时满足下面任意一条，就不要把所有动作一次性打包执行：

- 需要改代码或改配置。
- 需要开多张卡并行训练。
- 单次训练预计超过几分钟。
- 训练结束后还要继续做汇总、对比和可视化。

推荐把一次长实验固定拆成下面这些小步，并且**每一步都先给出中间结果，再进入下一步**：

1. 明确本轮实验目的，只保留必要变量，不要同时改太多条件。
2. 先写配置文件，再单独检查配置文件是否真的出现在终端侧仓库里。
3. 在启动训练前，先跑一次预检脚本。
4. 预检通过后，先确认 GPU 分配、输出目录和日志路径。
5. 再启动训练；如果是多卡并行，也应该把每张卡对应的配置和输出目录单独列清楚。
6. 一旦拿到关键中间结果，就立刻汇报，不要等全部实验结束才第一次汇报。
7. 所有训练结束后，再统一读取最终 JSON、生成图像、更新结论文档。

推荐汇报的中间结果至少包括：

- 本轮配置路径。
- 输出目录。
- 使用的 GPU 编号。
- 预检是否通过。
- 第 1 个 epoch 的训练摘要。
- 训练完成后的 final 10-trial 指标。
- 可视化图像输出目录。

这样做的目的不是让流程更慢，而是避免下面两种最常见的卡住方式：

- 已经启动了长训练，后来才发现终端侧看到的是旧文件。
- 四个实验同时开跑，但直到结束才发现其中一个配置写错或输出目录冲突。

如果某一步失败，建议立刻停在当前步，不要把后面的步骤也一起继续执行。

### 针对当前 SYSU-MM01 / LAST-ViT 仓库的额外建议

当前仓库已经确认过一次“采样器修复只改到了编辑器侧、终端侧仍是旧文件”的问题。因此后面只要实验依赖这个修复，预检脚本建议明确追加下面两个关键字：

```bash
--must-contain 'project/sysumm01/datasets/sysumm01.py::self.epoch = 0' \
--must-contain 'project/sysumm01/datasets/sysumm01.py::random.Random(self.seed + self.epoch)'
```

如果是官方 LAST 权重实验，建议再继续追加对应配置文件和权重路径的关键字，不要只检查代码文件本身。

### 四卡并行实验的执行节奏建议

如果需要在四张卡上同时跑四个实验，推荐顺序是：

1. 先把四份配置文件全部写好。
2. 单独确认四份配置在终端侧都能被 `cat` 到。
3. 再统一跑预检，确认输出目录可写、采样器修复可见、配置关键字可见。
4. 明确记录 GPU 与实验的映射关系。
5. 四条训练命令再同时启动。
6. 启动完成后，先回报每条训练的日志路径和首轮输出是否正常。
7. 所有训练结束后，再统一做 final JSON 汇总与绘图。

不要把“写配置、预检、开四卡、等结果、出图”压成一条大命令，因为一旦中间某一步出错，回退会非常困难。

## 七、推荐的训练前命令模板

```bash
cd /home/cgv841/ybj/LAST-ViT
bash ../non_research/check_lastvit_preflight.sh \
  --repo /home/cgv841/ybj/LAST-ViT \
  --strict-cwd \
  --output-dir runs/exp_sysumm01_last_seed42 \
  --must-contain 'project/sysumm01/engine/train.py::install_stream_tee' \
  --must-contain 'project/sysumm01/engine/train.py::original_stdout' \
  --must-contain 'project/sysumm01/engine/train.py::log_handle.close()'

conda run -n clipreid python project/sysumm01/engine/train.py \
  --config project/sysumm01/configs/sysumm01_vitb_lastvit.yaml \
  --output runs/exp_sysumm01_last_seed42 \
  --seed 42
```

## 八、回退原则

如果预检失败，不要继续往下跑，也不要立刻做大范围重置。建议按下面顺序回退：

1. 先停在当前步骤，不启动训练。
2. 先确认当前终端目录和仓库根目录。
3. 先确认终端侧文件里能不能看到关键标记。
4. 如果看不到，优先新开一个终端，再重试预检。
5. 如果只是这次实验目录有问题，换一个新的输出目录，不要覆盖旧结果。

这样做的目的不是“完全杜绝所有问题”，而是尽量把问题提前暴露在训练前，而且暴露得足够局部，方便回退。


## 九、GitHub 代理与官方 LAST 权重补充说明

这次接官方 LAST 权重时，额外验证出了两条后面会反复用到的经验，建议直接固化。

### 1. GitHub 代理不要再默认用 `socks5h`

在这台机器上，`socks5h://127.0.0.1:7897` 会出现一种很迷惑的状态：

- `lab_github_proxy_status` 旧版本只看端口时会误报“可达”
- `api.github.com` 可能能通
- 但 `github.com` 的 HTTPS git 会在 TLS 握手处失败

这次已经确认，稳定可用的默认值应当是：

- `socks5://127.0.0.1:7897`

也就是让本机先解析 `github.com`，再走 SOCKS 转发，而不是让代理端代做 DNS 解析。

因此后面遇到 GitHub 拉权重、`git clone`、`git ls-remote` 这类动作时，先做两件事：

```bash
use-lab-github-proxy
lab_github_proxy_status
```

只有 `lab_github_proxy_status` 明确显示 HTTPS git ready，再继续下载或 git 操作。

### 2. 官方 `ViT_190k.pth` 必须检查文件大小

这次本地曾经有一个同名文件，但实际上是截断坏包，只有几十 MB，不能直接用。

后续请直接用下面这个标准判断：

- 正常文件大小约 991 MB
- 精确字节数：`1039002177`

如果你下载到的 `ViT_190k.pth` 远小于这个值，就不要继续训练，先认定它是坏包。

### 3. 官方权重下载更稳的方式

如果直接访问 GitHub release 页面地址不稳定，不要反复撞网页下载链路，优先走：

1. GitHub API 的 release asset 接口
2. 拿到签名后的 `release-assets.githubusercontent.com` 直链
3. 再下载大文件

这条链路在当前环境里比直接打 release 网页 URL 更稳定。

### 4. `torch.load` 官方权重前，`clipreid` 环境要先补 `omegaconf`

官方 checkpoint 不只是裸 state_dict，还带训练侧对象元数据；如果 `clipreid` 环境没有 `omegaconf`，直接 `torch.load` 会报：

- `ModuleNotFoundError: No module named 'omegaconf'`

最小修复方式：

```bash
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
source /home/cgv841/anaconda3/etc/profile.d/conda.sh
conda activate clipreid
python -m pip install omegaconf
```

这里特意取消代理，是因为当前 pip 环境缺少 SOCKS 支持；保留 `ALL_PROXY=socks5://...` 时，pip 会先报代理依赖错误。

### 5. 官方 LAST 权重训练前，建议新增两个预检关键字

如果后面还要继续做官方 LAST 权重实验，预检脚本建议追加：

```bash
--must-contain 'project/sysumm01/models/backbones.py::_convert_torchvision_vit_to_timm' --must-contain 'runs/exp_configs/last_official_seed42.yaml::pretrained/ViT_190k.pth'
```

这两个关键字分别能防住：

- 终端侧没看到权重键名转换逻辑
- 终端侧没看到官方权重实验配置

这样做的价值很直接：在正式训练前，把“编辑器里改了，但终端真正跑的不是那份代码”这类问题尽可能挡在外面。
