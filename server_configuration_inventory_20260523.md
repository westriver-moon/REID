# 服务器当前配置清单（重置后基线）

生成时间：2026-05-23T11:09:56+08:00  
生成位置：/home/cgv841/ybj  
生成用户：cgv841

---

## 1. 清理执行结果（ybj 目录内）

已删除的代理/配置脚本与文档：

- /home/cgv841/ybj/non_research/codex_vscode_proxy_使用手册_zh.md
- /home/cgv841/ybj/non_research/rollback-vscode-extension-proxy.sh
- /home/cgv841/ybj/non_research/use-lab-github-proxy.sh
- /home/cgv841/ybj/non_research/use-lab-github-proxy.sh.bak-20260512-2039
- /home/cgv841/ybj/non_research/use-session-openai-proxy.sh
- /home/cgv841/ybj/non_research/ssh_proxy_tools/（目录）
- /home/cgv841/ybj/non_research/vscode_proxy_backups/（目录）

清理后 non_research 剩余文件：

- /home/cgv841/ybj/non_research/check_lastvit_preflight.sh
- /home/cgv841/ybj/non_research/training_preflight_guide_zh.md

验证结果：ybj 下已无上述关键词目标文件（use-session/openai-proxy/use-lab/rollback/vscode_proxy_backups/ssh_proxy_tools/使用手册）。

---

## 2. 系统基础信息

- 主机名：cgv841-SYS-7049GP-TRT
- 操作系统：Ubuntu 20.04.6 LTS (Focal)
- 内核：Linux 5.15.0-139-generic
- 运行时长：26 days, 6:38
- 登录用户：cgv841
- 当前工作目录：/home/cgv841/ybj

### CPU

- 型号：Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
- 逻辑 CPU：64
- 拓扑：2 sockets × 16 cores × 2 threads
- NUMA：2 节点

### 内存

- 总内存：188 GiB
- 已用：41 GiB
- 空闲：46 GiB
- 缓存/缓冲：101 GiB
- 可用：145 GiB
- 交换分区：30 GiB（已用 1.4 GiB）

### 磁盘

关键挂载：

- /：1.9T，已用 112G，剩余 1.7T
- /home（/dev/sda）：7.3T，已用 6.8T，剩余 135G，使用率 99%
- /home/lab929（/dev/sdb1）：15T，已用 3.4T，剩余 11T

---

## 3. 网络与端口状态

### 网络接口

- enp94s0：UP，IPv4 10.108.13.53/23，存在 IPv6 地址
- lo：127.0.0.1/8
- docker0：172.17.0.1/16

### 路由

- 默认网关：10.108.12.1 via enp94s0

### DNS

- /etc/resolv.conf 使用 systemd-resolved stub：127.0.0.53

### 本机监听端口（节选）

- 127.0.0.1:7890（监听）
- 127.0.0.1:7891（监听）
- 127.0.0.1:9090（监听）
- 0.0.0.0:22（SSH）
- 0.0.0.0:7860/6800/5556/5800（监听）

### 常见代理端口探测（127.0.0.1）

- 7897：CLOSED
- 7891：OPEN
- 1080：CLOSED
- 8080：CLOSED
- 3128：CLOSED

### 当前 shell 代理环境变量

- 未检测到 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / NO_PROXY

---

## 4. VS Code 远端配置状态

### VS Code Server 目录

- /home/cgv841/.vscode-server 存在

### 启动环境注入脚本

文件：/home/cgv841/.vscode-server/server-env-setup  
权限：-rwx------ (700)

当前脚本行为（摘要）：

- 固定设置：
  - TMPDIR=/home/cgv841/ybj/codex-local/tmp
  - XDG_RUNTIME_DIR=/home/cgv841/ybj/codex-local/tmp
- 代理目标：
  - VSCODE_PROXY_HOST=127.0.0.1
  - VSCODE_PROXY_PORT=7897
  - VSCODE_PROXY_SCHEME=socks5h
- 仅当 127.0.0.1:7897 可达时才导出 ALL_PROXY/HTTP(S)_PROXY
- 不可达则主动 unset 代理变量

说明：当前 7897 实测为 CLOSED，因此该脚本在当前状态下会走 unset 分支。

### Machine settings

文件：/home/cgv841/.vscode-server/data/Machine/settings.json

关键项：

- python.defaultInterpreterPath: /home/cgv841/anaconda3/envs/mindspore/bin/python
- remote.extensionKind:
  - openai.chatgpt: workspace
  - GitHub.copilot: workspace
  - GitHub.copilot-chat: workspace
  - ms-python.python: ui
- http.proxySupport: override

### VS Code / 扩展进程现状（节选）

- 当前存在 cgv841 用户的 vscode-server 主进程、extensionHost、fileWatcher、ptyHost
- 检测到 openai.chatgpt 对应 codex app-server 进程在运行
- 同机还有 lab929 用户独立的 vscode-server 与 openai.chatgpt/codex 进程

---

## 5. 与“从头配置”直接相关的事实

- ybj 下旧代理脚本与使用手册已清空（你可以从零重建）。
- 服务器本机当前可用的本地代理监听端口是 7891，不是 7897。
- 但 VS Code Server 环境脚本仍硬编码检查 7897。
- 这意味着：如果现在重启 VS Code Server，默认不会注入代理环境变量。

---

## 6. 建议作为下一步重建起点（非自动执行）

1. 先明确目标端口（当前建议以 7891 为起点做新配置）。
2. 新建一份最小化、单一来源的代理脚本（避免多版本漂移）。
3. 再调整 /home/cgv841/.vscode-server/server-env-setup 与新脚本保持一致。
4. 执行 Remote-SSH: Kill VS Code Server on Host 后重连验证。
5. 最后再补文档，避免先写文档后配置漂移。

---

## 7. 本次采集命令产物（临时文件）

- /tmp/ybj_server_snapshot_sys.txt
- /tmp/ybj_server_snapshot_net.txt
- /tmp/ybj_server_snapshot_vscode.txt

可按需保留或删除。