#!/usr/bin/env bash
set -Eeuo pipefail

info() {
    printf '[INFO] %s\n' "$*"
}

ok() {
    printf '[ OK ] %s\n' "$*"
}

warn() {
    printf '[WARN] %s\n' "$*" >&2
}

die() {
    printf '[FAIL] %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOH'
用途：在真正启动 LAST-ViT / SYSU-MM01 训练前，先做一次快速自检，尽量避免：

1. 终端不在正确仓库里。
2. 终端侧没有看到你以为已经写进去的代码改动。
3. 输出目录不可写，训练跑到一半才失败。

用法：
  bash non_research/check_lastvit_preflight.sh [选项]

常用选项：
  --repo PATH                 指定 LAST-ViT 仓库根目录。
  --output-dir PATH           指定本次实验输出目录；会测试该目录是否可创建、可写。
  --key-file RELPATH          指定需要检查存在性的关键文件，可重复传入。
  --must-contain RELPATH::TXT 指定关键文件里必须出现的标记字符串，可重复传入。
  --strict-cwd                要求当前终端目录必须位于仓库根目录内部，否则直接失败。
  --scan-duplicates           扫描同名仓库目录，辅助排查“是不是进错仓库”。
  -h, --help                  显示帮助。

默认会检查：
  - project/sysumm01/engine/train.py
  - project/sysumm01/configs/sysumm01_vitb_baseline.yaml
  - project/sysumm01/configs/sysumm01_vitb_lastvit.yaml

默认还会检查 train.py 中是否包含下面这些标记：
  - install_stream_tee
  - original_stdout
  - log_handle.close()

说明：
  这个脚本只能看到“终端侧真实文件系统”。
  它不能直接读取 VS Code 编辑器内部的虚拟视图；因此它无法数学上证明“两边 100% 同步”。
  但只要你把刚编辑过的关键字通过 --must-contain 传进来，它就能快速发现“终端侧其实没看到这次改动”。
EOH
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO=""
if [[ -d "$SCRIPT_DIR/../LAST-ViT/.git" || -f "$SCRIPT_DIR/../LAST-ViT/.git" ]]; then
    DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/../LAST-ViT" && pwd -P)"
fi

REPO_PATH="$DEFAULT_REPO"
OUTPUT_DIR=""
STRICT_CWD=0
SCAN_DUPLICATES=0
declare -a KEY_FILES=()
declare -a MUST_CONTAIN=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            [[ $# -ge 2 ]] || die "--repo 缺少参数"
            REPO_PATH="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || die "--output-dir 缺少参数"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --key-file)
            [[ $# -ge 2 ]] || die "--key-file 缺少参数"
            KEY_FILES+=("$2")
            shift 2
            ;;
        --must-contain)
            [[ $# -ge 2 ]] || die "--must-contain 缺少参数"
            MUST_CONTAIN+=("$2")
            shift 2
            ;;
        --strict-cwd)
            STRICT_CWD=1
            shift
            ;;
        --scan-duplicates)
            SCAN_DUPLICATES=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "未知参数: $1"
            ;;
    esac
done

[[ -n "$REPO_PATH" ]] || die "无法自动推断仓库路径，请显式传入 --repo /home/cgv841/ybj/LAST-ViT"
REPO_PATH="$(cd -- "$REPO_PATH" 2>/dev/null && pwd -P)" || die "仓库路径不存在: $REPO_PATH"

GIT_TOP="$(git -C "$REPO_PATH" rev-parse --show-toplevel 2>/dev/null)" || die "不是 git 仓库: $REPO_PATH"
GIT_TOP="$(cd -- "$GIT_TOP" && pwd -P)"

CURRENT_PWD="$(pwd -P)"
REPO_NAME="$(basename -- "$REPO_PATH")"

info "当前目录: $CURRENT_PWD"
info "仓库根目录: $GIT_TOP"
info "当前分支: $(git -C "$REPO_PATH" branch --show-current 2>/dev/null || printf 'unknown')"
info "当前提交: $(git -C "$REPO_PATH" rev-parse --short HEAD)"

if [[ "$REPO_PATH" != "$GIT_TOP" ]]; then
    die "--repo 指向的目录与 git 根目录不一致：repo=$REPO_PATH git_root=$GIT_TOP"
fi

if [[ "$CURRENT_PWD" == "$REPO_PATH" || "$CURRENT_PWD" == "$REPO_PATH"/* ]]; then
    ok "当前终端目录位于仓库内"
else
    if (( STRICT_CWD )); then
        die "当前终端目录不在仓库内，请先 cd 到 $REPO_PATH"
    fi
    warn "当前终端目录不在仓库内；后续训练命令请显式 cd 到 $REPO_PATH"
fi

STATUS_OUTPUT="$(git -C "$REPO_PATH" status --short --untracked-files=no)"
if [[ -n "$STATUS_OUTPUT" ]]; then
    warn "仓库存在未提交修改；这通常没问题，但训练前应确认终端侧确实看到了这些改动。"
    printf '%s\n' "$STATUS_OUTPUT"
else
    ok "git 工作区干净"
fi

if [[ ${#KEY_FILES[@]} -eq 0 ]]; then
    KEY_FILES=(
        "project/sysumm01/engine/train.py"
        "project/sysumm01/configs/sysumm01_vitb_baseline.yaml"
        "project/sysumm01/configs/sysumm01_vitb_lastvit.yaml"
    )
fi

if [[ ${#MUST_CONTAIN[@]} -eq 0 ]]; then
    MUST_CONTAIN=(
        "project/sysumm01/engine/train.py::install_stream_tee"
        "project/sysumm01/engine/train.py::original_stdout"
        "project/sysumm01/engine/train.py::log_handle.close()"
    )
fi

for rel_path in "${KEY_FILES[@]}"; do
    abs_path="$REPO_PATH/$rel_path"
    [[ -f "$abs_path" ]] || die "关键文件不存在: $rel_path"
    file_size="$(wc -c < "$abs_path" | tr -d '[:space:]')"
    file_hash="$(sha256sum "$abs_path" | awk '{print $1}')"
    file_mtime="$(stat -c '%y' "$abs_path" | cut -d'.' -f1)"
    ok "关键文件存在: $rel_path | size=${file_size}B | sha256=${file_hash:0:16} | mtime=${file_mtime}"
done

for item in "${MUST_CONTAIN[@]}"; do
    [[ "$item" == *"::"* ]] || die "--must-contain 参数格式错误，应为 相对路径::关键字，收到: $item"
    rel_path="${item%%::*}"
    needle="${item#*::}"
    abs_path="$REPO_PATH/$rel_path"
    [[ -f "$abs_path" ]] || die "marker 目标文件不存在: $rel_path"
    if match_line="$(grep -Fn -- "$needle" "$abs_path" | head -n 1)"; then
        ok "关键字命中: $rel_path -> $needle | $match_line"
    else
        die "关键字未命中: $rel_path -> $needle"
    fi
done

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_PATH/runs/preflight_check"
elif [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$REPO_PATH/$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"
probe_file="$OUTPUT_DIR/.preflight_write_test.$$"
: > "$probe_file" || die "输出目录不可写: $OUTPUT_DIR"
rm -f "$probe_file"
ok "输出目录可写: $OUTPUT_DIR"

if (( SCAN_DUPLICATES )); then
    SEARCH_ROOT="$(cd -- "$REPO_PATH/../.." && pwd -P)"
    info "扫描同名仓库目录: $SEARCH_ROOT"
    while IFS= read -r found_path; do
        printf '  - %s\n' "$found_path"
    done < <(find "$SEARCH_ROOT" -maxdepth 4 -type d -name "$REPO_NAME" 2>/dev/null | sort)
fi

ok "预检通过。建议下一步使用绝对路径进入仓库，并为这次实验使用一个新的输出目录。"
