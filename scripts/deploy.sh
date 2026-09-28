#!/bin/sh
# deploy.sh —— 把 swe_agent 安装到 SWE_AGENT_HOME（默认 ~/.swe_agent）
#
# 流程：backup -> delete -> install
#   backup : mv "$HOME_DIR" "$HOME_DIR.bak.<时间戳>"
#            用改名而非复制 —— 原子操作、零额外磁盘占用、原数据原封不动只是换名，
#            恢复只需把名字改回来。索引目录动辄几百 MB，复制一份代价不合理。
#   delete : 删除 $HOME_DIR。此步之前强制校验备份确实存在，否则立即退出。
#            （mv 生效后原路径本就空了，这一段是防御：万一 mv 未生效仍按顺序删除）
#   install: 建目录骨架 + 写 bin/swe-agent 启动器 + 打印 PATH 提示
#
# ⚠️ delete 会移除 $SWE_AGENT_HOME 下的全部内容，**包括 sessions/（对话记录）
#    与各项目的 KB 索引**。备份是唯一防线；备份失败时脚本立即退出，绝不继续删除。
#
# 用法：
#   scripts/deploy.sh                 # 打印将要执行的动作，确认后安装
#   scripts/deploy.sh --yes           # 跳过确认
#   SWE_AGENT_HOME=/some/path scripts/deploy.sh
#
set -eu

usage() {
    cat <<'USAGE'
usage: deploy.sh [--yes] [-h]

  --yes      跳过交互确认
  -h, --help 显示本帮助

安装位置取环境变量 SWE_AGENT_HOME，默认 ~/.swe_agent。
USAGE
}

REPO="$(cd "$(dirname "$0")/.." && pwd)"

HOME_DIR="${SWE_AGENT_HOME:-${HOME:-}/.swe_agent}"
if [ -z "${SWE_AGENT_HOME:-}" ] && [ -z "${HOME:-}" ]; then
    echo "!! 无法确定安装位置：\$HOME 与 \$SWE_AGENT_HOME 均为空" >&2
    exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
BAK="$HOME_DIR.bak.$TS"
ASSUME_YES=0

for arg in "$@"; do
    case "$arg" in
        --yes|-y)   ASSUME_YES=1 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "!! 未知参数：$arg" >&2; usage >&2; exit 2 ;;
    esac
done

PY="$REPO/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "!! 未找到项目 venv python：$PY" >&2
    echo "   先执行：cd \"$REPO\" && uv sync --all-groups" >&2
    exit 1
fi

echo "SWE_AGENT_HOME = $HOME_DIR"
echo "repo           = $REPO"
echo "将执行："
if [ -e "$HOME_DIR" ]; then
    echo "  1) backup : mv $HOME_DIR -> $BAK"
else
    echo "  1) backup : （跳过，$HOME_DIR 不存在）"
fi
echo "  2) delete : rm -rf $HOME_DIR"
echo "  3) install: 建 bin/ global_kb_raw/ projects/，写 bin/swe-agent"

if [ "$ASSUME_YES" != 1 ]; then
    printf '继续？ [y/N] '
    read -r ans || ans=""
    case "$ans" in y|Y) ;; *) echo "已取消，未做任何改动。"; exit 1 ;; esac
fi

# ---- 1) backup ----
if [ -e "$HOME_DIR" ]; then
    if [ -e "$BAK" ]; then
        echo "!! 备份目标已存在：$BAK" >&2
        exit 1
    fi
    mv "$HOME_DIR" "$BAK"
    if [ ! -e "$BAK" ]; then
        echo "!! 备份失败，中止（未删除任何数据）" >&2
        exit 1
    fi
    echo "[deploy] backup -> $BAK"
fi

# ---- 2) delete ----
if [ -e "$HOME_DIR" ]; then
    rm -rf "$HOME_DIR"
    echo "[deploy] deleted $HOME_DIR"
fi

# ---- 3) install ----
mkdir -p "$HOME_DIR/bin" "$HOME_DIR/global_kb_raw" "$HOME_DIR/projects"

cat > "$HOME_DIR/bin/swe-agent" <<EOF
#!/bin/sh
# 由 scripts/deploy.sh 生成（勿手改，重装会覆盖）
# 关键：这里【不能 cd】—— WORKSPACE 取「启动时的 cwd」，cd 会把工作区推导打偏。
# 只把仓库根加进 PYTHONPATH，让 -m swe_agent 能 import，cwd 保持用户所在目录。
exec env PYTHONPATH="$REPO" "$PY" -m swe_agent "\$@"
EOF
chmod +x "$HOME_DIR/bin/swe-agent"
echo "[deploy] installed $HOME_DIR/bin/swe-agent"

case ":$PATH:" in
    *":$HOME_DIR/bin:"*) ;;
    *)
        echo
        echo "把入口加入 PATH（写入你的 shell 配置后重开终端）："
        echo "  export PATH=\"$HOME_DIR/bin:\$PATH\""
        ;;
esac

echo
echo "[deploy] 完成。"
echo "  global 层文档投放目录：$HOME_DIR/global_kb_raw"
echo "  项目级文档请放各自项目的 docs/ 或 KnowledgeBase/"

# 旧布局遗留（不自动清理，避免误删仓库内数据）：确认无用后手动删除。
OLD=""
for d in "$REPO/.rag_cache" "$REPO/sessions" "$REPO/logs"; do
    [ -e "$d" ] && OLD="$OLD $d"
done
if [ -n "$OLD" ]; then
    echo
    echo "检测到旧布局遗留（新版不再读写，确认无用后可手动清理）："
    for d in $OLD; do echo "  $d"; done
fi
