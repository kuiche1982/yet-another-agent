#!/usr/bin/env bash
# 重跑 Go 基准：glm-4.1v-thinking-flashx(planner/reviewer) + qwen-lm-tjson(executor)
set -u
cd ~/kuiwork/workdir2/litertlm
set -a; . ./.env; set +a
export PLANNER_MODEL=glm-4.1v-thinking-flashx
export REVIEWER_MODEL=glm-4.1v-thinking-flashx
export EXECUTOR_MODEL=qwen-lm-tjson
export LMSTUDIO_BASE_URL=http://localhost:1234/v1
export LMSTUDIO_API_KEY=lm-studio
PY=.venv/bin/python

TS=$(date +%Y%m%d-%H%M%S)
LOGDIR=/tmp/swe_go_bench_$TS
mkdir -p "$LOGDIR"
echo "$LOGDIR" > /tmp/swe_go_bench_last.txt

run_one () {
  local task="$1"; local name="$2"
  local log="$LOGDIR/$name.log"
  echo "[bench] $(date) START :: $task" | tee -a "$log"
  # 干净沙箱：每任务独立
  rm -rf agent_sandbox && mkdir -p agent_sandbox
  $PY -m swe_agent "$task" < /dev/null >> "$log" 2>&1
  echo "[bench] $(date) END   :: $task (exit=$?)" | tee -a "$log"
}

run_one "golang 开发 命令行版 康威生命游戏" conway
run_one "golang开发命令行版todo list" todo

echo "[bench] ALL DONE -> $LOGDIR" | tee -a "$LOGDIR/_done.log"
