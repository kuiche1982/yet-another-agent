# guess,fib,fizzbuzz,conway
cd ~/kuiwork/workdir2/litertlm 
rm -rf logs/e2e_battery
rm logs/harness.log
rm logs/lmstudio_requests.jsonl
rm -rf agent_sandbox/*

LMSTUDIO_DUMP=1 \
LOGLEVEL=DEBUG \
SWE_MODEL_LOAD_UNLOAD=0 \
PLANNER_MODEL=lfm2.5-2.6b ANALYZER_MODEL=lfm2.5-2.6b EXECUTOR_MODEL=lfm2.5-2.6b \
TESTER_MODEL=lfm2.5-2.6b PLANNER_FALLBACK_MODEL=lfm2.5-2.6b ANALYZER_FALLBACK_MODEL=lfm2.5-2.6b \
SIDECAR_MODEL=lfm2.5-2.6b SIDECAR_COMPRESS_MODEL=lfm2.5-2.6b \
JUDGE_MODEL=lfm2.5-2.6b \
uv run --no-sync python run_e2e_battery.py --tasks fib --timeout 600

uv run --no-sync python run_e2e_battery.py --tasks guess,fib,fizzbuzz,conway --timeout 600

LMSTUDIO_DUMP=1 \
LOGLEVEL=DEBUG \
SWE_MODEL_LOAD_UNLOAD=0 \
PLANNER_MODEL=Ling-3.0-Tiny  \
ANALYZER_MODEL=Ling-3.0-Tiny  \
EXECUTOR_MODEL=Ling-3.0-Tiny  \
TESTER_MODEL=Ling-3.0-Tiny \
TESTER_MODEL=Ling-3.0-Tiny  \
PLANNER_FALLBACK_MODEL=Ling-3.0-Tiny  \
ANALYZER_FALLBACK_MODEL=Ling-3.0-Tiny \
SIDECAR_MODEL=Ling-3.0-Tiny  \
SIDECAR_COMPRESS_MODEL=Ling-3.0-Tiny \
JUDGE_MODEL=Ling-3.0-Tiny \
uv run --no-sync python run_e2e_battery.py --tasks fib --timeout 1800


MODEL_CONTEXT_LENGTH=32000 \
PYTHONUNBUFFERED=1 \
LMSTUDIO_DUMP=1 \
LOGLEVEL=DEBUG \
SWE_MODEL_LOAD_UNLOAD=0 \
PLANNER_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
ANALYZER_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
EXECUTOR_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
TESTER_MODEL=qwen3.5-4b-mtplx-optimized-speed \
TESTER_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
PLANNER_FALLBACK_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
ANALYZER_FALLBACK_MODEL=qwen3.5-4b-mtplx-optimized-speed \
SIDECAR_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
SIDECAR_COMPRESS_MODEL=qwen3.5-4b-mtplx-optimized-speed \
JUDGE_MODEL=qwen3.5-4b-mtplx-optimized-speed \
uv run --no-sync python run_e2e_battery.py --tasks fib --timeout 1800

LMSTUDIO_DUMP=1 \
LOGLEVEL=DEBUG \
SWE_MODEL_LOAD_UNLOAD=0 \
PLANNER_MODEL=Spark-X2.5-1.7B \
ANALYZER_MODEL=Spark-X2.5-1.7B \
EXECUTOR_MODEL=Spark-X2.5-1.7B \
TESTER_MODEL=Spark-X2.5-1.7B \
PLANNER_FALLBACK_MODEL=Spark-X2.5-1.7B \
ANALYZER_FALLBACK_MODEL=Spark-X2.5-1.7B \
SIDECAR_MODEL=Spark-X2.5-1.7B \
SIDECAR_COMPRESS_MODEL=Spark-X2.5-1.7B \
JUDGE_MODEL=Spark-X2.5-1.7B \
uv run --no-sync python run_e2e_battery.py --tasks fib --timeout 600


# child_env = dict(env)
#     child_env["PYTHONUNBUFFERED"] = "1"
#     # 把隔离目录透传给 harness（config.WORKSPACE 在 import 时读取），并关闭外层重复 reset
#     child_env["SWE_WORKSPACE"] = str(task_ws)
#     child_env["SWE_FRESH_WORKSPACE"] = "1"
#     cmd = ["uv", "run", "--no-sync", "python", "-m", "swe_agent", task]

SWE_FRESH_WORKSPACE=0 \
SWE_WORKSPACE=~/kuiwork/workdir2/litertlm/agent_sandbox \
MODEL_CONTEXT_LENGTH=32000 \
PYTHONUNBUFFERED=1 \
LMSTUDIO_DUMP=1 \
LOGLEVEL=DEBUG \
SWE_MODEL_LOAD_UNLOAD=0 \
PLANNER_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
ANALYZER_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
EXECUTOR_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
TESTER_MODEL=qwen3.5-4b-mtplx-optimized-speed \
TESTER_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
PLANNER_FALLBACK_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
ANALYZER_FALLBACK_MODEL=qwen3.5-4b-mtplx-optimized-speed \
SIDECAR_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
SIDECAR_COMPRESS_MODEL=qwen3.5-4b-mtplx-optimized-speed \
JUDGE_MODEL=qwen3.5-4b-mtplx-optimized-speed \
uv run python -m swe_agent '用 Python 实现斐波那契, 并测试'


SWE_WORKSPACE=~/kuiwork/workdir2/litertlm/agent_sandbox \
MODEL_CONTEXT_LENGTH=64000 \
PYTHONUNBUFFERED=1 \
LMSTUDIO_DUMP=1 \
LOGLEVEL=DEBUG \
SWE_MODEL_LOAD_UNLOAD=0 \
PLANNER_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
ANALYZER_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
EXECUTOR_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
TESTER_MODEL=qwen3.5-4b-mtplx-optimized-speed \
TESTER_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
PLANNER_FALLBACK_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
ANALYZER_FALLBACK_MODEL=qwen3.5-4b-mtplx-optimized-speed \
SIDECAR_MODEL=qwen3.5-4b-mtplx-optimized-speed  \
SIDECAR_COMPRESS_MODEL=qwen3.5-4b-mtplx-optimized-speed \
JUDGE_MODEL=qwen3.5-4b-mtplx-optimized-speed \
uv run python -m swe_agent


'analysis.md' 有些写的不是事实， 帮我分析当前项目并修正文档中的错误论述
