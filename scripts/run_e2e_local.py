import sys
sys.path.insert(0, "~/kuiwork/workdir2/litertlm")
from swe_agent.supervisor import main, DEFAULT_TASK

# 全本地单循环 e2e：qwen-7B coder（rapid-mlx 本地服务）当 Executor，Planner 走本地自规划
# （--no-glm 现在把 PLANNER_MODEL/REVIEWER_MODEL 置空 → 本地自规划 + harness 评审），
# 不依赖任何外部 LLM 服务。等价于：
#   python -m swe_agent --no-glm --executor local "<DEFAULT_TASK>"
# （--executor local 经 _resolve_executor_arg 映射到 catalog 模型 qwen-2.5-coder-7b）
sys.argv = ["swe_agent", "--no-glm", "--executor", "local", DEFAULT_TASK]
main()
