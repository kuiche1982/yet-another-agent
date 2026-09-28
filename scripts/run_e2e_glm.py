import sys
sys.path.insert(0, "~/kuiwork/workdir2/litertlm")
from swe_agent.supervisor import main, DEFAULT_TASK

# GLM 全链路 e2e：GLM 既当 Planner/Reviewer 又当 Executor（默认 glm-4.7，原生 tool_calls；
#   GLM_MODEL=glm-4.5-flash 等可回退到其它已登记 zhipu 模型）。
# 等价于：python -m swe_agent --executor glm "<DEFAULT_TASK>"
# （--executor glm 经 _resolve_executor_arg 映射到 catalog 模型 glm-4.7；
#  PLANNER_MODEL/REVIEWER_MODEL 默认即 glm-4.7，无需额外开关）
sys.argv = ["swe_agent", "--executor", "glm", DEFAULT_TASK]
main()
