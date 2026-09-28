import os
import sys

# LM Studio e2e：用 LM Studio 里的 qwen2.5.1-coder-7b-instruct 当 Executor（原生 tool_calls），
# Planner/Reviewer 仍由 glm-4.7 担任。不传 --executor，避免 legacy 映射强制成 lfm2.5；
# 直接通过 EXECUTOR_MODEL env 指向 qwen 模型（已在 models.MODELS 注册为 lmstudio provider）。
#
# 目的：验证「接手别人（弱/外部）写的代码」情景——外部/弱 executor 产出实现，
# 聪明 planner/reviewer + #84 隐藏验收闸门做客观校验，戳穿假绿（自测全绿但隐藏集挂）。
# 等价于：EXECUTOR_MODEL=qwen2.5.1-coder-7b-instruct python -m swe_agent "<DEFAULT_TASK>"
os.environ["EXECUTOR_MODEL"] = "qwen2.5.1-coder-7b-instruct"

sys.path.insert(0, "~/kuiwork/workdir2/litertlm")
from swe_agent.supervisor import main, DEFAULT_TASK

sys.argv = ["swe_agent", DEFAULT_TASK]
main()
