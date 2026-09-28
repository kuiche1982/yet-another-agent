import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys
os.environ["EXECUTOR_BACKEND"] = "glm"
os.environ.setdefault("GLM_API_TOKEN", "YOUR_GLM_API_TOKEN")

import demo
from swe_agent.log import logger

logger.info('%s %s', 'EXECUTOR_BACKEND =', demo.EXECUTOR_BACKEND)
logger.info('%s %s', 'GLM_API_TOKEN set:', bool(demo.GLM_API_TOKEN))

# 1) chat() 路由到 GLM：构造一个"写文件"的 agent 风格消息，看 GLM 是否返回可解析的 tool-call
sys_prompt = demo.build_system_prompt()
msgs = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": "# 任务\n在 /dev/null 不可写，请在工作区 agent_sandbox 里创建一个 hello.py，内容为一行 `print('hi')`，然后用 shell 运行它验证。"},
]
out = demo.chat(msgs, temperature=0.3, max_tokens=2048)
logger.info('%s', '===== chat() 返回（前 1200 字）=====')
logger.info('%s', out[:1200])
logger.error('%s %s', '===== 是否以 llm_error 开头 =====', out.startswith('llm_error'))
logger.info('%s %s %s %s %s', '===== GLM 调用计数 STATS =====', demo.STATS.get('glm_calls'), 'calls,', demo.STATS.get('glm_tokens'), 'tokens')
