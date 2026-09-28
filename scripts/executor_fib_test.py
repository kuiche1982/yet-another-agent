#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""executor 单 role 测试：planner 建 fib 契约后，executor（qwen7b + 当前 SYSTEM_PROMPT）写码 + pytest。
跑完用真实 pytest 验收绿/红，并列出生成文件。"""
import os, sys, logging, subprocess
REPO = "~/kuiwork/workdir2/litertlm"
os.chdir(REPO); sys.path.insert(0, REPO)
logging.disable(logging.WARNING)

from swe_agent import dbg
from swe_agent.state import GLOBAL_STATE, reset_state, get_current_task
from swe_agent import roles as R, supervisor as S, roles_config as RC, config as C
from swe_agent.log import logger

dbg._ensure_plugins()
reset_state()
task = "用 Python 实现斐波那契数列（fibonacci）"
ok = R.run_planner(task)
logger.info('%s %s', '[planner ok]', ok)

cur = get_current_task()
subtask = cur["desc"] if cur else "实现第一个任务"
messages = [
    {"role": "system", "content": S.build_system_prompt(task, lang=GLOBAL_STATE.get("lang", "python"))},
    {"role": "user", "content": S.build_context(subtask)},
]
C.MAX_STEPS = 14
agent = RC.make_agent("executor", loop=RC.single_loop(max_iter=C.MAX_STEPS))
end = agent.run(messages)
logger.info('%s %s', '\n[executor END reason]', end)

ws = C.WORKSPACE
logger.info('%s', '\n[生成文件]')
for p in sorted(ws.rglob("*")):
    if p.is_file():
        logger.info('%s', f'  - {p.relative_to(ws)} ({p.stat().st_size}B)')

logger.info('%s', '\n[真实 pytest 验收]')
r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(ws),
                   capture_output=True, text=True)
logger.info('%s %s', '  returncode:', r.returncode)
logger.info('%s %s', '  stdout:', r.stdout[-1800:])
logger.info('%s %s', '  stderr:', r.stderr[-600:])
