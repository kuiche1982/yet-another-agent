#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""planner 单 role 测试：用当前落盘的 _PLANNER_SYSTEM 对 fib 生成契约，验证质量。"""
import os, sys, json, logging
REPO = "~/kuiwork/workdir2/litertlm"
os.chdir(REPO); sys.path.insert(0, REPO)
logging.disable(logging.WARNING)

from swe_agent import dbg
from swe_agent.state import GLOBAL_STATE, reset_state
from swe_agent import roles as R, models as M

dbg._ensure_plugins()
logger.info('%s %s', '[setup] planner model =', M.role_model_id('planner'))

reset_state()
task = "用 Python 实现斐波那契数列（fibonacci）"
ok = R.run_planner(task)
logger.info('%s %s', '\n[run_planner 返回]', ok)

plan = GLOBAL_STATE.get("plan")
with open("/tmp/planner_fib_plan.json", "w", encoding="utf-8") as f:
    json.dump(plan, f, ensure_ascii=False, indent=2)
logger.info('%s', '\n[plan 已存] /tmp/planner_fib_plan.json')
vps = plan.get("verify_points") or []
logger.info('%s', f'[verify_points 数] {len(vps)}')
for v in vps:
    logger.info('%s', f"  - [{v.get('id')}] {v.get('point')}  (hint: {v.get('check_hint')})")
logger.info('%s', '[测试文件路径检查] modules/deliverables 是否含 tests/ 子目录：')
import re
from swe_agent.log import logger
bad = []
for m in plan.get("modules", []):
    if "tests/" in (m.get("path") or ""): bad.append(("module", m["path"]))
for t in plan.get("tasks", []):
    for d in (t.get("deliverables") or []):
        if "tests/" in d: bad.append(("deliverable", d))
logger.info('%s %s', '  BAD(tests/ 子目录):', bad if bad else '无 ✓')
