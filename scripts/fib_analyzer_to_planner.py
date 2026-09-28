#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实链路验证：把实际 fib analyzer 输出（取自 /tmp/analyzer_ab.log B版）作为
research_findings 注入 Planner（harness 原生 _make_plan 通道），看 json_schema 结构化输出效果。
对比基线：planner 直接吃原始需求（/tmp/planner_fib_plan.json）。
"""
import os, sys, json, logging
REPO = "~/kuiwork/workdir2/litertlm"
os.chdir(REPO); sys.path.insert(0, REPO)
logging.disable(logging.WARNING)

from swe_agent import dbg
from swe_agent.state import GLOBAL_STATE, reset_state
from swe_agent import roles as R, models as M
from swe_agent.log import logger

dbg._ensure_plugins()
logger.info('%s %s', '[setup] planner model =', M.role_model_id('planner'))

# 真实 fib analyzer 输出（B版，/tmp/analyzer_ab.log 第65行）
ANALYZER_OUTPUT = (
    "斐波那契数列的计算规则为F(0)=0、F(1)=1、F(n)=F(n-1)+F(n-2)。"
    "常见的边界约定是F(0)和F(1)是基准值，对于负数索引通常不定义。"
    "接口形状为函数fibonacci，参数n（int），返回值为斐波那契数列的第n个数。"
    "测试约定是输入非负整数，输出对应的斐波那契数列值。"
    "潜在坑点是负数索引未定义和大数计算时递归方法可能导致栈溢出。"
)

task = "用 Python 实现斐波那契数列（fibonacci）"

# ---- 真实链路：analyzer 输出 -> planner ----
reset_state()
ok = R.run_planner(task, research_findings=ANALYZER_OUTPUT)
logger.info('%s %s', '\n[run_planner(带analyzer) 返回]', ok)
plan = GLOBAL_STATE.get("plan")
with open("/tmp/fib_plan_from_analyzer.json", "w", encoding="utf-8") as f:
    json.dump(plan, f, ensure_ascii=False, indent=2)
logger.info('%s', '[plan 已存] /tmp/fib_plan_from_analyzer.json')

# ---- 验证结构化输出质量 ----
vps = plan.get("verify_points") or []
logger.info('%s', f'\n[verify_points 数] {len(vps)}')
for v in vps:
    logger.info('%s', f"  - [{v.get('id')}] {v.get('point')}  (hint: {v.get('check_hint')})")

bad = []
for m in plan.get("modules", []):
    if "tests/" in (m.get("path") or ""): bad.append(("module", m["path"]))
for t in plan.get("tasks", []):
    for d in (t.get("deliverables") or []):
        if "tests/" in d: bad.append(("deliverable", d))
logger.info('%s %s', '\n[tests/ 子目录检查] BAD =', bad if bad else '无 ✓')
logger.info('%s %s', '[tasks 步数]', len(plan.get('tasks', [])))
logger.info('%s %s', '[interface]', plan.get('interface'))
logger.info('%s %s', '[modules]', [(m.get('path'), m.get('public')) for m in plan.get('modules', [])])

# ---- 对照基线：planner 直接吃原始需求 ----
try:
    with open("/tmp/planner_fib_plan.json", encoding="utf-8") as f:
        base = json.load(f)
    logger.info('%s %s', '\n[基线对比] 原始需求直喂 planner 的 verify_points 数 =', len(base.get('verify_points') or []))
    logger.info('%s %s', '[基线对比] 原始需求直喂 planner 的 tasks 步数 =', len(base.get('tasks', [])))
except FileNotFoundError:
    logger.info('%s', '\n[基线对比] /tmp/planner_fib_plan.json 不存在，跳过对照（之前未落盘）')
