#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_multirole_verify.py —— #75 定向实跑验证脚本（康威生命游戏）

目的（5xWhy 闭环）：
  H1: 证真/证伪「lmstudio json_schema 路径可用」—— planner 对 lmstudio 下发 json_schema，
      且 lmstudio 接受并返回合法可解析的执行契约（非 400 / 非升级兜底）。
  H2: 证真/证伪「analyzer glm-4.7 升级链路」—— 本地弱模型(lfm2.5) 未产出可用调研结论时，
      升级子 agent 走 glm-4.7（zhipu remote_openai），而非静默降级为空。

配置（来自 .env，已就绪）：
  analyzer/planner/reviewer = liquid/lfm2.5-1.2b（lmstudio, 本地）
  executor                  = qwen2.5.1-coder-7b-instruct（lmstudio, 本地）
  *_FALLBACK_MODEL          = glm-4.7（仅主模型失败触发，不空耗远程配额）

运行：
  MAX_ITER=12 .venv/bin/python /tmp/run_multirole_verify.py
"""

import os
import sys
import json
import shutil
from pathlib import Path

# ---- 仓库根与解释器：脚本放到 /tmp，但包解析依赖仓库根 ----
REPO = Path("~/kuiwork/workdir2/litertlm")
sys.path.insert(0, str(REPO))

import swe_agent.config as config

# ---- 隔离工作区（/tmp，非个人目录，安全可清）----
WS = Path("/tmp/verify_sandbox")
if WS.exists():
    shutil.rmtree(WS)
WS.mkdir(parents=True, exist_ok=True)
config.WORKSPACE = WS

# ---- MAX_ITER（默认 12，可被 env 覆盖）----
config.MAX_ITER = int(os.environ.get("MAX_ITER", "12"))

from swe_agent import supervisor
from swe_agent import models as M
from swe_agent.log import logger

# ======================================================================
# 调用级探针：记录每个角色调用实际下发的 model / provider / response_format
# （这是判定 H1 / H2 的硬证据，比日志 grep 更可靠）
# ======================================================================
CALLS = []
_orig_text = M.chat_text
_orig_msgs = M.chat_messages


def _wrap_text(role, system, user, **kw):
    mid = kw.get("model_override") or M.role_model_id(role) or "(none)"
    spec = M.MODELS.get(mid, {})
    rf = kw.get("response_format")
    CALLS.append({
        "fn": "chat_text", "role": role, "model": mid,
        "provider": spec.get("provider"),
        "response_format_type": (rf or {}).get("type") if isinstance(rf, dict) else None,
        "overridden": bool(kw.get("model_override")),
    })
    return _orig_text(role, system, user, **kw)


def _wrap_msgs(role="executor", messages=None, **kw):
    mid = kw.get("model_override") or M.role_model_id(role) or "(none)"
    spec = M.MODELS.get(mid, {})
    CALLS.append({
        "fn": "chat_messages", "role": role, "model": mid,
        "provider": spec.get("provider"),
        "overridden": bool(kw.get("model_override")),
    })
    return _orig_msgs(role, messages, **kw)


M.chat_text = _wrap_text
M.chat_messages = _wrap_msgs

# ======================================================================
logger.info('%s', '=' * 64)
logger.info('%s', "MULTIROLE VERIFY — Conway's Game of Life (directed real-run)")
logger.info('%s', M.describe_config())
logger.info('%s', f'MAX_ITER={config.MAX_ITER}  WORKSPACE={config.WORKSPACE}')
logger.info('%s', '=' * 64)

# 跑完整的「analyzer → planner → executor loop → L3 单杠校验」
try:
    supervisor.run_agent(supervisor.DEFAULT_TASK)
except Exception as e:
    logger.info('%s', f'\n[verify] run_agent 抛异常（仍汇总已收集证据）：{type(e).__name__}: {e}')

# ======================================================================
# 汇总判定
# ======================================================================
plan = supervisor.GLOBAL_STATE.get("plan") or {}
fv = supervisor.GLOBAL_STATE.get("final_validation") or {}
telemetry = list(supervisor.RUN_TELEMETRY)

planner_calls = [c for c in CALLS if c["role"] == "planner"]
analyzer_calls = [c for c in CALLS if c["role"] == "analyzer"]

# H1: planner 主调用（非升级）对 lmstudio 下发 json_schema 且契约成功生成
h1_primary = [c for c in planner_calls if not c["overridden"]]
h1_json_schema_sent = any(c["provider"] == "lmstudio" and c["response_format_type"] == "json_schema"
                          for c in h1_primary)
h1_plan_ok = bool(plan.get("tasks"))
h1 = h1_json_schema_sent and h1_plan_ok

# H2: analyzer 升级子 agent 命中 glm-4.7
h2_calls = [c for c in analyzer_calls if c["overridden"] and c["model"] == "glm-4.7"]
h2 = len(h2_calls) > 0

summary = {
    "max_iter": config.MAX_ITER,
    "planner_model": M.role_model_id("planner"),
    "analyzer_model": M.role_model_id("analyzer"),
    "executor_model": M.role_model_id("executor"),
    "analyzer_fallback_model": M.role_fallback("analyzer"),
    "planner_calls": len(planner_calls),
    "analyzer_calls": len(analyzer_calls),
    "H1_lmstudio_json_schema_sent": h1_json_schema_sent,
    "H1_plan_generated": h1_plan_ok,
    "H1_result": "PROVEN" if h1 else "REFUTED",
    "H2_analyzer_glm47_upgrade_calls": len(h2_calls),
    "H2_result": "PROVEN" if h2 else "NOT_TRIGGERED",
    "final_validation_passed": fv.get("passed"),
    "done_tasks": len(supervisor.GLOBAL_STATE.get("done_list", [])),
    "total_tasks": len(supervisor.GLOBAL_STATE.get("tasks", [])),
    "telemetry": telemetry,
    "plan_summary": plan.get("summary"),
    "plan_modules": [m.get("path") for m in plan.get("modules", [])],
}

logger.info('%s', '\n' + '=' * 64)
logger.info('%s', 'VERIFY SUMMARY')
logger.info('%s', '=' * 64)
logger.info('%s', json.dumps(summary, ensure_ascii=False, indent=2))

# 关键证据行（一眼可辨）
logger.info('%s', '\n--- H1: lmstudio json_schema 路径 ---')
logger.info('%s', f'  planner 主调用对 lmstudio 下发 json_schema: {h1_json_schema_sent}')
logger.info('%s', f"  契约成功生成(有 tasks): {h1_plan_ok}  ->  H1 = {summary['H1_result']}")
logger.info('%s', '--- H2: analyzer glm-4.7 升级链路 ---')
logger.info('%s', f"  analyzer 升级到 glm-4.7 的调用次数: {len(h2_calls)}  ->  H2 = {summary['H2_result']}")

result_path = Path("/tmp/verify_result.json")
result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
logger.info('%s', f'\n[verify] 结果已写入 {result_path}')
