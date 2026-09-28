#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/dbg.py —— 逐模块 / 逐 loop 调试器（不进主流程，可独立跑每个角色与 loop）

设计目标（对齐用户 2026-08-31 调试方针）：
  1) 每个 agent 角色可单独调试：analyzer / planner / executor(单轮) / tester，
     且把【发给 lmstudio 的每条请求】原样打印，人工核对「每个期望的请求都发出了且正确」。
  2) 用 mock agent 验证 loop_1 / loop_2 / loop_3 的控制流（lint 闸门、pytest 单杠、
     tester 闸门、early-stop、漂移），不依赖真实模型。
  3) 只有模块与 loop 都验证通过，才跑 e2e（run_agent）。

用法（在仓库根目录用 envs/default 的 python 跑）：
  python -m swe_agent.dbg analyzer "任务..."
  python -m swe_agent.dbg planner  "任务..."
  python -m swe_agent.dbg executor "任务..."
  python -m swe_agent.dbg tester   "任务..."
  python -m swe_agent.dbg mock-loop
  python -m swe_agent.dbg mock-loop-earlystop
  python -m swe_agent.dbg e2e      "任务..."
"""

import os
import sys
import json
import argparse

# 确保把仓库根加入 path（本文件在 swe_agent/ 内，仓库根是父目录）
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import swe_agent.config as C
import swe_agent.models as M
from swe_agent import state
from swe_agent.state import GLOBAL_STATE, reset_state, get_current_task
from swe_agent import supervisor as S
from swe_agent import roles as R
from swe_agent import roles_config as RC
from swe_agent import verify as V
from swe_agent import harness as H
from swe_agent import plugins as _plugins
from swe_agent.agent import RunState
from swe_agent.log import logger, set_trace


def _ensure_plugins():
    """加载已配置插件（websearch / web_fetch / ocr / lsp 等），让 analyzer/executor 等角色
    能拿到注册的工具（如 web_search）。dbg 直接驱动 run_agent/run_analyzer，绕过了 supervisor.main()
    里的 load_plugins，所以这里补一次，确保逐模块调试与 e2e 下工具集与正式入口一致。"""
    if not C.ENABLE_PLUGINS:
        return
    try:
        _plugins.load_plugins(C.PLUGINS_ROOT, enable_mcp=C.ENABLE_PLUGIN_MCP)
    except Exception as e:
        logger.debug('%s', f'[plugins] 加载失败（忽略，不影响核心闭环）：{e}')


# ======================================================================
# 请求日志包装：把发给 lmstudio 的每条请求打印出来（验证「请求正确」）
# ======================================================================
_LOG = True


def _short(s, n=500):
    s = s or ""
    return s if len(s) <= n else s[:n] + f"...[+{len(s) - n}]"


def _pp_tools(tools):
    if not tools:
        return "(none)"
    return ", ".join(t.get("function", {}).get("name", t.get("name", "?")) for t in tools)


def _pp_msgs(messages, n=4):
    if not messages:
        return "  (no messages)"
    out = []
    for m in messages[-n:]:
        role = m.get("role")
        content = m.get("content")
        if content:
            out.append(f"    [{role}] {_short(content, 240)}")
        else:
            tcs = m.get("tool_calls")
            if tcs:
                out.append("    [" + role + "] tool_calls: " +
                           ", ".join(tc.get("function", {}).get("name", "?") for tc in tcs))
            else:
                out.append(f"    [{role}] (empty)")
    return "\n".join(out)


_real_tc = M.chat_toolcalls
_real_cte = M.chat_text_escalating
_real_cm = M.chat_messages


def _log_tc(role="executor", messages=None, temperature=0.3, max_tokens=None,
            tools=None, model_override=None, tool_choice=None):
    if _LOG:
        mid = model_override or M.role_model_id(role) or "(none)"
        logger.debug('%s', f'\n[REQ chat_toolcalls] role={role} model={mid} tool_choice={tool_choice}')
        logger.debug('%s', f'  tools: {_pp_tools(tools)}')
        logger.debug('%s', _pp_msgs(messages))
    return _real_tc(role=role, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, tools=tools,
                    model_override=model_override, tool_choice=tool_choice)


def _log_cte(role, system, user, temperature=0.3, max_tokens=None,
             response_format=None, require_json=False):
    if _LOG:
        mid = M.role_model_id(role) or M.role_fallback(role) or "(none)"
        logger.debug('%s', f"\n[REQ chat_text_escalating] role={role} model={mid} require_json={require_json} resp_fmt={(response_format.get('type') if response_format else None)}")
        logger.debug('%s', f'  SYSTEM: {_short(system, 600)}')
        logger.debug('%s', f'  USER:   {_short(user, 600)}')
    return _real_cte(role=role, system=system, user=user, temperature=temperature,
                     max_tokens=max_tokens, response_format=response_format,
                     require_json=require_json)


def _log_cm(role="executor", messages=None, temperature=0.3, max_tokens=None,
            tools=None, model_override=None, tool_choice=None):
    if _LOG:
        mid = model_override or M.role_model_id(role) or "(none)"
        logger.debug('%s', f'\n[REQ chat_messages(LEGACY string path)] role={role} model={mid} tool_choice={tool_choice}')
        logger.debug('%s', f'  tools: {_pp_tools(tools)}')
        logger.debug('%s', _pp_msgs(messages))
    return _real_cm(role=role, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, tools=tools,
                    model_override=model_override, tool_choice=tool_choice)


def install_logging():
    M.chat_toolcalls = _log_tc
    M.chat_text_escalating = _log_cte
    M.chat_messages = _log_cm


# ======================================================================
# 模块测试：每个角色单独跑（真实模型）
# ======================================================================
def cmd_analyzer(task: str):
    reset_state()
    logger.debug('%s', '########## ANALYZER 模块测试 ##########')
    findings = R.run_analyzer(task, max_steps=6)
    logger.debug('%s', '\n===== ANALYZER 返回（前 800 字）=====')
    logger.debug('%s', _short(findings, 800) or '（空：未产出调研）')
    logger.debug('%s', f"\n[判定] {('OK：产出调研' if findings.strip() else 'FAIL：未产出可用调研（模型不可用 / 旧路径 bug）')}")
    return findings


def cmd_planner(task: str):
    reset_state()
    logger.debug('%s', '########## PLANNER 模块测试 ##########')
    ok = R.run_planner(task)
    logger.debug('%s', f'\n[判定] run_planner 返回 {ok}')
    plan = GLOBAL_STATE.get("plan")
    logger.debug('%s', f"  language={(plan.get('language') if plan else None)} modules={(len(plan.get('modules', [])) if plan else 0)} tasks={(len(plan.get('tasks', [])) if plan else 0)} verify_points={(len(plan.get('verify_points', [])) if plan else 0)}")
    vps = GLOBAL_STATE.get("verify_points") or []
    for v in vps:
        logger.debug('%s', f"    - [{v.get('id')}] {v.get('point')}  (hint: {v.get('check_hint')})")
    return ok


def cmd_executor(task: str):
    reset_state()
    logger.debug('%s', '########## EXECUTOR 单轮（loop_3）模块测试 ##########')
    ok = R.run_planner(task)
    if not ok:
        logger.debug('%s', '[跳过] planner 失败，无法构造契约。')
        return
    cur = get_current_task()
    subtask = cur["desc"] if cur else "实现第一个任务"
    messages = [
        {"role": "system", "content": S.build_system_prompt(task, lang=GLOBAL_STATE.get("lang", "python"))},
        {"role": "user", "content": S.build_context(subtask)},
    ]
    C.MAX_STEPS = 4
    # 单轮 executor（loop_3）模块测试：统一 Agent（executor 角色 + 单步 LoopConfig）
    agent = RC.make_agent("executor", loop=RC.single_loop(max_iter=C.MAX_STEPS))
    end = agent.run(messages)
    logger.debug('%s', f'\n[判定] loop_3 结束原因={end}')
    logger.debug('%s', '===== 沙箱当前文件 =====')
    for p in sorted(C.WORKSPACE.rglob("*")):
        if p.is_file():
            logger.debug('%s', f'  - {p.relative_to(C.WORKSPACE)} ({p.stat().st_size}B)')
    return end


def cmd_tester(task: str):
    reset_state()
    logger.debug('%s', '########## TESTER（独立验收）模块测试 ##########')
    R.run_planner(task)
    verdict, detail = V.verify_gate()
    logger.debug('%s', f'\n[判定] tester verdict={verdict}')
    logger.debug('%s', f'  detail: {_short(detail, 400)}')
    return verdict


# ======================================================================
# Mock agent：验证 loop_1 / loop_2 / loop_3 控制流（不依赖真实模型）
# ======================================================================
def _tc(action: str, **kw) -> dict:
    """构造一个 chat_toolcalls 形态的返回（executor 一路 complete 推进任务）。"""
    return {
        "type": "toolcalls",
        "actions": [{"action": action, **kw}],
        "tool_calls": [{"id": "call_1", "name": action,
                        "arguments": json.dumps(kw, ensure_ascii=False)}],
    }


def cmd_mock_loop(early_stop: bool = False, empty: bool = False):
    reset_state()
    GLOBAL_STATE["lang"] = "python"
    GLOBAL_STATE["planning_done"] = True
    GLOBAL_STATE["tasks"] = [
        {"desc": "实现模块A", "deliverables": [], "status": "pending"},
        {"desc": "实现模块B", "deliverables": [], "status": "pending"},
        {"desc": "跑测试", "deliverables": [], "status": "pending"},
    ]
    GLOBAL_STATE["done_list"] = []

    transitions = []

    # —— 脚本化 harness 闸门（用调用计数驱动，适配 MAX_ATTEMPTS=3）——
    # 设计场景：attempt1 内 lint 先 fail 再 pass → pytest=no_tests；
    #          attempt2 pytest=fail；attempt3 pytest=pass → tester=pass（完成）。
    lint_n = {"n": 0}
    test_n = {"n": 0}
    verify_n = {"n": 0}

    def fake_run_lint(lang):
        lint_n["n"] += 1
        if lint_n["n"] == 1:
            transitions.append("lint=fail -> 重编码(loop_2 下一轮)")
            return ("fail", "mock: 静态校验未通过", None)
        transitions.append("lint=pass")
        return ("pass", "mock: lint ok", None)

    def fake_test_bar(messages):
        test_n["n"] += 1
        if test_n["n"] == 1:
            transitions.append("pytest=no_tests -> 下一轮 attempt")
            return ("no_tests", "mock: 无测试文件", "")
        if test_n["n"] == 2:
            transitions.append("pytest=fail -> 下一轮 attempt")
            return ("fail", "mock: 2 failed", "some traceback")
        transitions.append("pytest=pass -> 进 tester")
        return ("pass", "mock: 3 passed", "")

    def fake_verify():
        verify_n["n"] += 1
        if verify_n["n"] == 1:
            transitions.append("tester=pass -> Agent 完成")
            return ("pass", "mock: 全部验收点通过")
        transitions.append("tester=pass -> Agent 完成")
        return ("pass", "mock: 全部验收点通过")

    # —— mock 模型：executor 一路 complete；early_stop 直接返回 None（硬崩溃）；
    #    empty 模式返回软空响应 {"type":"empty"}（server 正常但模型无 tool call 且无 content）——
    #    新语义：空响应不跳出 loop_3，而是计空响应次数逼近 LOOP_REPEAT_THRESHOLD 后判 stuck，
    #    照常走 lint → pytest → tester（不 early-stop、不抛外层闸门）。
    if early_stop:
        def fake_tc(role="executor", messages=None, tools=None, **kw):
            transitions.append("chat_toolcalls=None (模型连续不可用) -> early_stop")
            return None
    elif empty:
        def fake_tc(role="executor", messages=None, tools=None, **kw):
            transitions.append("chat_toolcalls={'type':'empty'} (软空响应) -> 计空响应计数")
            return {"type": "empty"}
    else:
        def fake_tc(role="executor", messages=None, tools=None, **kw):
            return _tc("complete")

    # 注入 mock
    M.chat_toolcalls = fake_tc
    H.run_lint = fake_run_lint
    H._run_test_bar = fake_test_bar
    V.verify_gate = fake_verify
    M.chat_text_escalating = lambda *a, **k: "mock"
    M.chat_messages = lambda *a, **k: ""

    ctx = RunState(max_iter=C.MAX_ITER)
    messages = [{"role": "system", "content": "mock"},
                {"role": "user", "content": "mock task"}]

    logger.debug('%s', '########## MOCK LOOP 控制流验证 ##########')
    logger.debug('%s', f'MAX_ATTEMPTS={C.MAX_ATTEMPTS} MAX_ROUNDS={C.MAX_ROUNDS} MAX_STEPS={C.MAX_STEPS}')
    S._run_nested_loops(messages, ctx)

    logger.debug('%s', '\n===== 观察到的控制流转移 =====')
    for i, t in enumerate(transitions, 1):
        logger.debug('%s', f'  {i}. {t}')
    logger.debug('%s', f'\nctx.early_stop = {ctx.early_stop}')
    if early_stop:
        ok = ctx.early_stop is True
        logger.debug('%s', f"[判定] early_stop 场景: {('OK' if ok else 'FAIL')}（应触发提前终止）")
    elif empty:
        # 软空响应场景：executor 每步都返回 {"type":"empty"}，应「不 early-stop、不抛外层」，
        # 经 counter/fence 收敛为 stuck 后照常走 lint → pytest → tester，最终 tester=pass 收尾。
        ok = (ctx.early_stop is False and test_n["n"] == 3 and verify_n["n"] == 1
              and transitions[-1].startswith("tester=pass"))
        logger.debug('%s', f"[判定] 软空响应场景(不 early-stop、counter/fence→stuck、3 attempt 内 tester=pass): {('OK' if ok else 'FAIL')}")
        logger.debug('%s', f"  lint调用={lint_n['n']} pytest调用={test_n['n']} tester调用={verify_n['n']}")
    else:
        # 期望：lint fail 一次后 pass；pytest 依次 no_tests→fail→pass；tester=pass 收尾。
        ok = (lint_n["n"] >= 2 and test_n["n"] == 3 and verify_n["n"] == 1
              and transitions[-1].startswith("tester=pass"))
        logger.debug('%s', f"[判定] 正常场景(3 attempt 内走到 tester=pass): {('OK' if ok else 'FAIL')}")
        logger.debug('%s', f"  lint调用={lint_n['n']} pytest调用={test_n['n']} tester调用={verify_n['n']}")
    return ok


# ======================================================================
# e2e（真实模型，带任务参数）
# ======================================================================
def cmd_e2e(task: str):
    reset_state()
    logger.debug('%s', '########## E2E run_agent ##########')
    S.run_agent(task)
    logger.debug('%s', '\n[E2E 结束]')


# ======================================================================
# CLI
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="litertlm 逐模块调试器")
    ap.add_argument("cmd", choices=[
        "analyzer", "planner", "executor", "tester",
        "mock-loop", "mock-loop-earlystop", "mock-loop-empty", "e2e"])
    ap.add_argument("task", nargs="*", help="开发任务描述（模块测试/e2e 用）")
    ap.add_argument("--no-log", action="store_true", help="不打印请求明细")
    args = ap.parse_args()

    # dbg 是调试器：默认打开详细追踪（logger.debug），让请求明细/分析结果在控制台可见
    set_trace(True)

    global _LOG
    if args.no_log:
        _LOG = False
    else:
        install_logging()

    # 优先级：命令行显式传入的 task > config.DEFAULT_TASK > 内置兜底任务。
    # ⚠️ 曾写成 `A or B if cond else C`，被 Python 解析为 `(A or B) if cond else C`：
    # 当 config 没有 DEFAULT_TASK 属性时（该常量实际定义在 supervisor），
    # 命令行传入的任务被整段丢弃、静默换成兜底任务 —— 调试器看到的不是被测任务，
    # 单模块复现结论因此失真。显式传参必须优先。
    task = " ".join(args.task or []) or (
        C.DEFAULT_TASK if hasattr(C, "DEFAULT_TASK") else
        ("开发命令行版的康威生命游戏（Conway's Game of Life），默认 20x20，"
         "实现 GameOfLife 类（step/display）与 CLI 入口，写 test_game_of_life.py 用 pytest 通过。"))

    # dbg 直接驱动各角色 / run_agent，绕过了 supervisor.main() 的插件加载，这里补一次，
    # 保证 analyzer 能拿到 web_search 等插件工具（与正式入口行为一致）。
    _ensure_plugins()

    if args.cmd == "analyzer":
        cmd_analyzer(task)
    elif args.cmd == "planner":
        cmd_planner(task)
    elif args.cmd == "executor":
        cmd_executor(task)
    elif args.cmd == "tester":
        cmd_tester(task)
    elif args.cmd == "mock-loop":
        cmd_mock_loop(early_stop=False)
    elif args.cmd == "mock-loop-earlystop":
        cmd_mock_loop(early_stop=True)
    elif args.cmd == "mock-loop-empty":
        cmd_mock_loop(empty=True)
    elif args.cmd == "e2e":
        cmd_e2e(task)


if __name__ == "__main__":
    main()
