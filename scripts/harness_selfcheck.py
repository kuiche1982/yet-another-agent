#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harness_selfcheck.py —— 离线契约自检（无需 LLM / 网络，<1s）

为什么需要它：e2e 是「潜观测」——只在 20 分钟超时后才给 pass/fail，且任何中途控制流
bug（装饰器挂错函数 / 循环不终止）都表现为「静默空转 + 一坨重复日志」，没有任何一行指出
根因。本脚本把 harness 的两条核心契约编码成可离线执行的断言，让「契约被违反」在
import / dispatch 阶段就被点名，而不是等跑完 e2e 才发现。

检查项（只做「不会误报」的硬契约，避免对跨模块/插件注册的工具做脆弱的全集比对）：
  1) 核心工具注册 + 行为冒烟（Bug A 类）：write_file / read_file / edit_file / shell /
     grep / glob / ask 必须已注册；write_file / read_file 真实 dispatch 到临时 WORKSPACE
     能落盘/回读（executor 写不出文件会在这一步立刻暴露）。
  2) 装饰器隔离（Bug A 类）：
     - 每个核心工具必须有 @tool(name=...) 声明（防「装饰器被整段删掉」）；
     - 任何公开工具（name 不以 '_' 开头）不得由 '_' 私有函数实现，除非显式白名单
       （精确命中「write_file 的 @tool 被错挂到 _write_verify」这类 bug，且不会误杀
       shell→exec_shell 这种合法改名）；
     - 内部 helper（_write_verify 等）绝不能作为对外工具名泄露。
  3) 循环终止契约（Bug B 类）：tester 必须有 on_iter_end 闸门且对 "all_done" 返回 "done"；
     带 stop_actions 的角色，RoleConfig 必须声明 stop_actions。

用法：uv run python scripts/harness_selfcheck.py
返回 0 = 全部通过；非 0 = 有契约被破坏（并打印具体哪条）。
"""
import os
import sys
import ast
import tempfile
from pathlib import Path
from swe_agent.log import logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FAILS = []


def check(name, ok, detail=""):
    logger.info('%s %s %s', '✅' if ok else '❌', name, '— ' + detail if detail else '')
    if not ok:
        FAILS.append(name)


def main():
    import swe_agent.tools as T  # 触发 tools.py 全部 @tool 注册
    try:
        import swe_agent.supervisor as SUP  # 触发 meta 工具（plan/complete/report/todo_*/...）注册
    except Exception as e:
        check("harness 可导入（supervisor 导入成功）", False, f"导入失败：{e}")
        raise SystemExit(1)
    import swe_agent.roles_config as RC
    import swe_agent.verify as V
    from swe_agent.registry import ToolRegistry

    # ---------- 1) 核心工具注册 + 行为冒烟（Bug A）----------
    CORE_TOOLS = ["write_file", "read_file", "edit_file", "shell", "grep", "glob", "ask"]
    for name in CORE_TOOLS:
        check(f"核心工具 [{name}] 已注册", ToolRegistry.get(name) is not None)

    # 行为冒烟：把 WORKSPACE 指到临时目录，避免污染真实 agent_sandbox
    tmp = Path(tempfile.mkdtemp(prefix="selfcheck_"))
    orig_ws = T.WORKSPACE
    T.WORKSPACE = tmp
    try:
        p = tmp / "probe.txt"
        wres = ToolRegistry.dispatch(
            {"action": "write_file", "path": "probe.txt", "content": "hello-selfcheck"})
        wrote = p.exists() and p.read_text() == "hello-selfcheck"
        check("dispatch write_file 真实落盘", wrote, wres[:80] if not wrote else "")
        rres = ToolRegistry.dispatch({"action": "read_file", "path": "probe.txt"})
        check("dispatch read_file 回读一致",
              "hello-selfcheck" in rres, rres[:80] if "hello-selfcheck" not in rres else "")
    finally:
        T.WORKSPACE = orig_ws

    # ---------- 2) 装饰器隔离（Bug A）----------
    # 白名单：显式别名（内部函数名以下划线开头，但属故意，如 finish_verify→_finish_verify_tool）
    ALLOWED_INTERNAL_IMPLS = {"_finish_verify_tool", "_finish_analysis_tool"}
    tools_py = os.path.join(ROOT, "swe_agent", "tools.py")
    tree = ast.parse(open(tools_py, encoding="utf-8").read(), tools_py)

    declared = set()  # tools.py 中通过 @tool(name=...) 声明的全部工具名
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name)
                        and dec.func.id == "tool"):
                    name_kw = next((k.value for k in dec.keywords if k.arg == "name"), None)
                    if isinstance(name_kw, ast.Constant):
                        tool_name = name_kw.value
                        declared.add(tool_name)
                        fn_name = node.name
                        # 公开工具不得由 '_' 私有函数实现（除非显式白名单）
                        if not tool_name.startswith("_") and fn_name.startswith("_"):
                            ok = fn_name in ALLOWED_INTERNAL_IMPLS
                            check(f"@tool(name={tool_name!r}) 未错挂到私有函数 {fn_name!r}",
                                  ok, "" if ok else "装饰器被错挂到内部 helper（Bug A 类）")

    for name in CORE_TOOLS:
        check(f"核心工具 [{name}] 有 @tool(name=...) 声明", name in declared,
              "" if name in declared else "装饰器可能被整段删除")

    check("_write_verify 未作为工具注册", "_write_verify" not in ToolRegistry.names(), "")
    wf = ToolRegistry.get("write_file")
    check("write_file 实现未被错挂到内部 helper",
          wf is not None and wf.run.__name__ == "write_file",
          "" if (wf and wf.run.__name__ == "write_file")
          else f"run.__name__={getattr(wf, 'run', None).__name__ if wf else None}")

    # ---------- 3) 循环终止契约（Bug B）----------
    guard = getattr(V, "_tester_on_iter_end", None)
    has = guard is not None
    behaves = False
    if has:
        try:
            behaves = guard(None, "all_done") == "done"
        except Exception:
            behaves = False
    check("tester 终止闸门 _tester_on_iter_end 存在且对 all_done 返回 done", has and behaves,
          "" if (has and behaves) else f"has={has}")

    for role in ("analyzer", "tester", "executor"):
        rc = RC.make_role_config(role)
        check(f"[{role}] RoleConfig 带 stop_actions", bool(rc.stop_actions),
              f"stop_actions={rc.stop_actions}")

    # ---------- 4) 单轮多工具调用契约（2026-09-03）----------
    # 强模型（Ling 等）会一轮下发多个 tool_call；旧 harness 只跑第一个、其余回
    # 「已忽略」假消息，把模型的一轮多步计划砍成一步。这里做源码级防回退闸门：
    # 一旦有人把执行循环改回「只跑第一个」，自检立刻点名，不用等 e2e。
    import swe_agent.config as C
    import inspect
    from swe_agent.agent import Agent
    check("单轮工具调用上限 C.MAX_ACTIONS_PER_RESPONSE ≥ 1",
          int(getattr(C, "MAX_ACTIONS_PER_RESPONSE", 0)) >= 1,
          f"MAX_ACTIONS_PER_RESPONSE={getattr(C, 'MAX_ACTIONS_PER_RESPONSE', None)}")
    check("parallel_tool_calls 已放开（传输层不再强制单工具）",
          bool(getattr(C, "PARALLEL_TOOL_CALLS", False)),
          f"PARALLEL_TOOL_CALLS={getattr(C, 'PARALLEL_TOOL_CALLS', None)}")
    src = inspect.getsource(Agent._apply_toolcall)
    _multi_ok = ("已忽略：每轮只执行第一个工具调用" not in src
                 and "for idx, tc in enumerate(tcs)" in src)
    check("_apply_toolcall 遍历全部 tool_calls（不再只跑第一个）", _multi_ok,
          "" if _multi_ok else "检测到旧的「只执行第一个」实现回退")

    logger.info('')
    if FAILS:
        logger.info('%s', f'❌ 契约自检失败 {len(FAILS)} 项：{FAILS}')
        raise SystemExit(1)
    logger.info('%s', '✅ harness 契约自检全部通过（核心工具注册+行为正常 / 装饰器未错挂 / 循环终止闸门在位 / 单轮多工具调用已接线）')


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(2)
