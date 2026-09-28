"""write_file 单次写入行数护栏（BEFORE_TOOL_CALL gate）+ Y-a loop 支持契约测试。

设计（2026-09-13 重构）：工具内不再硬拒绝「单次写入行数超限」，改由 hook 总线上的
BEFORE_TOOL_CALL gate（WriteSizeGuard，注册于 swe_agent/guards.py）统一裁决——
这正是「tools 的硬限制迁到 guard」的可扩展范式：未来任意工具的前置护栏都可照此
挂一个 gate，loop 层无需改动即获得 REJECT（回灌+继续）/ BREAK_LOOP（回灌+终止）两种语义。

测试分三层：
  1) gate 契约：仅作用于 write_file、超限 REJECT、正常放行；
  2) Y-a 信号：BREAK_LOOP gate 经 dispatch 置 ctx.loop_break；
  3) loop 支持：_run_loop 把 "break_loop" 当权威终止（不经 on_iter_end gate 覆盖）。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import swe_agent.config as C  # noqa: E402
import swe_agent.tools as T  # noqa: E402
import swe_agent.models as M  # noqa: E402
import swe_agent.registry as R  # noqa: E402
from swe_agent.hooks import HookPoint, HOOK_HUB, GateDecision  # noqa: E402
from swe_agent.guards import register_all_builtin  # noqa: E402
from swe_agent.agent import Agent, RunState, LoopConfig  # noqa: E402


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    monkeypatch.setattr(C, "WORKSPACE", tmp_path)
    monkeypatch.setattr(M, "is_weak_executor", lambda: True)  # 弱模型源码上限 500 行
    yield tmp_path


def test_guard_ignores_non_write_file(ws):
    # 行数护栏仅作用于 write_file；其它工具即便带大 content 也不被拦截
    ctx = R.ActionContext()
    big = "\n".join(f"x{i}={i}" for i in range(600))
    out = R.ToolRegistry.dispatch({"action": "read_file", "path": "a.py", "content": big}, ctx)
    assert not ctx.loop_break
    assert "write_rejected" not in out  # 走 read_file 正常分支，而非被护栏拦截


def test_guard_rejects_oversized_write_file(ws):
    # 弱模型源码上限 500 行；600 行超限 → REJECT（回灌 reason、loop 继续、工具不执行）
    ctx = R.ActionContext()
    over = "\n".join(f"x{i}={i}" for i in range(600))
    out = R.ToolRegistry.dispatch({"action": "write_file", "path": "big.py", "content": over}, ctx)
    assert "write_rejected" in out
    assert "超过单次写入上限" in out
    assert not ctx.loop_break  # REJECT：回灌继续，不终止 loop
    assert not (ws / "big.py").exists()  # 工具未执行


def test_guard_allows_normal_write_file(ws):
    # 正常体量放行，文件真实落盘
    ctx = R.ActionContext()
    ok = "\n".join(f"x{i}={i}" for i in range(50))
    out = R.ToolRegistry.dispatch({"action": "write_file", "path": "ok.py", "content": ok}, ctx)
    assert "write_success" in out
    assert (ws / "ok.py").exists()


def test_break_loop_gate_sets_flag_and_reason(ws):
    # Y-a 的「REJECT+stop loop」：BREAK_LOOP gate 经 dispatch 置 ctx.loop_break
    HOOK_HUB.on(HookPoint.BEFORE_TOOL_CALL, _break_gate)
    try:
        ctx = R.ActionContext()
        out = R.ToolRegistry.dispatch(
            {"action": "write_file", "path": "x.py", "content": "a\nb"}, ctx)
        assert ctx.loop_break is True
        assert "stop" in out
    finally:
        # 还原内置护栏（tools 模块 import 时经 swe_agent.guards 注册了 WriteSizeGuard 等；
        # clear 会一并清掉，需补回，避免污染同进程其它测试）。register_all_builtin 定向清空
        # 内置订阅点并重注册全部内置 guard（含 WriteSizeGuard）。
        register_all_builtin()


def _break_gate(payload):
    if payload.fields.get("tool") == "write_file":
        return GateDecision.break_loop(reason="stop: guard says halt")
    return None


def test_run_loop_honors_break_loop(monkeypatch):
    # _run_loop 把 "break_loop" 当权威终止信号：即便 on_iter_end 返回 "continue"（不终止），
    # 也直接返回 "break_loop" 向上传播（Y-a 的「REJECT+stop loop」不经 gate 覆盖）。
    ctx = RunState(role="executor")
    role = SimpleNamespace(name="executor")
    loop = LoopConfig(
        max_iter=3,
        hook_point=HookPoint.L3_LOOP_START,
        on_iter_end=lambda c, r: "continue",  # 若被错误调用，会忽略 break_loop——验证它根本不被调用
    )
    agent = Agent(role, loop, ctx)
    monkeypatch.setattr(agent, "_step", lambda: "break_loop")
    assert agent._run_loop(loop) == "break_loop"


def test_first_block_decision_preserves_action():
    # first_block_decision 保留 GateAction 语义（REJECT vs BREAK_LOOP），
    # 区别于 first_block（只返回 reason 字符串）
    from swe_agent.hooks import HookHub, GateAction
    d_reject = HookHub.first_block_decision([None, GateDecision.reject("no"), None])
    assert d_reject is not None and d_reject.action == GateAction.REJECT
    d_break = HookHub.first_block_decision([GateDecision.break_loop("halt")])
    assert d_break is not None and d_break.action == GateAction.BREAK_LOOP
    assert HookHub.first_block_decision([None, GateDecision.allow()]) is None
