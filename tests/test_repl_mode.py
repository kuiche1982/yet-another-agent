#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互 REPL 模式单测（model-free / tool-free）。

目标：确认 `swe_agent.supervisor.interactive_repl` 完好可用。
不触 LLM、不触真实工具、不落盘——`input()` 用脚本化输入驱动，
所有外部依赖（run_agent / _sidecar.manual_compact / save_session /
list_sessions / reset_state / stats_summary）一律桩掉。

覆盖：
- 命令分发：plain text、/task、/clear、/compact、/help、/stats、/sessions、/exit
- 未知命令、空 /task 用法提示（两者都不应触发 run_agent）
- 消息线程传递（首轮 messages=None → 后续轮把上一轮 return 当 history 喂回 run_agent）
- EOF（Ctrl-D）干净退出不抛异常
- 退出时把最终 messages 交给 save_session
- resumed 状态并入 GLOBAL_STATE（排除 goal）
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest import mock

from swe_agent import supervisor as SV
import swe_agent.config as C


class _FakeStdin:
    """驱动 interactive_repl 的假 stdin。

    生产代码用 `sys.stdin.readline()` 读交互输入（而非 `input()`），pytest 的假 stdin
    会让 `readline()` 返回 '' 从而陷入无限空轮；这里提供可控的 readline + reconfigure
    （no-op）以脚本化驱动 REPL，且保持 model-free / tool-free（不触 LLM / 真实 TTY）。
    遇到 EOFError 哨兵或输入耗尽时抛 EOFError，等价 Ctrl-D 干净退出。
    """

    def __init__(self, items):
        self._items = list(items)
        self._idx = 0

    def readline(self):
        if self._idx >= len(self._items):
            raise EOFError()
        item = self._items[self._idx]
        self._idx += 1
        if isinstance(item, EOFError):
            raise EOFError()
        return item + "\n"

    def reconfigure(self, **_kw):
        pass

    def isatty(self):
        return False


def _fake_run_agent(task: str, messages=None):
    """记录调用，并把上一轮 history 透传 + 追加一条 assistant 回执（模拟 agent 跑完一轮）。"""
    _fake_run_agent.calls.append((task, len(messages) if messages else 0))
    base = list(messages) if messages else []
    base.append({"role": "assistant", "content": f"done:{task}"})
    return base


def _make_fake_sidecar():
    class _FakeSidecar:
        def manual_compact(self, messages, instructions=""):
            return ([{"role": "assistant", "content": "compacted"}], {"pre": 10, "post": 5})
    return _FakeSidecar()


class _DummyCM:
    """REPL 测试用的假 cm：纯文本 user 行记入 _chat_run_calls（供 ChatAgent 分发断言），
    to_list 返回空（让退出保存回退到 messages）。不触真实模型/工具。

    注意：/exit 保存时会取 cm.truth_list()（真相日志）一并归档，桩必须同签名实现，
    否则生产代码 AttributeError。
    """
    def append(self, role=None, content=None, *a, **k):
        if role == "user":
            _chat_run_calls.append(content)
    def to_list(self):
        return []
    def truth_list(self):
        return []


class _DummyChat:
    """make_chat_agent 的桩：返回带 ctx.cm 的假 Agent，不触真实模型/工具。

    纯文本分支（REPL 当前实现）会先 chat.ctx.cm.append('user', line) 再 chat.run()，
    故这里让 run() 记录调用次数即可验证「纯文本 → ChatAgent 分发路由正确」，
    无需触真实 Agent / LLM / 工具。
    """
    def __init__(self):
        self.ctx = type("X", (), {"cm": _DummyCM()})()

    def run(self):
        # 纯文本分发由 _DummyCM.append('user', line) 记录到 _chat_run_calls，run() 本身空转。
        pass


# 纯文本 → ChatAgent.run() 的分发计数（模块级，便于各用例断言后清理）。
_chat_run_calls: list = []


def _make_dummy_chat(*_a, **_k):
    return _DummyChat()


def _patch_repl_deps(inputs, save_recorder, sessions=(), stats="stats"):
    """组合所有 REPL 外部依赖的 mock patch（均为 `_patch` 管理器，由调用方负责 enter/exit）。

    注意：纯文本分支现已路由到 ChatAgent（普通 Agent 实例）→ 由 chat.run() 驱动，
    不再走 run_agent；/task 仍走 run_agent。故这里把 make_chat_agent 桩成返回 _DummyChat
    （run() 记录调用），仅验证「分发路由正确」。
    """
    sidecar = _make_fake_sidecar()
    return [
        mock.patch.object(sys, "stdin", _FakeStdin(inputs)),
        mock.patch.object(SV, "run_agent", _fake_run_agent),
        mock.patch.object(SV, "make_chat_agent", _make_dummy_chat),
        mock.patch.object(SV, "_sidecar", sidecar),
        mock.patch.object(SV, "reset_state", mock.MagicMock()),
        mock.patch.object(SV, "stats_summary", lambda: stats),
        mock.patch.object(SV, "list_sessions", lambda: list(sessions)),
        mock.patch.object(SV, "save_session",
                          lambda sid, msgs, truth=None: save_recorder.update(
                              session_id=sid, messages=msgs)),
    ]


@contextmanager
def _repl_harness(inputs, session_id="repl-test-1", resumed=None):
    """在全部依赖被桩掉的环境下跑一次 interactive_repl，yield save_recorder 供断言。

    注意：必须用 _patch 管理器本身做 enter/exit（不能用 __enter__ 的返回值，
    因为 patch.object(..., new=函数) 的 __enter__ 返回的就是那个函数，没有 __exit__）。
    """
    _fake_run_agent.calls = []
    _chat_run_calls.clear()
    save_recorder: dict = {}
    patches = _patch_repl_deps(inputs, save_recorder=save_recorder)
    C.SESSION_ID = session_id
    for p in patches:
        p.__enter__()
    try:
        SV.interactive_repl(resumed)
        yield save_recorder
    finally:
        for p in reversed(patches):
            p.__exit__(None, None, None)


# ============================ 主路径：命令分发 + 线程传递 ============================

def test_repl_full_session_dispatch_and_threading():
    # 纯文本 → _drive_chat(chat)；/task → run_agent(messages)。两条 buffer 独立。
    # 本例不插 /clear，以便 /exit 时 to_save 回退到 /task 产出的 messages 验证保存。
    inputs = ["/help", "hello", "/task build fib", "again", "/sessions", "/stats", "/exit"]
    _chat_run_calls.clear()
    with _repl_harness(inputs) as rec:
        # 必须在 patch 仍生效时读取 MagicMock 的 call_count（with 退出后已恢复成真实函数）
        reset_calls = SV.reset_state.call_count
        rt_calls = list(_fake_run_agent.calls)
        run_calls = list(_chat_run_calls)
    # 纯文本走 ChatAgent（chat.ctx.cm.append('user', line) → chat.run()），/task 走 run_agent：
    # hello → ChatAgent；/task build fib → run_agent(messages=None)；again → ChatAgent
    # （/help /sessions /stats /exit 都不触发二者）
    assert run_calls == ["hello", "again"], run_calls
    assert rt_calls == [("build fib", 0)], rt_calls
    # 全程未 /clear → 仅入口 reset_state 一次（/clear 不额外触发）
    assert reset_calls == 1
    # /exit → save_session 被调用，且 to_save 回退到 /task 产出的 messages（[done:build fib]）
    assert rec["session_id"] == "repl-test-1"
    assert rec["messages"] == [{"role": "assistant", "content": "done:build fib"}]


def test_repl_clear_resets_task_messages_and_state():
    """/clear 应清空 /task 产出 messages 并 reset_state；之后纯文本只走 ChatAgent。

    注：interactive_repl 入口本就会 reset_state() 一次（行 1614），/clear 再额外触发一次
    （行 1688），故总计 2 次；无 /clear 的会话仅入口 1 次。这里断言 /clear 确实多触发了一次。
    """
    inputs = ["/task build fib", "/clear", "again", "/exit"]
    _chat_run_calls.clear()
    with _repl_harness(inputs) as rec:
        reset_calls = SV.reset_state.call_count
        rt_calls = list(_fake_run_agent.calls)
        drive_calls = list(_chat_run_calls)
    # /task → run_agent；/clear 后 again → ChatAgent（不看 messages）
    assert rt_calls == [("build fib", 0)], rt_calls
    assert drive_calls == ["again"], drive_calls
    assert reset_calls == 2  # 入口 1 + /clear 1
    # /clear 把 messages 置 None，chat 哑 buffer 恒空 → /exit 不保存
    assert rec == {}


def test_repl_compact_compacts_task_messages():
    """未 /clear 时 /compact 应压缩 /task 产出的 messages（走 _sidecar.manual_compact）。"""
    inputs = ["/task build fib", "/compact", "/exit"]
    with _repl_harness(inputs) as rec:
        pass
    # /compact 后 messages == [compacted]（桩 sidecar 返回），/exit 保存它
    assert rec.get("messages") == [{"role": "assistant", "content": "compacted"}]


# ============================ 退出与保存 ============================

def test_repl_eof_exits_cleanly():
    """Ctrl-D（EOFError）应干净退出、不抛异常。纯文本走 _drive_chat，不触 run_agent。"""
    _chat_run_calls.clear()
    with _repl_harness(["hello", EOFError()]) as rec:
        pass
    # 纯文本 "hello" 被交给 ChatAgent 驱动；无 /task → run_agent 不被调用
    assert _chat_run_calls == ["hello"], _chat_run_calls
    assert _fake_run_agent.calls == []
    # 未跑 /task 且 chat buffer 为空 → 不保存
    assert rec == {}


def test_repl_exit_without_history_skips_save():
    """从未跑过任务（messages 始终 None）时退出不应调用 save_session。"""
    with _repl_harness(["/exit"], session_id="repl-test-2") as rec:
        pass
    assert _fake_run_agent.calls == []
    assert rec == {}  # messages 为 None → 不保存


# ============================ 边界命令 ============================

def test_repl_unknown_command_does_not_call_run_agent():
    """未知斜杠命令应给出提示、不触发 run_agent / 不触发 ChatAgent、且不崩溃。"""
    _chat_run_calls.clear()
    with _repl_harness(["/bogus-cmd", "/exit"]) as rec:
        pass
    assert _fake_run_agent.calls == [], _fake_run_agent.calls
    assert _chat_run_calls == [], _chat_run_calls


def test_repl_empty_task_prints_usage_no_run_agent():
    """"/task" 不带参数 → 用法提示，不触发 run_agent / ChatAgent。"""
    _chat_run_calls.clear()
    with _repl_harness(["/task", "/exit"]) as rec:
        pass
    assert _fake_run_agent.calls == [], _fake_run_agent.calls
    assert _chat_run_calls == [], _chat_run_calls


def test_repl_help_and_stats_do_not_call_run_agent():
    """/help、/stats 只打印、不触 run_agent / ChatAgent。"""
    _chat_run_calls.clear()
    with _repl_harness(["/help", "/stats", "/exit"]) as rec:
        pass
    assert _fake_run_agent.calls == [], _fake_run_agent.calls
    assert _chat_run_calls == [], _chat_run_calls


def test_repl_resumed_state_merges_global_state():
    """带 resumed 状态时，其 state 应并入 GLOBAL_STATE（且排除 goal）。"""
    from swe_agent import state as ST
    resumed = {"messages": None, "state": {"lang": "rust", "goal": "should-be-dropped"}}
    patches = _patch_repl_deps(["/exit"], save_recorder={})
    C.SESSION_ID = "repl-test-3"
    for p in patches:
        p.__enter__()
    try:
        SV.interactive_repl(resumed)
    finally:
        for p in reversed(patches):
            p.__exit__(None, None, None)
    # goal 被显式排除；lang 应并入
    assert ST.GLOBAL_STATE.get("lang") == "rust"
    assert "goal" not in ST.GLOBAL_STATE or ST.GLOBAL_STATE.get("goal") != "should-be-dropped"
    # 清理：避免污染进程内全局状态（reset_state 在测试里被桩成 MagicMock，不会自动清）
    ST.GLOBAL_STATE.pop("lang", None)
