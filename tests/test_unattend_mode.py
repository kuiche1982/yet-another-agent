#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UNATTEND（无人值守 / 批处理，非 REPL）模式 model-free/tool-free 单测。

覆盖 UNATTEND_MODE 下的关键不变量：后台 / 管道 / CI 下绝不调用 input() 阻塞 stdin，
危险操作被拒绝、ask 工具被 BUILD 层摘除、stdin 监听线程不启动、main() 派发走 agent 而非 REPL。

全程不触 LLM、不触真实工具、不落盘、不启线程——纯逻辑契约验证。
"""

from contextlib import contextmanager
from unittest import mock

import sys

import swe_agent.config as C
import swe_agent.tools as T
import swe_agent.registry as R
import swe_agent.compact as compact
import swe_agent.supervisor as SV


# —— 同时切换两个绑定（tools 模块级 UNATTENDED_MODE 与 config 模块级 UNATTENDED_MODE）—— #
@contextmanager
def _unattend(on: bool):
    with mock.patch.object(C, "UNATTENDED_MODE", on), \
         mock.patch.object(T, "UNATTENDED_MODE", on):
        yield


# ======================================================================
# 1) _gate_confirm：危险操作闸门
# ======================================================================
def test_gate_confirm_denies_in_unattended_without_input():
    """无人值守：危险操作一律拒绝，且绝不调用 input()（避免后台 stdin 卡死）。"""
    inp = mock.MagicMock()
    with _unattend(True), \
         mock.patch.object(T, "YOLO_MODE", False), \
         mock.patch("builtins.input", inp):
        res = T._gate_confirm("rm -rf /", "删除根目录")
    assert res is not None and res.startswith("denied:"), res
    assert not inp.called, "UNATTEND 模式下 _gate_confirm 不应调用 input()"


def test_gate_confirm_yolo_bypasses_gate():
    """--yolo 优先于无人值守：直接放行，不拒绝、不询问。"""
    inp = mock.MagicMock()
    with _unattend(True), \
         mock.patch.object(T, "YOLO_MODE", True), \
         mock.patch("builtins.input", inp):
        res = T._gate_confirm("rm -rf /", "删除根目录")
    assert res is None, "YOLO 模式应直接放行"
    assert not inp.called


def test_gate_confirm_prompts_in_interactive_tty():
    """交互 TTY（非无人值守）：询问，回答 y → 放行。"""
    inp = mock.MagicMock(return_value="y")
    with _unattend(False), \
         mock.patch.object(T, "YOLO_MODE", False), \
         mock.patch.object(sys.stdin, "isatty", return_value=True), \
         mock.patch("builtins.input", inp):
        res = T._gate_confirm("restart service", "重启服务")
    assert res is None, "TTY 下回答 y 应放行"
    assert inp.called


def test_gate_confirm_denies_in_non_tty_pipeline():
    """非 TTY 管道（非无人值守）：直接拒绝，不阻塞。"""
    inp = mock.MagicMock()
    with _unattend(False), \
         mock.patch.object(T, "YOLO_MODE", False), \
         mock.patch.object(sys.stdin, "isatty", return_value=False), \
         mock.patch("builtins.input", inp):
        res = T._gate_confirm("restart service", "重启服务")
    assert res is not None and res.startswith("denied:"), res
    assert not inp.called, "非 TTY 下不应调用 input()"


# ======================================================================
# 2) ask_user：人类提问工具
# ======================================================================
def test_ask_user_returns_unattended_message_without_input():
    """无人值守：ask 被兜底短路，返回占位提示，绝不调用 input()。"""
    inp = mock.MagicMock()
    with _unattend(True), mock.patch("builtins.input", inp):
        res = T.ask_user("用哪个数据库？", ["sqlite", "postgres"])
    assert "无人值守模式" in res, res
    assert not inp.called, "UNATTEND 模式下 ask_user 不应调用 input()"


def test_ask_user_non_tty_returns_placeholder():
    """非 TTY 管道：返回非交互占位提示，不阻塞。"""
    inp = mock.MagicMock()
    with _unattend(False), \
         mock.patch.object(sys.stdin, "isatty", return_value=False), \
         mock.patch("builtins.input", inp):
        res = T.ask_user("用哪个数据库？", ["sqlite", "postgres"])
    assert "非交互模式" in res, res
    assert not inp.called


def test_ask_user_interactive_tty_reads_answer():
    """交互 TTY：正常读取回答。"""
    inp = mock.MagicMock(return_value="sqlite")
    with _unattend(False), \
         mock.patch.object(sys.stdin, "isatty", return_value=True), \
         mock.patch("builtins.input", inp):
        res = T.ask_user("用哪个数据库？", ["sqlite", "postgres"])
    assert res == "用户回答: sqlite", res
    assert inp.called


# ======================================================================
# 3) glm_tools：BUILD 层工具 schema 组装
# ======================================================================
def test_glm_tools_strips_ask_in_unattended():
    """无人值守：ask 工具被摘除（BUILD 层拦截），模型根本调不到。"""
    with _unattend(True):
        fns = R.ToolRegistry.glm_tools(role=None)
    names = [f["function"]["name"] for f in fns]
    assert "ask" not in names, "UNATTEND 模式 glm_tools 不应下发 ask 工具"
    # 其余工具（如 shell）仍正常下发，证明不是整张表被清空
    assert "shell" in names, "UNATTEND 模式不应误删其它工具"


def test_glm_tools_keeps_ask_in_interactive():
    """交互模式：ask 正常下发（正向断言，防止测试假绿）。"""
    with _unattend(False):
        fns = R.ToolRegistry.glm_tools(role=None)
    names = [f["function"]["name"] for f in fns]
    assert "ask" in names, "交互模式 glm_tools 应下发 ask 工具"
    # 注册表里确有 ask（装饰器 name='ask'）
    assert "ask" in R.ToolRegistry.names(), "registry 未注册 ask 工具（与装饰器 name='ask' 不符）"


def test_glm_tools_role_filter_still_applies_with_gate():
    """无人值守 + per-role 过滤：ask 摘除与角色裁剪同时生效。"""
    with _unattend(True):
        # 任意角色（executor/planner/analyzer）都不应出现 ask
        for role in ("executor", "planner", "analyzer", "tester"):
            names = [f["function"]["name"] for f in R.ToolRegistry.glm_tools(role=role)]
            assert "ask" not in names, f"角色 {role} 不应下发 ask"


# ======================================================================
# 4) start_stdin_monitor：stdin 监听线程
# ======================================================================
def test_start_stdin_monitor_no_thread_in_unattended():
    """无人值守：不启动 stdin 监听线程（避免后台 readline 阻塞）。"""
    with _unattend(True), mock.patch("threading.Thread") as thr:
        compact.start_stdin_monitor()
    assert not thr.called, "UNATTEND 模式不应启动 stdin 监听线程"


def test_start_stdin_monitor_starts_thread_in_interactive():
    """交互模式：正常启动守护线程监听 /compact。"""
    fake_thread = mock.MagicMock()
    with _unattend(False), \
         mock.patch("threading.Thread", return_value=fake_thread) as thr:
        compact.start_stdin_monitor()
    assert thr.called, "交互模式应启动 stdin 监听线程"
    assert fake_thread.start.called, "应调用 Thread.start()"


# ======================================================================
# 5) supervisor.main()：派发决策（unattended → agent，非 REPL）
# ======================================================================
def test_main_unattended_no_task_runs_agent_not_repl():
    """无人值守 + 无 task 参数：走默认任务经 run_agent，绝不进 REPL。"""
    run_agent = mock.MagicMock()
    repl = mock.MagicMock()
    start_mon = mock.MagicMock()
    fake_ws = mock.MagicMock()
    with mock.patch.object(sys, "argv", ["prog"]), \
         mock.patch.object(C, "UNATTENDED_MODE", True), \
         mock.patch.object(SV, "run_agent", run_agent), \
         mock.patch.object(SV, "interactive_repl", repl), \
         mock.patch.object(SV, "start_stdin_monitor", start_mon), \
         mock.patch.object(SV, "new_session_id", lambda: "sess-unattend-x"), \
         mock.patch.object(SV._plugins, "load_plugins", lambda *a, **k: None), \
         mock.patch("shutil.rmtree"), \
         mock.patch.object(C, "WORKSPACE", fake_ws):
        SV.main()
    run_agent.assert_called_once()
    called_task = run_agent.call_args.args[0]
    assert called_task == SV.DEFAULT_TASK, "无人值守无 task 应使用 DEFAULT_TASK"
    repl.assert_not_called(), "无人值守不应进入 REPL"
    start_mon.assert_not_called(), "无人值守不应启动 stdin 监听"


def test_main_interactive_no_task_enters_repl():
    """对照：交互 TTY + 无 task → 走 REPL（证明派发分支正确，非 UNATTEND 路径）。

    注：main() 中 start_stdin_monitor() 已被注释掉（避免后台进程下 readline 阻塞），
    交互与无人值守模式均不再启动 stdin 监听线程，故此处断言不调用。
    """
    run_agent = mock.MagicMock()
    repl = mock.MagicMock()
    start_mon = mock.MagicMock()
    fake_ws = mock.MagicMock()
    with mock.patch.object(sys, "argv", ["prog"]), \
         mock.patch.object(C, "UNATTENDED_MODE", False), \
         mock.patch.object(sys.stdin, "isatty", return_value=True), \
         mock.patch.object(SV, "run_agent", run_agent), \
         mock.patch.object(SV, "interactive_repl", repl), \
         mock.patch.object(SV, "start_stdin_monitor", start_mon), \
         mock.patch.object(SV, "new_session_id", lambda: "sess-repl-x"), \
         mock.patch.object(SV._plugins, "load_plugins", lambda *a, **k: None), \
         mock.patch("shutil.rmtree"), \
         mock.patch.object(C, "WORKSPACE", fake_ws):
        SV.main()
    repl.assert_called_once(), "交互无 task 应进入 REPL"
    run_agent.assert_not_called(), "交互无 task 不应直接跑 run_agent"
    start_mon.assert_not_called(), "main() 已注释掉 start_stdin_monitor，交互模式也不启动 stdin 监听"
