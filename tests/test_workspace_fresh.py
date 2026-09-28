#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工作区清空策略（SWE_FRESH_WORKSPACE / --fresh）model-free / tool-free 单测。

核心不变量：默认【保留】工作区（非破坏性）；清空必须显式开启（--fresh 或
SWE_FRESH_WORKSPACE=1）；--session 恢复时一律不清空。全程不触 LLM、不落盘。
"""
import os
import sys
import shutil
from unittest import mock

import swe_agent.config as config
import swe_agent.supervisor as SV
import swe_agent.config as C


def _run_main(argv, *, env_fresh=None, resumed=None):
    """在依赖全桩环境下跑一次 main()，返回 shutil.rmtree 是否被调用。"""
    saved = os.environ.get("SWE_FRESH_WORKSPACE")
    if env_fresh is None:
        os.environ.pop("SWE_FRESH_WORKSPACE", None)
    else:
        os.environ["SWE_FRESH_WORKSPACE"] = env_fresh
    try:
        rmtree = mock.MagicMock()
        run_agent = mock.MagicMock()
        repl = mock.MagicMock()
        start_mon = mock.MagicMock()
        fake_ws = mock.MagicMock()
        fake_ws.exists.return_value = True  # 让 exists() 为真，触发「若 _fresh 则 rmtree」
        patches = [
            mock.patch.object(sys, "argv", ["prog"] + argv),
            mock.patch.object(config, "UNATTENDED_MODE", True),  # 确定性走 run_agent
            mock.patch.object(SV, "run_agent", run_agent),
            mock.patch.object(SV, "interactive_repl", repl),
            mock.patch.object(SV, "start_stdin_monitor", start_mon),
            mock.patch.object(SV, "new_session_id", lambda: "sess-fresh-test"),
            mock.patch.object(SV._plugins, "load_plugins", lambda *a, **k: None),
            mock.patch.object(SV, "load_session", lambda sid: resumed),
            mock.patch("shutil.rmtree", rmtree),
            mock.patch.object(C, "WORKSPACE", fake_ws),
        ]
        # 注意：mock.patch.object(..., new=函数) 的 __enter__() 返回的是被注入的函数本身
        # （无 __exit__），必须以 patch 对象 p 本体做 enter/exit，不能存 __enter__ 返回值。
        entered = []
        for p in patches:
            p.__enter__()
            entered.append(p)
        try:
            SV.main()
        finally:
            for p in reversed(entered):
                p.__exit__(None, None, None)
        return rmtree.called
    finally:
        if saved is None:
            os.environ.pop("SWE_FRESH_WORKSPACE", None)
        else:
            os.environ["SWE_FRESH_WORKSPACE"] = saved


def test_default_keeps_workspace():
    """未设 --fresh、未设 SWE_FRESH_WORKSPACE → 不清空（安全默认，修复破坏性默认值）。"""
    assert _run_main([]) is False


def test_fresh_flag_clears():
    """--fresh → 清空工作区。"""
    assert _run_main(["--fresh"]) is True


def test_env_fresh_equal_1_clears():
    """SWE_FRESH_WORKSPACE=1 → 清空（电池显式开启，不受影响）。"""
    assert _run_main([], env_fresh="1") is True


def test_env_fresh_equal_0_keeps():
    """SWE_FRESH_WORKSPACE=0 → 不清空（显式关闭）。"""
    assert _run_main([], env_fresh="0") is False


def test_resume_never_clears_even_with_fresh_flag():
    """--session 恢复（resumed 非空）时即便带 --fresh 也不清空。"""
    assert _run_main(["--session", "abc"], resumed={"messages": [], "state": {}}) is False
