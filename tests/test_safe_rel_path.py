#!/usr/bin/env python3
"""路径归一化单测 —— 锁定双层路径 bug（e2e run 9Kon2b 复现）。

根因：supervisor 把 {WORKSPACE} 字面（如 agent_sandbox/<run_id>/fib）注入系统提示，
弱模型把该完整路径当相对基回写；旧 _safe_rel 只剥首层字面 "agent_sandbox/" 再用
WORKSPACE 重拼 → 出现 agent_sandbox/<run_id>/fib/<run_id>/fib/... 双层路径。

本测试用 monkeypatch 把 state.WORKSPACE / state.REPO_ROOT 固定成电池每任务隔离目录
（绝对形态，与生产一致），断言 _safe_rel 把三种模型回写形态都规整回 WORKSPACE 根下：

  1. 模型写相对路径 src/fib.py                      → src/fib.py
  2. 模型写完整 WORKSPACE 前缀（相对仓库根形态）      → src/fib.py
  3. 模型写绝对 WORKSPACE 前缀                        → src/fib.py
  4. 模型已写出双层（再次回写完整前缀）               → src/fib.py
  5. 模型试图用 ../ 逃逸                              → 被 ".." 过滤，落 WORKSPACE 内

运行：uv run pytest tests/test_safe_rel_path.py -q
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pytest

from swe_agent import state

WS_REL = "agent_sandbox/20260902_110625_50c026/fib"


@pytest.fixture
def patch_workspace(monkeypatch):
    """固定 WORKSPACE 为绝对形态（与生产一致：config.WORKSPACE = SWE_WORKSPACE.resolve()）。

    用 tempfile.mkdtemp(dir="/tmp") 自建临时目录，绕开 brokered sandbox 对 pytest 默认
    tmp_path 基目录（/private/var/folders/.../pytest-of-unknown）的 mkdir 拦截。
    """
    base = tempfile.mkdtemp(dir="/tmp")
    repo = Path(base) / "repo"
    repo.mkdir()
    ws = repo / "agent_sandbox" / "20260902_110625_50c026" / "fib"
    ws.mkdir(parents=True)
    monkeypatch.setattr(state, "WORKSPACE", ws)
    monkeypatch.setattr(state, "REPO_ROOT", repo)
    return ws


def test_plain_relative_stays(patch_workspace):
    # 模型遵守约定，直接写相对 WORKSPACE 根的路径
    assert state._safe_rel("src/fib.py") == "src/fib.py"
    assert state._safe_rel("test_fib.py") == "test_fib.py"


def test_model_writes_full_workspace_prefix_relative(patch_workspace):
    # 弱模型把 prompt 注入的 {WORKSPACE} 字面当相对基回写（相对仓库根形态）
    got = state._safe_rel(f"{WS_REL}/src/fib.py")
    assert got == "src/fib.py", got


def test_model_writes_absolute_workspace_prefix(patch_workspace):
    # 模型写出绝对 WORKSPACE 前缀
    ws = patch_workspace
    got = state._safe_rel(str(ws / "src" / "fib.py"))
    assert got == "src/fib.py", got


def test_model_writes_already_double_nested(patch_workspace):
    # 已经双层（再次回写完整前缀）→ 仍规整回单层
    got = state._safe_rel(f"{WS_REL}/{WS_REL}/src/fib.py")
    assert got == "src/fib.py", got


def test_model_writes_double_nested_absolute(patch_workspace):
    ws = patch_workspace
    doubled = ws / WS_REL / "src" / "fib.py"
    got = state._safe_rel(str(doubled))
    assert got == "src/fib.py", got


def test_path_escape_filtered(patch_workspace):
    # 防逃逸：../ 被 parts 过滤，落到 WORKSPACE 内而非逃出沙箱
    got = state._safe_rel("../secrets/token.txt")
    assert ".." not in got.split("/")
    assert got == "secrets/token.txt", got


def test_backslash_normalized(patch_workspace):
    # Windows 反斜杠归一化
    got = state._safe_rel(f"{WS_REL}\\src\\fib.py")
    assert got == "src/fib.py", got
