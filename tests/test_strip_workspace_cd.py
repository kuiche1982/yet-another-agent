#!/usr/bin/env python3
"""cd 冗余剥离 + cwd 提示措辞单测。

根因（e2e run mptb1j 复现）：executor 在 `cd /workspace/agent_sandbox` 失败里反复横跳，
pytest 永远从错目录起 → "no tests collected"。两层病因：

  A) _strip_redundant_workspace_cd 只在 `Path(target).name == WORKSPACE.name` 时剥 cd；
     e2e 下 WORKSPACE = agent_sandbox/<run_id>/<name>，叶子是任务名（如 fizzbuzz），
     模型写出的 `agent_sandbox` 末段匹配不上 → strip 静默失效（tools.py 旧 bug）。
  B) supervisor cwd 提示写「已是绝对路径」又与「严禁绝对路径」自相矛盾，prime 弱模型去
     寻址它根本没拿到的绝对路径（supervisor.py 旧措辞）。

本测试固定 tools.WORKSPACE / config.WORKSPACE 为生产形态（嵌套隔离目录），断言：
  1. `cd /workspace/agent_sandbox && ...` 被剥（核心 bug 回归锁）
  2. `cd agent_sandbox` / `cd fizzbuzz` / `cd <run_id>/fizzbuzz` 被剥
  3. `cd src` / `cd src/foo` 等合法 cd 保留
  4. 纯命令、无前导 cd 不变
  5. cwd 提示不再含「已是绝对路径」矛盾句

运行：uv run pytest tests/test_strip_workspace_cd.py -q
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pytest

import swe_agent.tools as tools
import swe_agent.supervisor as supervisor
from swe_agent import config


@pytest.fixture
def nested_workspace(monkeypatch):
    """固定 WORKSPACE 为 e2e 生产形态：agent_sandbox/<run_id>/<name>（叶子=任务名）。

    mkdtemp(dir="/tmp") 绕开 brokered sandbox 对 pytest 默认 tmp_path 基目录的 mkdir 拦截。
    """
    base = tempfile.mkdtemp(dir="/tmp")
    repo = Path(base) / "repo"
    repo.mkdir()
    # 叶子是 "fizzbuzz"，而非 "agent_sandbox" —— 正是旧 strip 漏匹配的情形
    ws = repo / "agent_sandbox" / "run1" / "fizzbuzz"
    ws.mkdir(parents=True)
    monkeypatch.setattr(tools, "WORKSPACE", ws)          # 函数经 from .config import WORKSPACE 别名读取
    monkeypatch.setattr(config, "WORKSPACE", ws)
    monkeypatch.setattr(supervisor.config, "WORKSPACE", ws)
    return ws


def test_hallucinated_absolute_agent_sandbox_is_stripped(nested_workspace):
    # 核心回归：模型幻觉出的 /workspace/agent_sandbox（末段 agent_sandbox ≠ 叶子 fizzbuzz）
    cmd = "cd /workspace/agent_sandbox && pytest -q"
    assert tools._strip_redundant_workspace_cd(cmd) == "pytest -q"


def test_relative_agent_sandbox_is_stripped(nested_workspace):
    assert tools._strip_redundant_workspace_cd("cd agent_sandbox && pytest -q") == "pytest -q"


def test_workspace_leaf_cd_is_stripped(nested_workspace):
    # 模型用真实叶子名进入项目根
    assert tools._strip_redundant_workspace_cd("cd fizzbuzz && pytest -q") == "pytest -q"
    assert tools._strip_redundant_workspace_cd(
        "cd agent_sandbox/run1/fizzbuzz && pytest -q") == "pytest -q"


def test_exact_resolved_workspace_is_stripped(nested_workspace):
    ws = nested_workspace
    assert tools._strip_redundant_workspace_cd(
        f"cd {ws} && pytest -q") == "pytest -q"


def test_legit_subdir_cd_is_preserved(nested_workspace):
    # 合法 cd 不能误剥
    assert tools._strip_redundant_workspace_cd("cd src && pytest -q") == "cd src && pytest -q"
    assert tools._strip_redundant_workspace_cd("cd src/foo && ls") == "cd src/foo && ls"


def test_plain_command_unchanged(nested_workspace):
    assert tools._strip_redundant_workspace_cd("pytest -q") == "pytest -q"
    assert tools._strip_redundant_workspace_cd("ls -la") == "ls -la"


def test_workspace_context_has_no_absolute_path_claim(nested_workspace):
    # supervisor cwd 提示不得再含「已是绝对路径」自相矛盾句
    body = supervisor._workspace_context_section()
    assert "已是绝对路径" not in body, "cwd 提示仍在暗示模型持有绝对路径，会 prime 其去寻址"
    assert "无需 cd" in body
