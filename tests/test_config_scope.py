"""分层 KB 作用域与磁盘布局的守卫测试（2026-09-12 定稿）。

四条不变量：
  1. 项目作用域键必须按 git 仓库根归一（在仓库子目录启动不能分裂成两份缓存），
     且任何 git 失败都必须回落而不是抛异常。
  2. 目录常量必须派生自 SWE_AGENT_HOME，而不是仓库 checkout。
  3. KB 的源必须跟着作用域走：proj ← WORKSPACE，global ← global_kb_raw。
  4. 作用域共享粒度是「按 project key 键控」，不是进程内单例；
     日志目录必须惰性创建（import 不落盘）。
"""
import os
import subprocess
import sys

import pytest

from swe_agent import config as C


# ---------------------------------------------------------------- 项目作用域键

def test_project_name_uses_git_toplevel(tmp_path, monkeypatch):
    """A subdirectory of a git repo must resolve to the repo root name."""
    repo = tmp_path / "myrepo"
    sub = repo / "pkg"
    sub.mkdir(parents=True)
    try:
        subprocess.run(["git", "init", "-q", str(repo)], check=True,
                       capture_output=True, text=True)
    except Exception as e:                       # git 不可用则跳过，不算失败
        pytest.skip(f"git unavailable: {e}")
    monkeypatch.setenv("SWE_WORKSPACE", str(sub))
    assert C._project_name() == "myrepo"


def test_project_name_falls_back_when_git_reports_no_repo(tmp_path, monkeypatch):
    """When git answers 'not a repository', the basename is used.

    Note: a directory that merely *sits inside* a repo is not this case -- git walks up
    and returns the enclosing root, which is the desired normalisation (see the
    subdirectory test above). Here we exercise the genuine non-repo branch.
    """
    d = tmp_path / "plain-dir"
    d.mkdir()
    monkeypatch.setenv("SWE_WORKSPACE", str(d))

    class _NotARepo:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: _NotARepo())
    assert C._project_name() == "plain-dir"


def test_project_name_survives_git_failure(tmp_path, monkeypatch):
    """Any git failure (missing binary / timeout / error) must fall back, never raise."""
    d = tmp_path / "safe-dir"
    d.mkdir()
    monkeypatch.setenv("SWE_WORKSPACE", str(d))

    def _boom(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(C.subprocess, "run", _boom)
    assert C._project_name() == "safe-dir"


# ---------------------------------------------------------------- 目录派生

def test_project_dirs_are_nested_under_home():
    """Every scope directory must be derived from SWE_AGENT_HOME."""
    assert C.PROJECT_DIR == C.SWE_AGENT_HOME / "projects" / C.PROJECT_NAME
    assert C.CODE_KB_DIR == C.PROJECT_DIR / "code_kb"
    assert C.PROJ_KB_DIR == C.PROJECT_DIR / "proj_kb"
    assert C.LOGS_DIR == C.PROJECT_DIR / "logs"
    assert C.SESSIONS_DIR == C.PROJECT_DIR / "sessions"
    assert C.GLOBAL_KB_DIR == C.SWE_AGENT_HOME / "global_kb"
    assert C.GLOBAL_KB_RAW_DIR == C.SWE_AGENT_HOME / "global_kb_raw"
    assert C.BIN_DIR == C.SWE_AGENT_HOME / "bin"


def test_legacy_cache_dir_is_retired():
    """RAG_CACHE_DIR must be gone; CODE_KB_DIR / PROJ_KB_DIR now play its role."""
    assert not hasattr(C, "RAG_CACHE_DIR")


@pytest.mark.skipif(bool(os.environ.get("KB_ROOTS")), reason="KB_ROOTS overridden by env")
def test_default_proj_roots_follow_workspace():
    """Project KB roots must come from the workspace, not the harness checkout."""
    assert C.KB_ROOTS == [C.WORKSPACE / "KnowledgeBase", C.WORKSPACE / "docs"]


@pytest.mark.skipif(bool(os.environ.get("KB_GLOBAL_ROOTS")),
                    reason="KB_GLOBAL_ROOTS overridden by env")
def test_default_global_roots_is_raw_dropbox():
    """Global KB must read the hand-curated dropbox under SWE_AGENT_HOME."""
    assert C.KB_GLOBAL_ROOTS == [C.GLOBAL_KB_RAW_DIR]


# ---------------------------------------------------------------- 共享粒度

def test_layered_kb_registry_is_keyed_by_project(monkeypatch):
    """LayeredKB must be cached per project key, not as a process-wide singleton."""
    from swe_agent import management as MG

    class _StubKB:                     # 不触发真实磁盘构建
        def __init__(self):
            self.marker = object()

    monkeypatch.setattr(MG, "_LAYERED_KB_BY_PROJECT", {})
    monkeypatch.setattr(MG, "LayeredKB", _StubKB)

    monkeypatch.setattr(MG.C, "PROJECT_NAME", "proj-a")
    a1 = MG._get_layered_kb()
    a2 = MG._get_layered_kb()
    monkeypatch.setattr(MG.C, "PROJECT_NAME", "proj-b")
    b = MG._get_layered_kb()

    assert a1 is a2, "same project must reuse the same instance"
    assert b is not a1, "different projects must never share an instance"


def test_no_module_level_singleton_remains():
    """The old process-wide _LAYERED_KB global must be gone (it leaked across projects)."""
    from swe_agent import management as MG
    assert not hasattr(MG, "_LAYERED_KB")


# ---------------------------------------------------------------- 日志惰性落盘

def test_importing_log_module_creates_no_directories(tmp_path):
    """Importing swe_agent.log must not materialise the logs directory tree."""
    target = tmp_path / "scope-home"
    env = dict(os.environ)
    env["SWE_AGENT_HOME"] = str(target)
    env["PYTHONPATH"] = str(C.REPO_ROOT)
    r = subprocess.run([sys.executable, "-c", "import swe_agent.log"],
                       env=env, cwd=str(C.REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not target.exists(), \
        "importing swe_agent.log must not create any directory under SWE_AGENT_HOME"
