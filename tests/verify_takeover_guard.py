import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 接手保护闸（fences）离线验证：护栏现已落到 supervisor.py 的 hook 层实现，
# 本测试直接驱动这些 hook / 辅助函数，证明「预置测试夹具不被覆盖、正常写入不受误伤」。
from swe_agent import config as C
from swe_agent.state import GLOBAL_STATE
from swe_agent.supervisor import (
    _is_test_file,
    _detect_takeover_mode,
    _check_takeover_protection,
    _snapshot_existing_tests,
    _hook_write_protection,
    _hook_plan_mode_guard,
)
from swe_agent.registry import ActionContext


def _reset_state():
    GLOBAL_STATE.clear()
    GLOBAL_STATE.update(
        {
            "takeover_mode": False,
            "protected_tests": set(),
            "protected_impl": set(),
            "plan_mode": False,
        }
    )


def test_is_test_file():
    hit = [
        "tests/unit/test_game_of_life.py",
        "test_app.py",
        "src/tests/test_todo.py",
        "app.test.js",
        "foo.spec.js",
        "tests/unit/calc_test.py",
    ]
    miss = [
        "src/game_of_life.py",
        "index.html",
        "src/app.js",
        "README.md",
        "tests/conftest.py",
        "src/tests/__init__.py",
        "package.json",
    ]
    assert all(_is_test_file(p) for p in hit), [p for p in hit if not _is_test_file(p)]
    assert not any(_is_test_file(p) for p in miss), [p for p in miss if _is_test_file(p)]


def test_detect_takeover_mode():
    PRE = ["tests/unit/test_game_of_life.py"]
    assert _detect_takeover_mode("接手别人的代码，修复 bug 使测试通过", PRE)
    assert _detect_takeover_mode(
        "完成康威生命游戏的开发。注意：这是中途接手别人的代码，"
        "agent_sandbox 里已有测试，不要从头重写整个项目。",
        PRE,
    )
    # 有接手语义但无既有测试 → 不算接手
    assert not _detect_takeover_mode("接手现有代码", [])
    # 有既有测试但非接手语义 → 不算接手
    assert not _detect_takeover_mode("从零开发一个 TODO 页面", PRE)


def test_takeover_block():
    _reset_state()
    GLOBAL_STATE["takeover_mode"] = True
    GLOBAL_STATE["protected_tests"] = {"tests/unit/test_game_of_life.py"}

    blocked_exact = _check_takeover_protection("tests/unit/test_game_of_life.py")
    blocked_pref = _check_takeover_protection("agent_sandbox/tests/unit/test_game_of_life.py")
    blocked_abs = _check_takeover_protection(
        "~/kuiwork/workdir2/litertlm/agent_sandbox/tests/unit/test_game_of_life.py"
    )
    allowed_new = _check_takeover_protection("tests/unit/test_extra.py")
    allowed_impl = _check_takeover_protection("src/game_of_life.py")

    assert blocked_exact and blocked_pref and blocked_abs
    assert not allowed_new and not allowed_impl
    assert "接手保护" in blocked_exact


def test_non_takeover_free():
    _reset_state()
    GLOBAL_STATE["takeover_mode"] = False
    assert not _check_takeover_protection("tests/unit/test_game_of_life.py")


def test_hook_write_protection_blocks():
    _reset_state()
    GLOBAL_STATE["takeover_mode"] = True
    GLOBAL_STATE["protected_tests"] = {"tests/unit/test_game_of_life.py"}

    ctx = ActionContext(action="write_file")
    block = _hook_write_protection(ctx, {"path": "tests/unit/test_game_of_life.py"}, None)
    assert isinstance(block, str) and "接手保护" in block

    # 非受保护文件放行
    assert _hook_write_protection(ctx, {"path": "src/game_of_life.py"}, None) is None


def test_hook_plan_mode_guard():
    _reset_state()
    GLOBAL_STATE["plan_mode"] = True
    ctx = ActionContext(action="write_file")
    block = _hook_plan_mode_guard(ctx, {}, None)
    assert isinstance(block, str) and "plan_mode_blocked" in block

    # 非规划模式放行
    GLOBAL_STATE["plan_mode"] = False
    assert _hook_plan_mode_guard(ctx, {}, None) is None

    # 规划模式仍允许只读动作
    assert _hook_plan_mode_guard(ActionContext(action="read_file"), {}, None) is None


def test_snapshot_existing_tests(tmp_path):
    _reset_state()
    # _safe_rel 会把路径里的 'agent_sandbox/' 前缀剥离并去掉前导斜杠，
    # 因此让临时 WORKSPACE 含 agent_sandbox 段，返回值即为相对路径。
    ws = tmp_path / "agent_sandbox"
    ws.mkdir()
    (ws / "tests" / "unit").mkdir(parents=True)
    (ws / "tests" / "unit" / "test_x.py").write_text("def test_x(): pass\n")
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "impl.py").write_text("x = 1\n")

    saved = C.WORKSPACE
    try:
        C.WORKSPACE = ws
        snap = _snapshot_existing_tests()
        assert "tests/unit/test_x.py" in snap
        assert not any(f == "src/impl.py" for f in snap)
    finally:
        C.WORKSPACE = saved


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
