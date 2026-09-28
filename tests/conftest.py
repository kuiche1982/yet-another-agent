import os
import sys

# Ensure the repo root is importable so that `import demo` and `import swe_agent`
# resolve when pytest collects tests from this directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from swe_agent.state import reset_state
import swe_agent.guards as _guards


@pytest.fixture(autouse=True)
def _reset_global_state():
    """#25 root-cause fix: module-level GLOBAL_STATE leaked across tests.

    reset_state() resets in-place (clear+update, no reassignment) so every
    `from .state import GLOBAL_STATE` reference stays consistent. Running it
    before (and after) each test function kills cross-test state bleed that
    previously let _run_test_bar's GLOBAL_STATE["final_validation"] write and
    test-side GLOBAL_STATE mutations leak into sibling tests.
    """
    reset_state()
    yield
    reset_state()


@pytest.fixture(autouse=True)
def _register_builtin_guards():
    """2026-09-13 重构：内置行为护栏（guards.py）经 HOOK_HUB 自注册。

    test_hooks 会调 HOOK_HUB.clear() 清空全局订阅，若不重注册，后续测试
    （test_harness_contracts 的 unsolvable、test_write_guard 的 write/read 护栏）
    将因 guard 被清而失效。每个测试前定向清空内置 guard 订阅点并重注册，
    既保证内置护栏常驻，又避免误伤 register_global_hook 等其它订阅者。
    """
    _guards.register_all_builtin()
    yield
    _guards.register_all_builtin()
