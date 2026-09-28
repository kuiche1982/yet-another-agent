"""GLOBAL_STATE 默认键契约：模块加载即完整、reset 原地生效、读取点永不 KeyError。

背景：REPL 路径（`interactive_repl`）**从不调用 `reset_state()`** —— 该函数只在
`run_agent`（UNATTEND）与 REPL 的 `/clear` 里调。所以交互式会话全程 GLOBAL_STATE
都停在模块加载时的内容。旧实现 `GLOBAL_STATE = {}`，任何 `GLOBAL_STATE["tasks"]`
直接取键都会 KeyError（实测：REPL 里 `todo_read` 报 `KeyError: 'tasks'`）。

🔴 不变量：
1. 模块加载后 GLOBAL_STATE 已含全部默认键（不是在 reset 之后才有）；
2. `_default_state()` 是唯一事实来源，reset 后的状态与模块加载态逐键一致；
3. `reset_state()` 必须【原地】修改（字典对象 id 不变），否则按引用持有的
   supervisor / llm_glm 会拿到旧字典 → 迭代第 1 轮就「所有任务已完成」；
4. 任务读取辅助函数在默认/空状态下返回 None，不抛 KeyError。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import swe_agent.state as st


# ----------------------------------------------------------------------
# 夹具：本模块会真实改动全局单例，必须逐用例快照 + 原地还原，避免污染其他测试。
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_global_state():
    # 本模块断言「默认态契约」，必须先回到干净的默认态，避免被其他用例（跨模块全局态串扰）
    # 污染。reset_state() 原地重置为 _default_state()，与模块加载态逐键一致。
    st.reset_state()
    yield
    st.reset_state()


def test_module_import_has_all_default_keys():
    """① 模块加载即完整：不调 reset_state() 也能安全取 'tasks'（REPL 的真实路径）。"""
    for key in ("goal", "tasks", "done_list", "current_focus", "planning_done",
                "plan_mode", "worktree", "plan", "generated_files",
                "protected_tests", "takeover_mode", "final_validation"):
        assert key in st.GLOBAL_STATE, f"GLOBAL_STATE 缺少默认键 {key!r}"
    assert st.GLOBAL_STATE["tasks"] == []
    assert st.GLOBAL_STATE["planning_done"] is False
    assert isinstance(st.GLOBAL_STATE["protected_tests"], set)


def test_default_state_is_single_source_of_truth():
    """② _default_state() 与模块级 GLOBAL_STATE 键集合一致，两者永不漂移。"""
    assert set(st._default_state()) == set(st.GLOBAL_STATE)


def test_reset_state_mutates_in_place():
    """③ reset 必须是原地 clear+update：外部 `from .state import GLOBAL_STATE` 引用不失效。"""
    before = st.GLOBAL_STATE
    before_id = id(st.GLOBAL_STATE)
    st.GLOBAL_STATE["tasks"] = [{"desc": "脏数据", "status": "pending"}]

    st.reset_state()

    assert id(st.GLOBAL_STATE) == before_id, "reset_state 重新赋值了字典，引用会全部失效"
    assert st.GLOBAL_STATE is before
    assert st.GLOBAL_STATE["tasks"] == []


def test_reset_state_restores_every_default():
    """④ reset 后逐键回到默认值（含被污染的可变容器与接手模式位）。"""
    st.GLOBAL_STATE["goal"] = "脏目标"
    st.GLOBAL_STATE["done_list"].append("脏完成项")
    st.GLOBAL_STATE["protected_tests"].add("tests/test_dirty.py")
    st.GLOBAL_STATE["takeover_mode"] = True
    st.GLOBAL_STATE["planning_done"] = True

    st.reset_state()

    for key, val in st._default_state().items():
        assert st.GLOBAL_STATE[key] == val, f"reset 后 {key!r} 未回到默认值"
    # 可变容器必须是新对象，避免与 reset 前的引用串改。
    assert st.GLOBAL_STATE["done_list"] is not st.GLOBAL_STATE["tasks"]


def test_task_readers_never_raise_on_empty_state():
    """⑤ 任务读取点在默认态与空态都不 KeyError（钩子/工具应为可拼装、永不阻断）。"""
    assert st.get_current_task() is None
    assert st.mark_current_task_done() is None

    st.GLOBAL_STATE.clear()  # 模拟最坏情况：连默认键都被清掉
    assert st.get_current_task() is None
    assert st.mark_current_task_done() is None
