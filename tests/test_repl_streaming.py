"""REPL 流式输出契约：ContextManager.on_message 增量渲染钩子。

背景：REPL 原先在 chat.run() 跑完整轮（可能几十次工具调用）后才批量打印，
表现为「思考和工具消息到最后才一起输出一大块」。修法是加增量钩子：
每条消息一入 buffer 就立即上屏。

🔴 不变量：on_message 恒为 None（UNATTEND 零影响），只有 REPL 的
_attach_repl_streaming 会挂；钩子是旁支，不得影响 buffer，且必须 fail-open。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

from swe_agent.agent import Agent, LoopConfig, RunState
from swe_agent.management import ContextManager
from swe_agent.roles_config import RoleConfig, AgentMode


def _bare_cm():
    """绕过 __init__（避免 RagEngine 扫工作区拖慢/挂死），只测 buffer 读写与钩子。

    注意：必须补齐被调用方法所依赖的**全部**实例字段 —— append() 除写 _msgs 外还写
    _truth（真相日志），漏了会直接 AttributeError。
    """
    cm = object.__new__(ContextManager)
    cm._msgs = []
    cm._truth = []
    cm.on_message = None
    return cm


def test_emit_fires_on_append_and_add_in_order():
    """每条入 buffer 的消息都立即触发回调，且严格按发生顺序。"""
    cm = _bare_cm()
    seen = []
    cm.on_message = lambda m: seen.append(m)

    cm.append("user", "帮我看下项目")
    cm.add({"role": "assistant", "content": "我先看目录",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "glob", "arguments": "{}"}}]})
    cm.append("tool", "test.py", tool_call_id="c1")

    assert [m["role"] for m in seen] == ["user", "assistant", "tool"]
    assert seen[2]["content"] == "test.py"
    # 钩子是旁支，不得影响 buffer 本身
    assert len(cm.to_list()) == 3
    assert cm.to_list()[1]["content"] == "我先看目录"


def test_hook_default_none_unattend_safe():
    """默认无钩子：UNATTEND（run_agent）路径零回调、零开销。"""
    cm = _bare_cm()
    cm.append("user", "hi")
    cm.add({"role": "assistant", "content": "ok"})
    assert len(cm.to_list()) == 2


def test_emit_fail_open():
    """显示回调炸了也不能带走 agent 主链路。"""
    cm = _bare_cm()

    def boom(_m):
        raise RuntimeError("display boom")

    cm.on_message = boom
    cm.append("user", "hi")
    assert len(cm.to_list()) == 1


def test_agent_construction_never_attaches_renderer():
    """UNATTEND 建 Agent 时不得挂任何渲染钩子（REPL 独有）。"""
    ctx = RunState(role="executor")
    ctx.cm = _bare_cm()
    agent = Agent(
        RoleConfig(name="executor", mode=AgentMode.TOOLCALL),
        LoopConfig(max_iter=1),
        ctx=ctx,
    )
    assert agent.ctx.cm.on_message is None


def test_reset_preserves_hook_for_cross_turn_streaming():
    """跨轮上下文保留：reset(messages) 换内容但不该冲掉 REPL 钩子。"""
    cm = _bare_cm()
    cm.on_message = lambda m: None
    cm.reset([{"role": "user", "content": "新的一轮"}])
    assert cm.on_message is not None
    assert cm.to_list()[0]["content"] == "新的一轮"


def test_attach_streams_incrementally():
    """_attach_repl_streaming 真正按发生顺序实时渲染（非 run() 后成块）。"""
    from swe_agent.supervisor import AgentDisplay, _attach_repl_streaming

    lines = []
    display = AgentDisplay(show_thinking=True, show_tool_args=True,
                           sink=lines.append)
    chat = SimpleNamespace(ctx=SimpleNamespace(cm=_bare_cm()))
    _attach_repl_streaming(chat, display)

    chat.ctx.cm.append("user", "帮我看下项目")
    chat.ctx.cm.add({
        "role": "assistant", "content": "我先看目录",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "glob",
                                     "arguments": '{"pattern":"**/*.py"}'}}],
    })
    chat.ctx.cm.append("tool", "test.py", tool_call_id="c1")

    def idx(sub):
        for i, s in enumerate(lines):
            if sub in s:
                return i
        return -1

    i_assistant, i_call, i_tool = idx("[assistant]"), idx("调用工具：glob"), idx("[tool]")
    assert i_assistant >= 0, lines
    assert i_call >= 0, lines
    assert i_tool >= 0, lines
    assert i_assistant < i_call < i_tool, lines  # 实时顺序，非攒到最后
    assert "**/*.py" in lines[i_call]
