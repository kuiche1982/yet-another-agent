#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChatAgent（普通 Agent 实例，全量工具）model-free / tool-free 单测。

验证 REPL 交互模式的核心不变量：
- ChatAgent 是 Agent 的实例（非子类），工具调用路径与普通 Agent 完全一致；
- allow_all_tools：已注册但不在任何角色 ROLE_TOOLS 白名单里的工具也能被执行；
- stop_on_no_tool：模型给出纯文本答复即结束本轮（返回 all_done），不再逼它调工具；
- 多步：模型先调工具 → 结果回灌 → 再给文本答复 → 自然终止；
- REPL 自有打印（_print_repl_msg / _drive_chat）把回复渲染到控制台，不依赖 logger。

全程不触 LLM、不触真实工具、不落盘（模型/工具调度全部桩掉，workspace kb 用空实例替代）。
"""

import pytest

import swe_agent.models as M
from contextmgr import ContextManager as RagEngine
from swe_agent import supervisor as SV
from swe_agent.agent import Agent
from swe_agent.registry import ToolRegistry, ToolDef
from swe_agent.roles_config import make_chat_agent, make_agent
from swe_agent.supervisor import _drive_chat, _print_repl_msg, AgentDisplay
from swe_agent.management import ModelManager


_TEST_TOOL = "zzz_chat_fulltool_test"


class _ScriptedModel:
    """脚本化 chat_toolcalls：第 1 次返回工具调用，之后返回纯文本答复。"""
    def __init__(self):
        self.calls = 0
        self.tool_then_text = True

    def __call__(self, role, messages, tools=None, model_override=None):
        self.calls += 1
        if self.tool_then_text and self.calls == 1:
            return {"type": "toolcalls", "tool_calls": [
                {"id": "c1", "name": _TEST_TOOL, "arguments": '{"x": 1}'}]}
        return {"type": "text", "content": "done: 当前目录有 file.py", "tool_calls": []}


@pytest.fixture
def chat_env(monkeypatch):
    # 1) workspace kb 用空实例替代，避免单测扫描真实工作区
    monkeypatch.setattr(
        "swe_agent.management._get_layered_kb",
        lambda: RagEngine(budget_tokens=2048))
    # 1.5) 中性化模型 load/unload hook（chat 默认走本地 lmstudio → 工厂会绑真 load 钩子），
    #      避免单测触发真实 LM Studio HTTP 请求；保留 hook 绑定逻辑本身。
    monkeypatch.setattr(ModelManager, "load", lambda mid: None)
    monkeypatch.setattr(ModelManager, "unload", lambda mid: None)

    # 2) 注册一个「不在任何角色 ROLE_TOOLS 白名单」的工具，验证 allow_all_tools 放行
    def _fake_tool(ctx=None, x=0):
        return f"ran {_TEST_TOOL} with x={x}"
    ToolRegistry.register(ToolDef(
        name=_TEST_TOOL, description="test full-tool", category="meta",
        schema={"type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": []},
        run=_fake_tool))

    # 3) 桩掉模型与工具调度
    scripted = _ScriptedModel()
    dispatch_calls = []
    monkeypatch.setattr(M, "chat_toolcalls", scripted)
    monkeypatch.setattr(
        ToolRegistry, "dispatch",
        staticmethod(lambda action, ctx=None: dispatch_calls.append(action["action"]) or f"result for {action['action']}"))

    yield {"scripted": scripted, "dispatch_calls": dispatch_calls}

    ToolRegistry.unregister(_TEST_TOOL)


def test_chat_agent_is_plain_agent_instance(chat_env):
    """ChatAgent 应是 Agent 的实例（不是子类），由工厂产出。"""
    agent = make_chat_agent()
    assert isinstance(agent, Agent)
    assert agent.role.name == "chat"
    assert agent.role.allow_all_tools is True
    assert agent.role.stop_on_no_tool is True


def test_chat_agent_full_tools_and_termination(chat_env, capsys):
    """全量工具放行 + 多步（工具→文本）+ 终止 + 自有打印，一气呵成。"""
    agent = make_chat_agent()
    _drive_chat(agent, "当前目录有啥文件", AgentDisplay(sink=print))

    # 1) 白名单外的工具被真正执行（allow_all_tools 生效）
    assert _TEST_TOOL in chat_env["dispatch_calls"], chat_env["dispatch_calls"]
    # 2) 模型被调用两次：第 1 次给工具、第 2 次给文本
    assert chat_env["scripted"].calls == 2
    # 3) 回复被打印到控制台（REPL 自有处理页）
    out = capsys.readouterr().out
    assert "[assistant]" in out
    assert "done: 当前目录有 file.py" in out
    assert f"调用工具：{_TEST_TOOL}" in out
    # 4) buffer 含工具结果与最终文本
    roles = [m["role"] for m in agent.ctx.cm.to_list()]
    assert "tool" in roles
    assert any(m.get("role") == "assistant" and "done: 当前目录有 file.py" in (m.get("content") or "")
              for m in agent.ctx.cm.to_list())


def test_chat_agent_text_only_terminates(chat_env, capsys):
    """模型直接给纯文本（不调工具）→ 立即 all_done，不再逼它调工具。"""
    chat_env["scripted"].tool_then_text = False  # 首次即返回文本
    chat_env["dispatch_calls"].clear()
    agent = make_chat_agent()
    _drive_chat(agent, "你好", AgentDisplay(sink=print))
    # 没调任何工具
    assert chat_env["dispatch_calls"] == []
    # 但回复仍被打印
    out = capsys.readouterr().out
    assert "done: 当前目录有 file.py" in out
    assert chat_env["scripted"].calls == 1


def test_normal_role_rejects_tool_outside_allowlist(chat_env):
    """对照：普通 executor 角色（无 allow_all_tools）会拒绝白名单外的工具——
    反证 ChatAgent 的 allow_all_tools 才是放行来源。"""
    exec_agent = make_agent("executor")
    reason = exec_agent._apply_toolcall({
        "type": "toolcalls",
        "tool_calls": [{"id": "cX", "name": _TEST_TOOL, "arguments": "{}"}]})
    # 工具被拒：产生一条 error 回执，且本轮未前进（continue）
    tool_msgs = [m for m in exec_agent.ctx.cm.to_list() if m["role"] == "tool"]
    assert tool_msgs, "executor 应产出一条 tool 回执（即使是拒绝）"
    assert "不允许调用" in tool_msgs[-1]["content"]
    assert reason == "continue"


def test_print_repl_msg_format(capsys):
    """REPL 自有打印格式：assistant 文本 + tool_calls 名 + tool 结果。"""
    display = AgentDisplay(sink=print)
    _print_repl_msg({"role": "assistant", "content": "hi",
                     "tool_calls": [{"id": "c", "type": "function",
                                     "function": {"name": "shell", "arguments": "{}"}}]}, display)
    _print_repl_msg({"role": "tool", "content": "ls output here"}, display)
    out = capsys.readouterr().out
    assert "[assistant]" in out
    assert "调用工具：shell" in out
    assert "[tool] ls output here" in out
