#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ContextManager × contextmgr 集成单测（model-free / tool-free）。

覆盖：RAG 命中、RAG 未命中自动跳过、prepare_messages 只读 buffer、
极小上下文窗口触发两级压缩不崩、ingest 走白名单、
model_context_length 解析（MODELS → 配置默认 96000）、
以及 agent._step 的 outgoing 组装契约（取末条 user 作 query + prepare_messages）。
"""

import os
import tempfile
import textwrap

from swe_agent.management import ContextManager
from swe_agent import models as M
from swe_agent import config as C
from contextmgr import ContextManager as RagEngine


def _last_user_text(messages):
    """复刻 agent._last_user_text：取 buffer 末条 user 内容作 RAG query。"""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            return c if isinstance(c, str) else ""
    return ""


def _make_cm_with_file(code: str):
    """建一个独立 kb（不碰共享 WORKSPACE KB），把单个 .py 喂进 RAG 知识库。"""
    kb = RagEngine(budget_tokens=4096)
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sample.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    cm = ContextManager(kb=kb)
    cm.ingest_code_file(path)
    cm.set_system("You are a dev agent.")
    return cm, d


def test_prepare_messages_rag_hit():
    code = textwrap.dedent("""
        def foo(x):
            # foo 计算平方
            return x * x

        class Bar:
            def baz(self):
                return 42
    """)
    cm, _ = _make_cm_with_file(code)
    cm.append("user", "请解释 foo 函数")
    cm.append("assistant", "foo 返回 x 的平方")
    out = cm.prepare_messages(model_context_length=64000, user_input="foo 函数是怎么实现的")

    assert out[0]["role"] == "system"          # ① system 在最前且不重复
    assert sum(1 for m in out if m["role"] == "system") == 1

    joined = "\n".join(m["content"] for m in out if m["role"] == "user")
    assert "我找到了如下相关信息" in joined      # ② RAG 命中
    assert "出处含文件与行号" in joined          # ③ 注入块保留出处（文件:行号），不再是纯指针表
    assert "### [1]" in joined                   # ④ 按相关性精选的正文条目
    assert "foo" in joined

    assert out[-1]["role"] == "user"
    assert "foo 函数是怎么实现的" in out[-1]["content"]  # ③ 当前轮 input 带 RAG


def test_prepare_messages_no_rag_when_empty_kb():
    kb = RagEngine(budget_tokens=4096)
    cm = ContextManager(kb=kb)                 # KB 空，不 ingest 任何东西
    cm.set_system("sys")
    cm.append("user", "hello")
    out = cm.prepare_messages(model_context_length=64000, user_input="完全无关的随机词zzzqwk")

    joined = "\n".join(m["content"] for m in out if m["role"] == "user")
    assert "我找到了如下相关信息" not in joined   # 空 KB 自动跳过 RAG，不强制
    assert out[-1]["content"] == "完全无关的随机词zzzqwk"


def test_prepare_messages_is_readonly():
    cm, _ = _make_cm_with_file("def a():\n    return 1\n")
    cm.append("user", "问 a")
    before = list(cm.to_list())
    cm.prepare_messages(64000, "问 a")
    assert cm.to_list() == before             # 只读 buffer，不缩历史


def test_prepare_messages_compression_triggers():
    kb = RagEngine(budget_tokens=4096)
    cm = ContextManager(kb=kb)
    cm.set_system("sys")
    cm.append("user", "start")
    for i in range(40):
        cm.append("assistant", "x" * 500 + f" step {i}")
    out = cm.prepare_messages(model_context_length=2000, user_input="start")

    assert out[0]["role"] == "system"
    assert sum(1 for m in out if m["role"] == "system") == 1
    total_chars = sum(len(m.get("content", "")) for m in out)
    # 40 条 * 500 字符原文明显超预算 → 压缩后总字符应远小于原文
    assert total_chars < 40 * 500


def test_ingest_code_file_builds_rag_fragments():
    cm, _ = _make_cm_with_file("def ping():\n    return 'pong'\n")
    # 直接走 kb.retrieve 验证 RAG 物料确实建好了
    refs = cm.kb.retrieve("ping", budget_tokens=2000, sources={"code"})
    assert any("ping" in (f.text or "") for f in refs)


def test_model_context_length_resolves_from_models_then_config_default():
    # 已登记模型：取 MODELS[...]["context_length"]（统一为配置默认 96000）
    assert M.model_context_length("qwen2.5.1-coder-7b-instruct") == C.MODEL_CONTEXT_LENGTH
    # 未登记模型：回退配置默认
    assert M.model_context_length("nonexistent-model-xyz") == C.MODEL_CONTEXT_LENGTH
    # 空模型 id：回退配置默认
    assert M.model_context_length(None) == C.MODEL_CONTEXT_LENGTH
    # 配置默认值确为 96000
    assert C.MODEL_CONTEXT_LENGTH == 96000


def test_prepare_messages_no_user_duplicate_after_tool_call():
    """回归：tool_call 轮后，原始 user input 不得被重复塞入、RAG 表不得重复追加。

    复现用户反馈的 bug：buffer=[user(orig), assistant(toolcalls), tool(result)] 时，
    prepare_messages 曾在末尾再拼一条 user(orig+RAG)，导致 orig 出现两次、RAG 每轮重复。
    修复后：处于 tool_call 循环中（末尾是 tool/assistant）不再把当前用户轮重复注入，
    user input 保留在原本位置（仅 1 条），tool 结果也保留，且不再凭空追加 RAG。
    """
    code = textwrap.dedent("""
        def ping():
            return 'pong'
    """)
    cm, _ = _make_cm_with_file(code)
    cm.set_system("You are a dev agent.")
    cm.append("user", "实现 ping 函数")          # 原始 user 轮
    cm.append("assistant", None, tool_calls=[{
        "id": "c1", "type": "function",
        "function": {"name": "write_file", "arguments": "{}"},
    }])
    cm.append("tool", "write_success: src/ping.py")  # tool 结果，在 user 轮之后

    # 末条 user 仍是“实现 ping 函数”——与 _step 取 query 行为一致
    user_input = _last_user_text(cm.to_list())
    assert user_input == "实现 ping 函数"

    out = cm.prepare_messages(
        model_context_length=M.model_context_length("qwen2.5.1-coder-7b-instruct"),
        user_input=user_input,
    )
    user_msgs = [m for m in out if m["role"] == "user"]
    # 关键不变量：user 消息恰好 1 条，不得重复塞入
    assert len(user_msgs) == 1, f"user 消息被重复塞入：{len(user_msgs)} 条"
    assert user_msgs[0]["content"] == "实现 ping 函数"
    # tool 结果仍保留
    assert any(m["role"] == "tool" for m in out)
    # 末尾是 tool 结果，不应再凭空追加一条带 RAG 的 user 消息
    assert out[-1]["role"] == "tool"
    # tool 轮不再重复追加 RAG
    joined = "\n".join(m.get("content", "") for m in out if m["role"] == "user")
    assert "我找到了如下相关信息" not in joined


def test_step_assembly_contract_uses_prepare_messages():
    """复刻 agent._step 的 outgoing 组装：取末条 user 文本作 query → prepare_messages。

    验证 agent 主路径接电后：输出单 system + RAG 表、且 prepare_messages 确实被 consume。
    """
    code = textwrap.dedent("""
        def ping():
            return 'pong'
    """)
    cm, _ = _make_cm_with_file(code)
    cm.set_system("You are a dev agent.")
    cm.append("user", "实现 ping 函数")
    cm.append("assistant", "我来实现")
    cm.append("user", "ping 怎么实现")  # 末条 user → RAG query

    user_input = _last_user_text(cm.to_list())  # 复刻 _step 的取 query
    out = cm.prepare_messages(
        model_context_length=M.model_context_length("qwen2.5.1-coder-7b-instruct"),
        user_input=user_input,
    )
    assert out[0]["role"] == "system"
    assert sum(1 for m in out if m["role"] == "system") == 1
    joined = "\n".join(m["content"] for m in out if m["role"] == "user")
    assert "我找到了如下相关信息" in joined
    assert "ping" in joined  # 末条 user query 触发了 RAG 命中
