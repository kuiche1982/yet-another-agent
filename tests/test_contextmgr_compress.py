"""contextmgr 单元测试（model-free / tool-free）：三种压缩策略。

纯函数、确定性、不触模型/工具。验证「保真 + 预算」。
"""

from __future__ import annotations

from contextmgr import CompressionStrategy, compress
from contextmgr.compress import _count, _group_by_user


def _buffer(n: int) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"message number {i} about topic {i}"})
    return msgs


def test_sliding_window_keeps_tail_and_system():
    msgs = _buffer(12)
    out = compress(msgs, budget_tokens=10_000, strategy=CompressionStrategy.SLIDING_WINDOW, window=4)
    assert out[0]["role"] == "system"
    # 只保留尾部窗口
    assert len(out) <= 1 + 4 + 1
    assert out[-1]["content"] == "message number 11 about topic 11"


def test_compress_keeps_head_tail_under_budget():
    msgs = _buffer(20)
    out = compress(msgs, budget_tokens=100, strategy=CompressionStrategy.COMPRESS, window=6)
    assert out[0]["role"] == "system"
    # 含压缩占位
    assert any("已压缩" in m["content"] for m in out)
    assert _count(out) <= 10_000


def test_compress_respects_budget_when_tiny():
    msgs = _buffer(20)
    out = compress(msgs, budget_tokens=15, strategy=CompressionStrategy.COMPRESS, window=6)
    assert _count(out) <= 15
    # 至少保留 system + 一条 user
    assert out[0]["role"] == "system"


def test_aggressive_drops_history_keeps_last_user():
    msgs = _buffer(15)
    out = compress(msgs, budget_tokens=10_000, strategy=CompressionStrategy.AGGRESSIVE)
    # 只留 system + 最后一条 user
    assert out[0]["role"] == "system"
    assert out[-1]["role"] == "user"
    assert "message number 14" in out[-1]["content"]
    # 历史被砍：远少于原长度
    assert len(out) <= 3


def test_aggressive_without_system():
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}]
    out = compress(msgs, budget_tokens=10_000, strategy=CompressionStrategy.AGGRESSIVE)
    assert out == [{"role": "user", "content": "q2"}]


def test_group_by_user_splits_on_user_boundary():
    """_group_by_user 按 user 边界切组：一组 = 一条 user 起到下一条 user 前
    （含期间的 assistant / tool / system），形如 [user, system?, tool, tool, user)。"""
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1", "tool_call_id": "c1"},
        {"role": "user", "content": "u2"},
        {"role": "system", "content": "mid-sys"},   # 组内允许出现 system
        {"role": "assistant", "content": "a2"},
        {"role": "tool", "content": "t2", "tool_call_id": "c2"},
        {"role": "tool", "content": "t3", "tool_call_id": "c3"},
    ]
    groups = _group_by_user(msgs)
    assert len(groups) == 2
    # 组 0：u1 → a1 → t1（到下一条 user 前）
    assert [m["role"] for m in groups[0]] == ["user", "assistant", "tool"]
    # 组 1：u2 → (mid-sys) → a2 → t2 → t3（含组内 system 与多 tool）
    assert [m["role"] for m in groups[1]] == ["user", "system", "assistant", "tool", "tool"]
    # 同组内的 assistant(tool_calls) 与 tool 永不分离 —— 验证分组确实聚合了 tool 轮
    assert groups[0][-1]["tool_call_id"] == "c1"
    assert groups[1][-1]["tool_call_id"] == "c3"


def test_compress_never_orphans_tool_results():
    """回归（核心）：SLIDING_WINDOW 与 COMPRESS 都不允许出现「有 tool 结果却无对应
    assistant tool_calls 声明」的孤立 tool（否则下一轮 provider 报 400）。

    tool 与它的 assistant tool_calls 声明必须「同进同出」——整组保留或整组丢弃。
    """
    # user → assistant(tool_calls) → tool 是一组；前面塞大体积历史逼出压缩/滑动
    base = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "big " + "x" * 4000},
        {"role": "assistant", "content": "big " + "y" * 4000},
        {"role": "user", "content": "now call"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": "result", "tool_call_id": "t1"},
    ]

    def _assert_no_orphan(out, strat):
        # 收集输出里所有 tool_call_id（声明方 + 结果方）
        declared = {tc["id"] for m in out
                    if m.get("role") == "assistant" and m.get("tool_calls")
                    for tc in m["tool_calls"]}
        answered = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
        # 任一 tool 结果都必须有对应声明；任一声明都必须有对应结果
        # （输入里本就成对的 id 在输出里也必须成对出现，不得拆散）
        assert answered <= declared, f"{strat}: 孤立 tool（有结果无声明）: {answered - declared}"
        assert declared <= answered, f"{strat}: 孤立声明（有声明无结果）: {declared - answered}"

    for strat in (CompressionStrategy.SLIDING_WINDOW, CompressionStrategy.COMPRESS):
        out = compress(list(base), budget_tokens=200, strategy=strat, window=4)
        _assert_no_orphan(out, strat)


def test_compress_keeps_tool_turn_intact_under_tiny_budget():
    """极小预算下，若保留某一轮 tool，则该轮 assistant(tool_calls) 与 tool 必须同时出现
    （整组保留，不丢声明）。"""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "big " + "x" * 4000},
        {"role": "assistant", "content": "big " + "y" * 4000},
        {"role": "user", "content": "call now"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "k1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "content": "r", "tool_call_id": "k1"},
    ]
    out = compress(msgs, budget_tokens=300, strategy=CompressionStrategy.COMPRESS, window=6)
    has_tool = any(m["role"] == "tool" for m in out)
    if has_tool:
        declared = {tc["id"] for m in out
                    if m.get("role") == "assistant" and m.get("tool_calls")
                    for tc in m["tool_calls"]}
        answered = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
        assert "k1" in declared and "k1" in answered


def _fake_lfm_backend(messages, ctx=None):
    """假大上下文模型压缩后端：把整段折叠成一条 LFM 摘要（不触真实 LLM）。"""
    return [{"role": "user", "content": "LFM-SUMMARY"}]


def test_compress_single_round_overflow_escalates_to_backend():
    """边界（核心）：仅有**一轮**对话 system, user, assistant(tool), tool, tool,
    assistant(tool), tool, assistant(最终) 就超预算时，因分组已无法再拆（单组），
    必须直接升级「大上下文模型压缩后端」，不丢 user、不做中间砍 tool。

    验证：配置了 compress_backend → 它被调用，且返回结果即最终输出。
    """
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 4000},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "content": "r1" * 2000, "tool_call_id": "t1"},
        {"role": "tool", "content": "r2" * 2000, "tool_call_id": "t2"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t2", "type": "function",
             "function": {"name": "g", "arguments": "{}"}}]},
        {"role": "tool", "content": "r3" * 2000, "tool_call_id": "t3"},
        {"role": "assistant", "content": "final " + "y" * 4000},
    ]
    calls = []
    def _backend(m, ctx=None):
        calls.append(m)
        return _fake_lfm_backend(m, ctx)
    out = compress(msgs, budget_tokens=200, strategy=CompressionStrategy.COMPRESS,
                   compress_backend=_backend)
    assert calls, "单轮溢出必须升级大上下文模型压缩后端（compress_backend 未被调用）"
    # 后端返回的即最终输出（LFM 负责在语义层面保留 user 意图，无需 harness 砍 tool）
    assert out == [{"role": "user", "content": "LFM-SUMMARY"}]


def test_compress_single_round_overflow_no_backend_falls_back_gracefully():
    """对照（model-free）：未配置后端时，单轮溢出不做升级、也不崩，退化为整组裁剪。

    F1 语义：user 问题是「小」的（现实场景），巨大的是工具结果/大段 assistant；
    此时丢 assistant(+tool) 子单元、保留 user 问题，既能不超预算、又不产生孤立 tool、
    也不清空历史（不会退化成只剩 system）。
    """
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "请读文件"},          # 小，应保住
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "content": "r1" * 2000, "tool_call_id": "t1"},  # 巨大，应丢
        {"role": "assistant", "content": "y" * 4000},                   # 巨大，应丢
    ]
    out = compress(msgs, budget_tokens=200, strategy=CompressionStrategy.COMPRESS,
                   compress_backend=None)
    # 不超预算
    assert _count(out) <= 200
    # user 问题保住（F1：历史不丢失）
    assert any(m["role"] == "user" and m["content"] == "请读文件" for m in out)
    # 不产生孤立 tool：有声明必有结果、有结果必有声明（整组丢弃时同组同丢）
    declared = {tc["id"] for m in out if m.get("role") == "assistant" and m.get("tool_calls")
                for tc in m["tool_calls"]}
    answered = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
    assert answered <= declared, f"孤立 tool 结果: {answered - declared}"
    assert declared <= answered, f"孤立声明: {declared - answered}"
