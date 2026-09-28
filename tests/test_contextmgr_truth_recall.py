"""contextmgr —— F1 压缩缺陷 + 真相日志(truth) + recall 单测。

model-free / tool-free。覆盖：
- F1：极端预算（连最新一轮都装不下）下，压缩不再把历史清空成只剩 system，
      而是轮内子单元降级（丢 assistant(+tool)，保 user 问题）。
- F2/F3：SLIDING_WINDOW（仅尾部）与 COMPRESS（首+尾+占位）语义可区分。
- 真相日志：SweCM.append(original=) 把原文落进 _truth，工作集 _msgs 存压缩版。
- recall：压缩掉的历史细节可按 query 从 _truth 取回；prepare_messages 支持 opt-in 注入。
- 持久化：supervisor.save_session 把 truth 并列落盘，load_session 可恢复。
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from contextmgr import CompressionStrategy, compress
from swe_agent.management import ContextManager
from swe_agent import supervisor as _supervisor


class _StubKB:
    """避免构造期扫整个工作区的桩 RAG 后端。"""

    def retrieve(self, *a, **k):
        return []

    def ingest_code_file(self, *a, **k):
        return self

    def ingest_library(self, *a, **k):
        return self


def _cm(messages=None, kb=None):
    cm = ContextManager(messages=list(messages or []), kb=kb or _StubKB())
    # 关掉二级 LFM 压缩后端（否则会触网络）；纯 model-free 路径。
    cm.compress_backend = None
    return cm


# ---------------------------------------------------------------- F1
def test_f1_intraturn_fallback_keeps_user_question():
    """单轮（user 小 + assistant(tool_calls) + 巨大 tool 结果）超预算时：
    丢 assistant(+tool)，保 user 问题，绝不退回「只剩 system」。"""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "帮我读文件"},          # 小，应保住
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": "r" * 4000, "tool_call_id": "t1"},  # 巨大，应丢
    ]
    out = compress(msgs, budget_tokens=200, strategy=CompressionStrategy.COMPRESS,
                   compress_backend=None)
    assert out[0]["role"] == "system"
    assert len(out) >= 2, f"F1 失败：历史被清空成只剩 system（{len(out)} 条）"
    # user 问题保住
    assert any(m["role"] == "user" and m["content"] == "帮我读文件" for m in out)
    # 巨大 tool 结果被丢（不占预算）
    assert not any("r" * 4000 in (m.get("content") or "") for m in out)
    # 协议：无孤立 tool（此处本就无 tool 残留）
    declared = {c["id"] for m in out if m.get("tool_calls") for c in m["tool_calls"]}
    answered = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
    assert declared == answered


def test_f1_single_round_overflow_no_backend_keeps_user():
    """对照原 test_compress_single_round_overflow_no_backend：F1 后仍不超预算、不孤立 tool、
    且小体积 user 问题被保住（巨大工具结果/assistant 被丢）。"""
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
    from contextmgr.compress import _count
    assert _count(out) <= 200
    assert any(m["role"] == "user" and m["content"] == "请读文件" for m in out)
    declared = {c["id"] for m in out if m.get("tool_calls") for c in m["tool_calls"]}
    answered = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
    assert declared == answered


# ---------------------------------------------------------------- F2/F3
def test_compress_vs_sliding_distinct():
    """F2/F3：COMPRESS 保首部（HEAD_UNIQUE），SLIDING 仅保尾部（丢首部）。二者语义可区分。"""
    buf = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "HEAD_UNIQUE_QUESTION"},
        {"role": "assistant", "content": "a1 " + "x" * 50},
        {"role": "user", "content": "middle q"},
        {"role": "assistant", "content": "a2 " + "x" * 50},
        {"role": "user", "content": "tail q"},
        {"role": "assistant", "content": "a3 " + "x" * 50},
    ]
    out_c = compress(buf, budget_tokens=10_000, strategy=CompressionStrategy.COMPRESS, window=6)
    out_s = compress(buf, budget_tokens=10_000, strategy=CompressionStrategy.SLIDING_WINDOW, window=4)
    # COMPRESS：首部 HEAD_UNIQUE 仍在
    assert any("HEAD_UNIQUE_QUESTION" in (m.get("content") or "") for m in out_c)
    # SLIDING：仅尾部窗口，HEAD_UNIQUE 已被丢弃
    assert not any("HEAD_UNIQUE_QUESTION" in (m.get("content") or "") for m in out_s)


# ---------------------------------------------------------------- 真相日志 + recall
def test_swecm_truth_preserves_original_after_compress():
    """append(original=) 把原文落进 _truth；工作集 _msgs 存压缩版（result_compress 后）。"""
    cm = _cm()
    # 模拟 agent：压缩前原文 + 压缩后文本
    cm.append("tool", "[抽取式压缩 · 原文 5000 字符]\n压缩后摘要", original="原始超长工具结果" * 100)
    assert cm.to_list()[-1]["content"] == "[抽取式压缩 · 原文 5000 字符]\n压缩后摘要"
    assert cm.truth_list()[-1]["content"] == "原始超长工具结果" * 100
    # 普通消息：原文 == content
    cm.append("user", "普通问题")
    assert cm.truth_list()[-1]["content"] == "普通问题"


def test_swecm_recall_returns_dropped_detail():
    """压缩掉的早期历史细节，可按 query 从 _truth 取回。"""
    cm = _cm()
    for i in range(30):
        cm.append("user", f"关于主题{i}的问题")
        cm.append("assistant", "答" * 200)
    cm.compress_if_needed()  # 压缩工作集 _msgs（_truth 不动）
    # _truth 长度不变（仍是完整历史）
    assert len(cm.truth_list()) == 60
    # recall 命中早期被压缩掉的主题3
    got = cm.recall("主题3")
    assert got, "recall 未取回任何内容"
    assert any("主题3" in (m.get("content") or "") for m in got)


def test_prepare_messages_optin_recall_injection():
    """prepare_messages(recall_query=) 在压缩掉历史时注入「历史补充」system 块。"""
    cm = _cm()
    for i in range(30):
        cm.append("user", f"关于主题{i}的问题")
        cm.append("assistant", "答" * 200)
    # 极小预算强制压缩；recall_query 命中早期主题
    out = cm.prepare_messages(model_context_length=800, user_input="主题2",
                              recall_query="主题2")
    blocks = [m for m in out if "历史补充" in (m.get("content") or "")]
    assert blocks, "opt-in recall 未注入历史补充块"
    assert "主题2" in blocks[0]["content"]


def test_prepare_messages_recall_off_is_unchanged():
    """recall_query 缺省时，prepare_messages 行为与改动前完全一致（无历史补充块）。"""
    cm = _cm()
    for i in range(30):
        cm.append("user", f"关于主题{i}的问题")
        cm.append("assistant", "答" * 200)
    out = cm.prepare_messages(model_context_length=800, user_input="主题2")
    assert not any("历史补充" in (m.get("content") or "") for m in out)


# ---------------------------------------------------------------- 持久化
def test_save_session_persists_truth(monkeypatch):
    """supervisor.save_session 把 truth 并列落盘；load_session 可恢复。"""
    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(_supervisor.config, "SESSIONS_DIR", pathlib.Path(tmp))
    _supervisor.save_session(
        "s-truth", [{"role": "user", "content": "hi"}],
        truth=[{"role": "user", "content": "hi-original"}])
    data = _supervisor.load_session("s-truth")
    assert data is not None
    assert data.get("truth") == [{"role": "user", "content": "hi-original"}]
    # 默认（未传 truth）→ 退回 messages，不丢
    _supervisor.save_session("s-nodef", [{"role": "user", "content": "x"}])
    assert _supervisor.load_session("s-nodef")["truth"] == [{"role": "user", "content": "x"}]
