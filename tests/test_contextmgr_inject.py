#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""contextmgr.inject 守卫：model-free 默认、fail-open、忠实性闸门、调用配额。

这些不变量是「注入层下沉」能验收的前提 —— 守住 design doc §8：contextmgr 单测必须
model-free / tool-free（不触 LLM、不触向量模型、不触网络）。故此处所有「模型」都是
注入进来的**假后端**（计数器 / 抛异常 / 返回垃圾），真模型一次都不碰。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextmgr import InjectOptions, clip, format_local_search, rag_block, ref_location
from contextmgr.types import Fragment, Source

# 远超 max_chars(=200 下限)，确保走「截断」分支
LONG_MULTILINE = "\n".join("line_%02d %s" % (i, "y" * 30) for i in range(1, 15))
SHORT = "short body"


def _frag(fid, text, source=Source.CODE, origin="a/b.py", lineno=0, end_lineno=0, score=0.5):
    return Fragment(fid=fid, source=source, text=text, origin=origin,
                    lineno=lineno, end_lineno=end_lineno, score=score)


def _opts(**kw):
    base = dict(max_chars=200, allow_summary=False, summary_min_tokens=0,
                summary_max_calls=3, summary_model="fake-model", snippet_chars=160, top_n=5)
    base.update(kw)
    return InjectOptions(**base)


def test_allow_summary_without_backend_stays_model_free():
    """allow_summary=True 但未注入 summarize → 行为与关闭摘要逐字节一致（零模型调用）。"""
    refs = [_frag("f1", LONG_MULTILINE)]
    off = rag_block(refs, "q", _opts(allow_summary=False))
    on = rag_block(refs, "q", _opts(allow_summary=True), summarize=None)
    assert on == off, "summary must be a no-op when no summarize backend is injected"
    assert "…（中间省略）…" in on


def test_summarize_backend_exception_falls_back_to_extraction():
    """后端抛异常 → fail-open：回退抽取式截断，不抛给调用方。"""
    calls = []

    def boom(query, text, model):
        calls.append(model)
        raise RuntimeError("backend down")

    refs = [_frag("f1", LONG_MULTILINE)]
    out = rag_block(refs, "q", _opts(allow_summary=True), summarize=boom)
    assert calls == ["fake-model"], "backend should have been attempted once"
    assert out == rag_block(refs, "q", _opts(allow_summary=False)), \
        "failed summary must fall back to the exact extraction path"


def test_short_summary_is_adopted_and_drops_the_truncation_marker():
    refs = [_frag("f1", LONG_MULTILINE)]
    out = rag_block(refs, "q", _opts(allow_summary=True),
                    summarize=lambda q, t, m: SHORT)
    assert SHORT in out
    assert "…（中间省略）…" not in out, "adopted summary must not keep the clip marker"
    assert "（已截断，读全文：" not in out


def test_longer_summary_is_rejected_by_faithfulness_gate():
    """模型输出比原文更长 → 判为不可信，保留抽取式截断结果。"""
    refs = [_frag("f1", LONG_MULTILINE)]
    out = rag_block(refs, "q", _opts(allow_summary=True),
                    summarize=lambda q, t, m: t + " EXTRA" * 20)
    assert "…（中间省略）…" in out, "longer-than-source summary must be rejected"


def test_summary_ready_gate_blocks_before_consuming_quota():
    """summarize_ready=False → 后端一次都不该被调用（也不消耗 max_calls 配额）。"""
    calls = []

    def spy(q, t, m):
        calls.append(m)
        return SHORT

    refs = [_frag("f1", LONG_MULTILINE), _frag("f2", LONG_MULTILINE)]
    out = rag_block(refs, "q", _opts(allow_summary=True), summarize=spy,
                    summarize_ready=lambda model: False)
    assert calls == [], "summarize_ready gate must short-circuit the backend"
    assert "…（中间省略）…" in out


def test_summary_calls_are_capped_by_max_calls():
    calls = []

    def spy(q, t, m):
        calls.append(t)
        return SHORT

    refs = [_frag("f1", LONG_MULTILINE), _frag("f2", LONG_MULTILINE)]
    rag_block(refs, "q", _opts(allow_summary=True, summary_max_calls=1), summarize=spy)
    assert len(calls) == 1, "max_calls must cap model round-trips per injection"


def test_summary_skipped_below_min_tokens():
    calls = []
    refs = [_frag("f1", LONG_MULTILINE)]
    rag_block(refs, "q", _opts(allow_summary=True, summary_min_tokens=10 ** 6),
              summarize=lambda q, t, m: calls.append(t) or SHORT)
    assert calls == [], "short-enough fragments must not pay a model round-trip"


def test_short_fragment_never_triggers_summary():
    calls = []
    refs = [_frag("f1", SHORT)]
    out = rag_block(refs, "q", _opts(allow_summary=True),
                    summarize=lambda q, t, m: calls.append(t) or "x")
    assert calls == [] and SHORT in out


def test_ref_location_without_kb_degrades_to_raw_origin():
    """无 KB 时 library 片段不能凭空编路径，应回落到 origin。"""
    frag = _frag("f1", "x", source=Source.LIBRARY, origin="kb:doc", lineno=3, end_lineno=5)
    loc, pointer = ref_location(None, frag)
    assert loc == "kb:doc:3-5", loc
    assert pointer == "read_file kb:doc offset=3 limit=3", pointer


def test_clip_zero_budget_returns_text_untouched():
    assert clip("abc", 0) == ("abc", False)
    assert clip("", 100) == ("", False)


def test_format_local_search_numbers_pages_from_offset():
    refs = [("proj", _frag("f1", "body one", score=0.9)),
            ("code", _frag("f2", "body two", score=0.4))]
    out = format_local_search(refs, scope="all", offset=2, snippet_chars=160)
    assert "第 3-4 条" in out, out
    assert "3. [proj] " in out and "4. [code] " in out, out
    assert "相关性 0.90" in out and "相关性 0.40" in out, out
