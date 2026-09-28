#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG retrieval + injection layer contracts (model-free / tool-free).

Guards the 2026-09-12 fixes for "contextmanager search 后没有再确定相关性、只注入文件位置":
- cross-layer relevance uses MERGED-CORPUS BM25, so each layer's best no longer ties at 1.00
  (per-layer max-normalize made cross-layer order degenerate to raw BM25 comparison, which is
  exactly the comparison that is not valid across layers);
- the injected block carries the L0 body + provenance (real file:line), not bare pointers;
- over-cap fragments fall back to extractive clip and keep a read_file pointer;
- top-N caps how many refs get injected;
- the optional model summary / rerank is LAZY (never called unless the model is resident);
- local_search pagination (offset) and line-aware snippet clipping.
"""

from __future__ import annotations

import pytest

from contextmgr import ContextManager as RagEngine
from contextmgr import Fragment, Source
from swe_agent import config as C
from swe_agent import models as M
from swe_agent.management import ContextManager, LayeredKB, _parse_order, _ref_location


@pytest.fixture(autouse=True)
def _no_model_rerank(monkeypatch):
    """Keep this module model-free: the optional model rerank must never fire by default."""
    monkeypatch.setattr(C, "RAG_RERANK", False, raising=False)


def _frag(origin, text, score, source=Source.LIBRARY, lineno=0, end_lineno=0, offset=0):
    return Fragment(fid=f"{origin}#{offset}", source=source, text=text, origin=origin,
                    score=score, lineno=lineno, end_lineno=end_lineno, offset=offset)


class _StubEngine:
    """Minimal engine stub that returns a fixed (already ranked) fragment list."""

    def __init__(self, frags):
        self._frags = list(frags)

    def retrieve(self, query, budget_tokens=2048, sources=None, top_k=8):
        return list(self._frags)[:top_k]


def _stub_layered(code_frags=(), proj_frags=(), global_frags=()):
    kb = LayeredKB.__new__(LayeredKB)  # bypass the on-disk build
    kb.code_kb = _StubEngine(code_frags)
    kb.proj_kb = _StubEngine(proj_frags)
    kb.global_kb = _StubEngine(global_frags)
    kb._lib_path_index = {}
    return kb


def test_rerank_prefers_relevant_kb_over_weak_code_hits():
    """The original complaint: weak code hits used to outrank the truly relevant KB doc.

    Only the KB fragment matches the query, so merged-corpus BM25 scores it above 0 and the code
    hits exactly 0; the KB doc must come first regardless of the per-layer raw scores.
    """
    kb = _stub_layered(
        code_frags=[_frag("judge/probe.py", "weak code hit", 1.0, Source.CODE, offset=0),
                    _frag("tests/other.py", "weaker code hit", 0.5, Source.CODE, offset=1)],
        proj_frags=[_frag("kb:workflow", "Judge-Mode design doc", 6.0, offset=0)],
    )
    out = kb.retrieve("judge mode", budget_tokens=4096)
    order = [f.origin for f in out]
    assert order == ["kb:workflow", "judge/probe.py", "tests/other.py"], f"relevance order wrong: {order}"
    assert out[0].score == pytest.approx(1.0)
    # merged-corpus scoring: only the query-matching doc scores > 0; non-matching hits are 0
    assert out[1].score == pytest.approx(0.0)
    assert out[2].score == pytest.approx(0.0)


def test_rerank_falls_back_to_layer_normalize_when_nothing_matches():
    """When no candidate matches the query, keep the old per-layer normalize + raw tie-break."""
    kb = _stub_layered(
        code_frags=[_frag("a.py", "code best", 1.0, Source.CODE, offset=0),
                    _frag("b.py", "code second", 0.5, Source.CODE, offset=1)],
        proj_frags=[_frag("kb:x", "kb best", 5.0, offset=0)],
    )
    out = kb.retrieve("zzz-no-overlap", budget_tokens=4096)
    assert [f.origin for f in out] == ["kb:x", "a.py", "b.py"]
    assert out[0].score == pytest.approx(1.0)
    assert out[2].score == pytest.approx(0.5)


def test_search_all_returns_global_top_k_not_layer_quota():
    """`all` must be a cross-layer relevance top-k, not a fixed per-layer quota.

    Uses the same all-zero-score fallback as above (stub text never matches the query).
    """
    kb = _stub_layered(
        code_frags=[_frag("a.py", "code best", 1.0, Source.CODE, offset=0)],
        proj_frags=[_frag("kb:x", "kb best", 5.0, offset=0),
                    _frag("kb:y", "kb second", 4.0, offset=1)],
    )
    res = kb.search("anything", scope="all", top_k=3)
    order = [f.origin for _, f in res]
    # layer-relative relevance: kb:x (1.0) beats a.py (1.0) on raw tie-break,
    # then a.py (1.0, layer best) beats kb:y (0.8, layer second).
    assert order == ["kb:x", "a.py", "kb:y"], f"unexpected order: {order}"


def test_injection_block_carries_body_and_provenance():
    """The injected block must include the L0 body and a real file:line provenance."""
    kb = RagEngine(budget_tokens=4096)
    kb.ingest_library("Python 虚拟环境隔离依赖。\n第二行补充说明。\n\n无关段落：午饭吃什么。",
                      doc_id="kb:venv")
    cm = ContextManager(kb=kb)
    cm.set_system("sys")
    cm.append("user", "venv 怎么隔离依赖")
    out = cm.prepare_messages(model_context_length=64000, user_input="venv 怎么隔离依赖")
    joined = "\n".join(m["content"] for m in out if m["role"] == "user")
    assert "我找到了如下相关信息" in joined
    assert "Python 虚拟环境隔离依赖" in joined   # L0 body itself is injected, not just a pointer
    assert "kb:venv:1-2" in joined               # library fragment now carries a line range
    assert "相关性" in joined


def test_over_cap_fragment_is_clipped_and_keeps_pointer(monkeypatch):
    monkeypatch.setattr(C, "RAG_INJECT_MAX_CHARS", 120, raising=False)
    monkeypatch.setattr(C, "RAG_INJECT_SUMMARY", False, raising=False)
    kb = RagEngine(budget_tokens=4096)
    kb.ingest_library("HEAD-" + "x" * 400 + "-TAIL\n\nunrelated text", doc_id="kb:big")
    cm = ContextManager(kb=kb)
    cm.set_system("sys")
    cm.append("user", "HEAD")
    out = cm.prepare_messages(model_context_length=64000, user_input="HEAD")
    joined = "\n".join(m["content"] for m in out if m["role"] == "user")
    assert "…（中间省略）…" in joined
    assert "读全文：" in joined and "read_file" in joined
    assert "HEAD-" in joined


def test_top_n_caps_injected_refs(monkeypatch):
    monkeypatch.setattr(C, "RAG_INJECT_TOP_N", 2, raising=False)
    kb = RagEngine(budget_tokens=4096)
    for i in range(6):
        kb.ingest_library(f"alpha keyword block number {i}", doc_id=f"kb:d{i}")
    cm = ContextManager(kb=kb)
    cm.set_system("sys")
    cm.append("user", "alpha")
    assert len(cm._rag_refs("alpha", 2048)) == 2


def test_summary_path_is_lazy_for_short_content(monkeypatch):
    """Short fragments must never probe the summarizer model (keeps tests model-free)."""

    def _boom(model):
        raise AssertionError("summarizer must not be consulted for short content")

    monkeypatch.setattr(ContextManager, "_summary_model_ready", staticmethod(_boom))
    kb = RagEngine(budget_tokens=4096)
    kb.ingest_library("short doc about caching", doc_id="kb:s")
    cm = ContextManager(kb=kb)
    cm.set_system("sys")
    cm.append("user", "caching")
    out = cm.prepare_messages(model_context_length=64000, user_input="caching")
    joined = "\n".join(m["content"] for m in out if m["role"] == "user")
    assert "caching" in joined


def test_ref_location_resolves_code_and_library_paths():
    code = _frag("swe_agent/x.py", "def f(): pass", 1.0, Source.CODE, lineno=3, end_lineno=4)
    loc, ptr = _ref_location(None, code)
    assert loc == "swe_agent/x.py:3-4"
    assert ptr == "read_file swe_agent/x.py offset=3 limit=2"

    lib = _frag("kb:doc", "text", 1.0, Source.LIBRARY, lineno=1, end_lineno=5)
    loc2, _ = _ref_location(None, lib)
    assert loc2 == "kb:doc:1-5"


def test_merged_corpus_bm25_makes_cross_layer_scores_comparable():
    """A query-matching KB doc must win on CONTENT, and top scores must not tie at 1.00.

    Regression guard: per-layer max-normalization made every layer's best hit exactly 1.00, so
    cross-layer order degenerated to comparing raw BM25 across layers - the very comparison that
    is invalid across layers. Here the code stub has the far higher raw score (9.0 vs 0.7) but
    weaker query coverage, so a raw comparison would rank it first. Merged-corpus scoring must
    rank by content instead and yield distinct scores.
    """
    kb = _stub_layered(
        code_frags=[_frag("a.py", "judge mode helper", 9.0, Source.CODE, offset=0)],
        proj_frags=[_frag("kb:doc", "judge mode 是什么：软裁判用 LLM 判定语义质量",
                          0.7, offset=0)],
    )
    out = kb.retrieve("judge mode 是什么", budget_tokens=4096)
    order = [f.origin for f in out]
    assert order == ["kb:doc", "a.py"], f"merged-corpus order wrong: {order}"
    assert out[0].score == pytest.approx(1.0)
    assert out[1].score < 1.0, "cross-layer scores must not tie at 1.00 (per-layer normalize bug)"


def _three_way_kb():
    return _stub_layered(
        code_frags=[_frag("a.py", "alpha code", 1.0, Source.CODE, offset=0)],
        proj_frags=[_frag("kb:x", "alpha doc one", 1.0, offset=0),
                    _frag("kb:y", "alpha doc two", 1.0, offset=1)],
    )


def test_model_rerank_reorders_when_model_is_ready(monkeypatch):
    monkeypatch.setattr(C, "RAG_RERANK", True, raising=False)
    monkeypatch.setattr(ContextManager, "_summary_model_ready", staticmethod(lambda m: True))
    kb = _three_way_kb()
    base = [f.origin for f in kb.retrieve("alpha", budget_tokens=4096)]
    assert len(base) == 3
    monkeypatch.setattr(M, "chat_text", lambda **kw: '{"order":[3,1,2]}')
    out = kb.retrieve("alpha", budget_tokens=4096)
    assert [f.origin for f in out][:2] == [base[2], base[0]]
    assert sorted(f.origin for f in out) == sorted(base)


def test_model_rerank_is_fail_open_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(C, "RAG_RERANK", True, raising=False)
    monkeypatch.setattr(ContextManager, "_summary_model_ready", staticmethod(lambda m: True))
    kb = _three_way_kb()
    base = [f.origin for f in kb.retrieve("alpha", budget_tokens=4096)]
    monkeypatch.setattr(M, "chat_text", lambda **kw: "sorry, I cannot rank that")
    assert [f.origin for f in kb.retrieve("alpha", budget_tokens=4096)] == base


def test_model_rerank_never_called_when_model_not_ready(monkeypatch):
    """Lazy gate: an unloaded model must cost nothing (no request, no retry)."""
    monkeypatch.setattr(C, "RAG_RERANK", True, raising=False)
    monkeypatch.setattr(ContextManager, "_summary_model_ready", staticmethod(lambda m: False))
    calls = []
    monkeypatch.setattr(M, "chat_text", lambda **kw: calls.append(kw) or "")
    _three_way_kb().retrieve("alpha", budget_tokens=4096)
    assert not calls


def test_parse_order_rejects_invalid_indices():
    assert _parse_order('{"order":[2,1,3]}', 3) == [2, 1, 3]
    assert _parse_order('```json\n{"order":[1,1,9,0]}```', 2) == [1]
    assert _parse_order("no json here", 3) == []
    assert _parse_order('{"order":"nope"}', 3) == []


def test_search_offset_paginates_without_overlap():
    kb = _stub_layered(proj_frags=[
        _frag(f"kb:d{i}", f"alpha block number {i}", float(10 - i), offset=i)
        for i in range(6)])
    page1 = [f.origin for _, f in kb.search("alpha", scope="kb", top_k=3, offset=0)]
    page2 = [f.origin for _, f in kb.search("alpha", scope="kb", top_k=3, offset=3)]
    assert len(page1) == 3 and len(page2) == 3
    assert not set(page1) & set(page2), f"pages must not overlap: {page1} vs {page2}"
    allsix = [f.origin for _, f in kb.search("alpha", scope="kb", top_k=6, offset=0)]
    assert page1 + page2 == allsix


def test_clip_prefers_line_boundaries():
    """Snippets are cut on line boundaries (whole lines), not mid-line at a char count."""
    text = "\n".join(f"line {i} " + "x" * 30 for i in range(20))
    out, clipped = ContextManager._clip(text, 200)
    assert clipped and "…（中间省略）…" in out
    head, tail = out.split("…（中间省略）…")
    assert head.lstrip().startswith("line 0 ")
    assert "line 19" in tail              # tail (conclusions) kept
    # every kept line must be WHOLE (source lines are 37 chars for i<10, 38 chars for i>=10);
    # a plain char-count cut would leave a partial line at both ends.
    assert [len(ln) for ln in head.strip().splitlines()] == [37, 37, 37], head
    assert [len(ln) for ln in tail.strip().splitlines()] == [38, 38], tail
    assert len(out) < len(text)
    one_line, clipped2 = ContextManager._clip("A" * 500, 100)
    assert clipped2 and "…（中间省略）…" in one_line   # single over-budget line -> char fallback


def test_rerank_dispatches_merged_bm25_through_subclass():
    """_rerank 必须经 `cls` 分派 `_merged_bm25`，不能写死基类名。

    harness 的 LayeredKB 是本类的**子类**；一旦写死基类名，子类覆写会被静默忽略，
    排序悄悄退化成基类 BM25 / raw 回退 —— 而排序错了不会有任何报错。
    """
    from contextmgr.layered import LayeredKB as BaseKB

    class _Override(BaseKB):
        @classmethod
        def _merged_bm25(cls, query, cands):
            return [0.0, 2.0, 1.0]        # 覆写生效时应得到 b > c > a

    kb = object.__new__(_Override)
    frags = [_frag("a", "zzz", 0.9), _frag("b", "zzz", 0.5), _frag("c", "zzz", 0.1)]
    out = kb._rerank([("proj", None, frags)], "q-not-present-in-docs")
    order = [f.origin for _n, _r, _l, f in out]
    # 基类路径（query 词不命中 → 回退按 raw score 归一）会得到 a > b > c，故顺序可判别
    assert order == ["b", "c", "a"], f"subclass override ignored: {order}"
