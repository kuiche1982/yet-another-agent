"""contextmgr Code L1/L2/descend 补全测试。

覆盖三项新增能力（commit 1）：
- L1.key_points / L3.label 含文件:行号
- L2 模块级调用图：def→def 同模块边、def→外部孤立节点
- retrieve(include_levels=True) 返回 (L3, L1, L0) 三元组
- split_code_fragments 携带 lineno/end_lineno
- persist 落盘 + load 圆周转 lineno / calls
"""

from __future__ import annotations

import os
import tempfile

from contextmgr import (
    ContextManager, BM25Embedder, KeywordDistiller, Source,
    split_code_fragments,
)
from contextmgr import persist as P


# ---------- split_code_fragments ----------

def test_split_code_fragments_carry_line_numbers():
    code = "def alpha():\n    pass\n\nclass Beta:\n    def gamma(self):\n        pass\n\ndef delta():\n    return 1\n"
    frags = split_code_fragments(code, "mod.py", fid_prefix="mod.py")
    by_name = {f.fid.rsplit("#", 1)[-1]: f for f in frags}
    assert by_name["alpha"].lineno == 1 and by_name["alpha"].end_lineno == 2
    # class Beta 跨 4 行
    assert by_name["Beta"].lineno == 4 and by_name["Beta"].end_lineno == 6
    # 末位 def delta 在 8 行起
    assert by_name["delta"].lineno == 8


def test_split_code_fragments_whole_file_has_line_range():
    code = "# 仅注释与配置\nKEY = 1\n"   # 无顶层 def/class，走 #whole
    frags = split_code_fragments(code, "conf.py", fid_prefix="conf.py")
    assert len(frags) == 1 and frags[0].fid == "conf.py#whole"
    assert frags[0].lineno == 1 and frags[0].end_lineno == code.count("\n") + 1


# ---------- L1 / L2 ----------

def test_code_l1_key_points_include_file_line():
    code = "def foo(a, b):\n    '''do foo'''\n    return a + b\n\ndef bar():\n    return foo(1, 2)\n"
    cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder())  # 默认 CodeDistiller
    cm.ingest_code(code, name="m.py")
    # bar 是 L1（含 calls=[foo]）
    bar_l1 = cm.store._l1["m.py#bar"]
    assert any("@ m.py:" in kp for kp in bar_l1.key_points), bar_l1.key_points
    assert "foo" in bar_l1.calls, f"expected foo in calls, got {bar_l1.calls}"


def test_code_l2_graph_has_call_edges():
    code = "def foo():\n    return 1\n\ndef bar():\n    return foo()\n"
    cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder())
    cm.ingest_code(code, name="m.py")
    # bar 片段的 L2 应该画出 D0["def bar()"] -->|"calls"| C0["foo"]
    bar_l2 = cm.store._l2["m.py#bar"]
    assert "calls" in bar_l2.diagram
    assert "foo" in bar_l2.diagram


def test_module_callgraph_persists_def_to_def_edges():
    """persist.write_l2_callgraphs 聚合跨片段 def→def 边（同模块内）。"""
    code = "def foo():\n    return 1\n\ndef bar():\n    return foo()\n"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        fpath = os.path.join(src, "m.py")
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write(code)
        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder())
        P.rebuild(cm, src, cache, "keyword", set(), {".py"}, base=src)
        cg = open(os.path.join(cache, "L2_callgraphs.md"), encoding="utf-8").read()
        # 模块级图：含 def→def 同模块边（bar -->|"calls"| foo）
        assert "m.py" in cg
        assert "calls" in cg
        assert "foo" in cg and "bar" in cg


# ---------- retrieve(include_levels=True) ----------

def test_retrieve_default_returns_l0_only():
    cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder())
    cm.ingest_code("def foo():\n    return 1\n", name="m.py")
    hits = cm.retrieve("foo", sources={Source.CODE})
    assert hits and not isinstance(hits[0], tuple)


def test_retrieve_with_levels_returns_triple():
    cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder())
    cm.ingest_code("def foo():\n    return 1\n\ndef bar():\n    return foo()\n", name="m.py")
    hits = cm.retrieve("bar", sources={Source.CODE}, include_levels=True)
    assert hits, "expected at least one hit"
    triple = hits[0]
    assert len(triple) == 3
    l3, l1, frag = triple
    # L3 一行索引含「m.py:行号 + defs」
    assert "m.py" in l3.label and "defs" in l3.label
    # L1 含行号 key_point
    assert any("@ m.py:" in kp for kp in l1.key_points)
    # L0 是完整 def 源码（snippet 是可解析 Python）
    import ast
    ast.parse(frag.text)
    assert "def bar" in frag.text


# ---------- persist roundtrip ----------

def test_persist_roundtrip_preserves_lineno_and_calls():
    code = "def foo():\n    return 1\n\ndef bar():\n    return foo()\n"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        fpath = os.path.join(src, "m.py")
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write(code)
        cm1 = ContextManager(budget_tokens=4096, embedder=BM25Embedder())
        P.rebuild(cm1, src, cache, "keyword", set(), {".py"}, base=src)
        # 再起一个 cm2 从磁盘 load，验证行号 + calls 圆周转
        cm2 = ContextManager(budget_tokens=4096, embedder=BM25Embedder())
        P.load_store(cache)
        # 直接调 load_store 返回新 store；手动塞入并 inspect
        from contextmgr.store import FragmentStore
        store = P.load_store(cache)
        for f in store.all_fragments():
            if f.fid == "m.py#bar":
                assert f.lineno == 4
                assert f.end_lineno == 5
                l1 = store._l1[f.fid]
                assert "foo" in l1.calls