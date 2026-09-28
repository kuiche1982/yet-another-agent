"""contextmgr 单元测试（model-free / tool-free）：蒸馏层 L0→L1/L2/L3。

不依赖任何 LLM、向量模型或网络。确定性。
"""

from __future__ import annotations

import ast

from contextmgr import (
    CodeDistiller, Fragment, KeywordDistiller, L1Structured, L2Visual, L3Index, Source,
    split_code_fragments,
)


def _doc_frag() -> Fragment:
    return Fragment(
        fid="doc1#p0", source=Source.LIBRARY,
        text="Redis 用作缓存层。配置项 cache.ttl=300。\n必须处理缓存击穿问题。",
        origin="doc1", priority=2,
    )


def test_keyword_distiller_l0_pointer_preserved():
    frag = _doc_frag()
    l1, l2, l3 = KeywordDistiller().distill(frag)
    assert isinstance(l1, L1Structured) and isinstance(l2, L2Visual) and isinstance(l3, L3Index)
    # L3 必须带回 L0 精确指针（descend 可达原文）
    assert l3.l0_pointer == frag.l0_pointer()
    assert l3.source is Source.LIBRARY
    assert l3.label  # 一行索引非空


def test_keyword_distiller_l2_is_valid_mermaid():
    frag = _doc_frag()
    _, l2, _ = KeywordDistiller().distill(frag)
    assert l2.diagram.startswith("graph TD")
    # 红线：不能出现 `end` 作裸节点 id（mermaid subgraph 保留字陷阱）
    assert "end" not in [ln.strip().split("[")[0].strip() for ln in l2.diagram.splitlines() if "[" in ln]
    # 不能出现嵌套方括号（坏语法）
    assert "]]" not in l2.diagram


def test_keyword_distiller_extracts_heat_sentences():
    frag = _doc_frag()
    l1, _, _ = KeywordDistiller().distill(frag)
    joined = " ".join(l1.key_points)
    assert "缓存" in joined or "Redis" in joined
    assert any("击穿" in kp for kp in l1.key_points)  # 含高信号词被抽中


def test_code_distiller_merges_l0_l1():
    code = "def foo(a):\n    '''do foo'''\n    return a\n\nclass Bar:\n    def baz(self):\n        pass\n"
    frag = Fragment(fid="m.py#whole", source=Source.CODE, text=code, origin="m.py",
                    lineno=1, end_lineno=6)
    l1, l2, l3 = CodeDistiller().distill(frag)
    # 代码源：L0 与 L1 合并
    assert l1.merged_with_l0 is True
    assert "foo" in l1.entities and "Bar" in l1.entities
    # L2 脑图由 AST 直接生成，含结构与定义名
    assert l2.diagram.startswith("graph TD")
    assert "foo" in l2.diagram and "Bar" in l2.diagram
    # L3 一行索引：含文件 + 行号 + defs 数（行号信息供模型定位）
    assert "m.py" in l3.label and "2 defs" in l3.label and ":1" in l3.label
    # L1.key_points 含「签名 @ 文件:行号」格式
    assert any("@ m.py:1" in kp for kp in l1.key_points)


def test_split_code_fragments_per_definition():
    code = "def alpha():\n    pass\n\nclass Beta:\n    def gamma(self):\n        pass\n\ndef delta():\n    return 1\n"
    frags = split_code_fragments(code, "mod.py", fid_prefix="mod.py")
    names = {f.fid for f in frags}
    assert "mod.py#alpha" in names and "mod.py#Beta" in names and "mod.py#delta" in names
    for f in frags:
        assert f.source is Source.CODE
        assert f.offset >= 0 and f.length == len(f.text)
        # 片段本身是合法 Python（精确 descend 不丢结构）
        ast.parse(f.text)


def test_code_distiller_handles_syntax_error_gracefully():
    frag = Fragment(fid="bad#whole", source=Source.CODE, text="def ((( not valid")
    l1, l2, _ = CodeDistiller().distill(frag)
    # 退化为 keyword 蒸馏，不抛异常
    assert l2.diagram.startswith("graph TD")
    assert l1.merged_with_l0 is False
