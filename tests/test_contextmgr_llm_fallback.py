"""LLMDistiller 兜底测试 —— model-free（不连真模型，验证失败不炸链路）。

LLMDistiller 默认依赖真模型；本测试用不可达端点触发兜底，断言：
- 任何异常都被吞，回退 KeywordDistiller，产出合法 L1/L2/L3
- Code 源直接走 AST（不经过 LLM），与是否可达无关
"""

from __future__ import annotations

import pytest

from contextmgr.llm_backends import LLMDistiller
from contextmgr.store import FragmentStore
from contextmgr.types import Fragment, Source


def _unreachable() -> LLMDistiller:
    # 指向不存在的端点，强制 distill 抛连接错误 -> 兜底
    return LLMDistiller(model="x", base_url="http://127.0.0.1:9/v1", timeout=2.0)


def test_llm_distiller_falls_back_on_unreachable():
    d = _unreachable()
    f = Fragment(text="Redis 缓存击穿可用互斥锁解决。", source=Source.LIBRARY,
                 fid="lib:t", priority=0)
    l1, l2, l3 = d.distill(f)  # 不应抛
    assert isinstance(l1.summary, str) and l1.summary
    assert isinstance(l2.diagram, str)
    assert l3.fid == "lib:t"
    assert l3.l0_pointer[0] == "lib:t"


def test_llm_distiller_code_routes_to_ast_without_llm():
    d = _unreachable()
    code = "def f():\n    return 1\n"
    frags = __import__("contextmgr.distill", fromlist=["split_code_fragments"]).split_code_fragments(
        code, "m.py", fid_prefix="m.py"
    )
    l1, l2, l3 = d.distill(frags[0])
    # 代码源走 AST，merged_with_l0=True，不依赖 LLM
    assert l1.merged_with_l0 is True
    assert "f" in l2.diagram
    assert l3.source is Source.CODE


def test_llm_distiller_store_arg_ignored_for_signature_compat():
    # Distiller ABC 的 distill 只收 frag；LLMDistiller 也应只用 frag
    d = _unreachable()
    f = Fragment(text="abc", source=Source.SESSION, fid="s:1", priority=3)
    l1, l2, l3 = d.distill(f)
    assert l3.priority == 3
