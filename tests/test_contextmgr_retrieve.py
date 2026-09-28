"""contextmgr 单元测试（model-free / tool-free）：检索 descend + 预算 + 优先级。

验证「L3 索引 → 沿指针精确回落 L0」与「会话高优先级」「预算贪婪填充」。
"""

from __future__ import annotations

from contextmgr import BM25Embedder, ContextManager, Fragment, Source


def _build() -> ContextManager:
    cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder())
    cm.ingest_library(
        "Redis 用作缓存层，配置 cache.ttl=300，需处理缓存击穿。\n第二段无关内容：今天天气不错。",
        doc_id="redis",
    )
    cm.ingest_library(
        "PostgreSQL 数据库迁移步骤：备份、停写、迁移、校验。\n另一个无关段：午饭吃啥。",
        doc_id="db",
    )
    return cm


def test_retrieve_descends_to_l0_exact_text():
    cm = _build()
    frags = cm.retrieve("Redis 缓存击穿", top_k=3)
    assert frags, "应召回 Redis 相关片段"
    # 第一名应是 redis 文档片段，且是原文（L0 Truth）
    assert frags[0].source is Source.LIBRARY
    assert "Redis" in frags[0].text and "击穿" in frags[0].text


def test_retrieve_budget_truncates():
    cm = _build()
    tiny = cm.retrieve("Redis 缓存", budget_tokens=20, top_k=5)
    # 单片段 token 远 > 20，贪婪填充应截断为空或极少
    total = sum(f.tokens for f in tiny)
    assert total <= 20 + max((f.tokens for f in tiny), default=0)


def test_session_high_priority_ranks_first():
    cm = _build()
    # 会话里也说 Redis（命中同一关键词），应排在资料库之前
    cm.append("user", "我们服务用 Redis 做缓存，注意击穿")
    frags = cm.retrieve("Redis 缓存", top_k=5)
    assert frags
    assert frags[0].source is Source.SESSION
    assert "Redis" in frags[0].text


def test_retrieve_filters_by_source():
    cm = _build()
    frags = cm.retrieve("配置", sources={Source.CODE}, top_k=5)
    # 没有代码源，应返回空
    assert frags == []
