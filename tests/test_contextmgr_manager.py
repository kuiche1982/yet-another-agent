"""contextmgr 集成测试（model-free / tool-free）：ContextManager 全流程。

覆盖需求 #1-#7 的主链路：三源接入、L3 索引、RAG 回落、三种压缩、Code 参与。
不依赖 LLM / 向量模型 / 网络。
"""

from __future__ import annotations

from contextmgr import BM25Embedder, CompressionStrategy, ContextManager, Source

CODE = (
    "def connect_redis(host):\n"
    "    '''建立 Redis 连接'''\n"
    "    return host\n\n"
    "class Cache:\n"
    "    def get(self, key):\n"
    "        return None\n\n"
    "def migrate_db():\n"
    "    pass\n"
)


def _cm() -> ContextManager:
    cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                         strategy=CompressionStrategy.COMPRESS)
    cm.set_system("你是编码助手。")
    cm.ingest_library(
        "Redis 用作缓存层，配置 cache.ttl=300，需处理缓存击穿与雪崩。\n无关段：周末去爬山。",
        doc_id="redis",
    )
    cm.ingest_library(
        "数据库迁移流程：备份、停写、迁移、校验。\n无关段：买咖啡。",
        doc_id="db",
    )
    cm.ingest_code(CODE, name="cache.py")
    cm.append("user", "我想用 Redis 做缓存，怎么处理击穿？")
    cm.append("assistant", "可以用互斥锁或逻辑过期。")
    cm.append("user", "那数据库迁移要注意什么？")
    return cm


def test_three_sources_ingested():
    cm = _cm()
    s = cm.stats()
    assert s["by_source"]["library"] == 2
    assert s["by_source"]["code"] == 3          # connect_redis / Cache / migrate_db
    assert s["by_source"]["session"] == 3       # user/assistant/user 三条非 system 消息各建 L0


def test_code_l3_participates_in_retrieval():
    cm = _cm()
    frags = cm.retrieve("connect_redis 函数", sources={Source.CODE}, top_k=5)
    assert frags
    assert any("connect_redis" in f.text for f in frags)


def test_build_context_includes_rag_and_recent_session():
    cm = _cm()
    msgs = cm.build_context("Redis 缓存击穿怎么处理", budget_tokens=2000)
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    # RAG 参考块应包含资料库 Redis 内容
    rag = [m for m in msgs if "资料库/代码参考" in m.get("content", "")]
    assert rag, "应注入 RAG 参考块"
    assert "Redis" in rag[0]["content"]
    # 最近会话 user 消息应保留
    assert any("击穿" in m["content"] for m in msgs if m["role"] == "user")


def test_build_context_under_budget():
    cm = _cm()
    msgs = cm.build_context("Redis", budget_tokens=600)
    total = sum(len(m["content"]) // 4 for m in msgs)
    assert total <= 600 + 50  # 启发式容差


def test_aggressive_strategy_drops_history():
    cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                         strategy=CompressionStrategy.AGGRESSIVE)
    cm.set_system("sys")
    cm.ingest_library("Redis 缓存击穿用互斥锁。", doc_id="redis")
    cm.append("user", "历史问题1")
    cm.append("assistant", "回答1")
    cm.append("user", "Redis 击穿怎么处理")
    msgs = cm.build_context("Redis 击穿", budget_tokens=2000)
    # 极简：system + RAG + 最后一条 user，历史被砍
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert "Redis 击穿怎么处理" in user_msgs[0]["content"]
    assert any("Redis" in m["content"] for m in msgs if "参考" in m["content"])


def test_compress_if_needed_is_inplace():
    cm = _cm()
    before = len(cm.buffer)
    cm.compress_if_needed(budget_tokens=50)
    assert len(cm.buffer) <= before
    # system 仍在
    assert cm.buffer[0]["role"] == "system"
