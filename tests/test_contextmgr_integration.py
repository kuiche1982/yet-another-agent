"""model-free 三线连调（集成测试）：Library + Code + Session 在 save/load 周期后全通。

模拟 agent 运行一轮（建三线）-> 落盘 -> 新进程重启加载 -> 验证三线俱在且检索可跨线回落。
不依赖 LLM / 网络（KeywordDistiller），验证 contextmgr 架构自身的连接正确性。
"""

import os
import tempfile

from contextmgr import (
    BM25Embedder,
    CompressionStrategy,
    ContextManager,
    KeywordDistiller,
    Source,
)
from contextmgr import persist


def _cm() -> ContextManager:
    return ContextManager(
        budget_tokens=2048,
        embedder=BM25Embedder(),
        strategy=CompressionStrategy.COMPRESS,
        library_distiller=KeywordDistiller(),
    )


def test_three_lines_persist_load_roundtrip():
    lib_doc = "订火车票要去 12306，选车次后支付。这是外部知识库的说明文档。"
    code_src = "def book_train(to):\n    return f'book {to}'\n"

    with tempfile.TemporaryDirectory() as d:
        kb = os.path.join(d, "KB")
        os.makedirs(kb)
        # 真实文件落盘，rebuild 才能按 mtime 增量持久化 Code 线
        with open(os.path.join(kb, "train.md"), "w", encoding="utf-8") as fh:
            fh.write(lib_doc)
        with open(os.path.join(kb, "book.py"), "w", encoding="utf-8") as fh:
            fh.write(code_src)

        # ---------- 第一轮：agent 运行，建立三线 ----------
        cm1 = _cm()
        cm1.set_system("dev agent")
        cm1.ingest_library(lib_doc, doc_id="kb:train")          # Library 线
        cm1.ingest_code(code_src, name="book.py")               # Code 线
        cm1.append("user", "我订了火车票去北京，下周一出发")      # Session 线
        cm1.append("assistant", "好的，已记录你的行程")
        # 落盘（模拟重启前保存）
        cache = os.path.join(d, ".ctx")
        persist.save_session(cm1, cache)                        # 会话线落盘
        persist.rebuild(cm1, kb, cache, "keyword",
                        {".md"}, {".py"}, base=kb)              # lib/code 落盘 + 写 L2/L3 索引

        # ---------- 第二轮：新进程，重启加载 ----------
        cm2 = _cm()
        cm2.load_session(cache)                                 # 恢复会话线
        persist.rebuild(cm2, kb, cache, "keyword",
                        {".md"}, {".py"}, base=kb)              # 恢复 lib/code 线

        # 断言：三线片段都在
        assert cm2.store.fragment_count(Source.LIBRARY) >= 1
        assert cm2.store.fragment_count(Source.CODE) >= 1
        assert cm2.store.fragment_count(Source.SESSION) >= 2

        # 断言：会话连续性（buffer 含火车票）
        joined = " ".join(m.get("content", "") for m in cm2.buffer)
        assert "火车票" in joined

        # 断言：Session 线检索跨重启命中（火车票缺口闭合）
        sess_hits = cm2.retrieve("火车票去北京", sources={Source.SESSION})
        assert sess_hits and "火车票" in sess_hits[0].text

        # 断言：Library 线检索命中
        lib_hits = cm2.retrieve("怎么订火车票", sources={Source.LIBRARY})
        assert lib_hits and "12306" in lib_hits[0].text

        # 断言：Code 线检索命中（匹配 L1 key_points 里的 def 签名）
        code_hits = cm2.retrieve("book_train", sources={Source.CODE})
        assert code_hits and "book_train" in code_hits[0].text

        # 断言：build_context 组装含会话 + RAG（不超预算）
        msgs = cm2.build_context("火车票怎么订", budget_tokens=1024)
        text = " ".join(m.get("content", "") for m in msgs)
        assert "火车票" in text


def test_three_lines_stats_and_sources():
    """stats() 正确反映三源计数（连接 manager.stats 与 store）。"""
    cm = _cm()
    cm.set_system("dev agent")          # buffer: system + user（stats 断言依赖此条）
    cm.ingest_library("a b c d e", doc_id="x")
    cm.ingest_code("def f():\n    return 1\n", name="m.py")
    cm.append("user", "hi")
    s = cm.stats()
    assert s["by_source"]["library"] >= 1
    assert s["by_source"]["code"] >= 1
    assert s["by_source"]["session"] >= 1
    assert s["buffer_messages"] >= 2  # system + user
