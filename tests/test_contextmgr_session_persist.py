"""model-free 单测：Session 会话持久化（save_session / load_session）。

验证「火车票缺口」闭合：进程退出前 save_session，重启后 load_session 还原 buffer + 片段，
会话连续性不丢。不依赖 LLM / 网络 / tiktoken。
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
        budget_tokens=1024,
        embedder=BM25Embedder(),
        strategy=CompressionStrategy.COMPRESS,
        library_distiller=KeywordDistiller(),
    )


def test_save_load_session_roundtrip_buffer_and_fragments():
    with tempfile.TemporaryDirectory() as d:
        cm = _cm()
        cm.set_system("sys prompt")
        cm.append("user", "我订了火车票去北京")
        cm.append("assistant", "好的，已记录行程")

        persist.save_session(cm, d)

        cm2 = _cm()
        persist.load_session(cm2, d)

        # buffer 原样恢复（含 system + 两条对话）
        assert cm2.buffer == cm.buffer
        assert cm2.buffer[0]["role"] == "system"
        assert "火车票" in cm2.buffer[1]["content"]
        # Session 片段数一致
        assert cm2.store.fragment_count(Source.SESSION) == 2


def test_session_fragment_retrievable_after_load():
    with tempfile.TemporaryDirectory() as d:
        cm = _cm()
        cm.append("user", "我订了火车票去北京，下周一出发")
        persist.save_session(cm, d)

        cm2 = _cm()
        persist.load_session(cm2, d)
        hits = cm2.retrieve("火车票去北京", sources={Source.SESSION})
        assert hits, "重启后应按 Session 线命中火车票片段"
        assert "火车票" in hits[0].text


def test_load_session_no_crash_when_empty():
    with tempfile.TemporaryDirectory() as d:
        cm = _cm()
        persist.load_session(cm, d)  # 目录空/不存在均不崩
        assert cm.buffer == []
        assert cm.store.fragment_count(Source.SESSION) == 0


def test_save_session_writes_buffer_file_and_fragment_files():
    with tempfile.TemporaryDirectory() as d:
        cm = _cm()
        cm.append("user", "hello session")
        persist.save_session(cm, d)
        assert os.path.isfile(os.path.join(d, "session_buffer.json"))
        # 至少落盘 1 个 Session 片段 .md
        md_files = [n for n in os.listdir(d)
                    if n.endswith(".md") and n not in ("L2_overview.md", "L3_index.md")]
        assert md_files, "Session 片段应落盘为 .md"
        # 片段 frontmatter source=session
        txt = open(os.path.join(d, md_files[0]), encoding="utf-8").read()
        assert "source: session" in txt
