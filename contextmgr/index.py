"""contextmgr —— L3 索引构建 + 检索排序。

Index 把 Distiller 产出的 L1/L2/L3 写入 Store，并提供 search：
- query 匹配 L3.label + L1.key_points（短、高信噪）
- 用 Embedder 打分，返回按 (score 降, source 优先级降) 排序的 L3 列表
- Retriever 再沿 l0_pointer 精确 descend 到 L0
"""

from __future__ import annotations

from .distill import Distiller
from .embedder import BM25Embedder, Embedder
from .store import FragmentStore
from .types import Fragment, L3Index, Source

SESSION_BOOST = 5.0  # 会话源匹配即优先（需求 #2：Session 高优先级 L0）


class Index:
    def __init__(self, store: FragmentStore, embedder: Embedder | None = None):
        self.store = store
        self.embedder = embedder or BM25Embedder()

    def index_fragment(self, frag: Fragment, distiller: Distiller) -> L3Index:
        l1, l2, l3 = distiller.distill(frag)
        self.store.add_fragment(frag)
        self.store.add_derived(l1, l2, l3)
        return l3

    def _doc_for(self, l3: L3Index) -> str:
        """检索文档 = L3 标签 + L1 关键词（短文本，避免整块 L0 向量误区）。"""
        l1 = self.store._l1.get(l3.fid)
        kp = " ".join(l1.key_points) if l1 else ""
        return f"{l3.label} {kp}"

    def search(self, query: str, top_k: int = 8) -> list[L3Index]:
        l3s = self.store.all_l3()
        if not l3s:
            return []
        docs = [self._doc_for(l3) for l3 in l3s]
        # BM25 需先 fit；若 embedder 非 BM25 则逐条 similarity
        if isinstance(self.embedder, BM25Embedder):
            self.embedder.fit(docs)
        scores = [self.embedder.similarity(query, d) for d in docs]
        # 仅召回真正命中的片段（real score > 0）；会话源在命中项内享高优先级 boost
        matched = [(l3, sc) for l3, sc in zip(l3s, scores) if sc > 0]
        matched.sort(
            key=lambda x: (
                x[1] + (SESSION_BOOST if x[0].source is Source.SESSION else 0.0),
                _src_pri(x[0].source),
                x[0].priority,
            ),
            reverse=True,
        )
        out: list[L3Index] = []
        for l3, sc in matched[:top_k]:
            l3.score = sc
            out.append(l3)
        return out


def _src_pri(source) -> int:
    from .types import SOURCE_PRIORITY
    return SOURCE_PRIORITY.get(source, 0)
