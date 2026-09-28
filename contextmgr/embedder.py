"""contextmgr —— 检索打分（Embedder 接口；默认 model-free BM25）。

embedding 误区的根因（见设计文档）：旧 `memory/` 对整块 700-token 碎片做 dense+sparse
向量，检索粒度被 chunk 锁死、聚合后丢失回 L0 的指针。

本模块修正：
- 只对 **L3 一行标签 + L1 关键词**（短、高信噪）做相似度；
- 命中后靠 L3.l0_pointer 精确 descend 到 L0 片段；
- Embedder 是可插拔接口：生产可换 BGE-M3（见 BGEEmbedder 占位），默认用 BM25 保证
  model-free 单测与零依赖。
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

_TOKEN_RE = re.compile(r"[一-鿿]|[a-z0-9_]+", re.I)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class Embedder(ABC):
    """相似度打分接口。rank/score 越大越相关。"""

    @abstractmethod
    def similarity(self, query: str, doc: str) -> float:
        ...


class BM25Embedder(Embedder):
    """model-free BM25。无外部依赖，确定可复现。

    适合对短文本（L3 标签、L1 关键词）打分；长文档检索建议仍走 L3 入口。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._dfs: dict[str, int] = {}
        self._docs: list[list[str]] = []
        self._avgdl: float = 0.0
        self._n: int = 0

    def fit(self, corpus: list[str]) -> "BM25Embedder":
        self._docs = [_tokenize(d) for d in corpus]
        self._n = len(self._docs)
        self._avgdl = (sum(len(d) for d in self._docs) / self._n) if self._n else 0.0
        self._dfs = {}
        for d in self._docs:
            for t in set(d):
                self._dfs[t] = self._dfs.get(t, 0) + 1
        return self

    def _idf(self, term: str) -> float:
        df = self._dfs.get(term, 0)
        # 平滑 IDF，避免未登录词爆负无穷
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def similarity(self, query: str, doc: str) -> float:
        if not self._docs:  # 未 fit：退化为词重叠
            qt = set(_tokenize(query))
            dt = set(_tokenize(doc))
            inter = qt & dt
            return float(len(inter)) / (1 + len(qt | dt) - len(inter)) if inter else 0.0
        q_tokens = _tokenize(query)
        d_tokens = _tokenize(doc)
        dl = len(d_tokens)
        f = {}
        for t in d_tokens:
            f[t] = f.get(t, 0) + 1
        score = 0.0
        for qt in q_tokens:
            if qt not in f:
                continue
            idf = self._idf(qt)
            denom = f[qt] + self.k1 * (1 - self.b + self.b * (dl / self._avgdl if self._avgdl else 1))
            score += idf * (f[qt] * (self.k1 + 1)) / denom
        return score


class BGEEmbedder(Embedder):
    """BGE-M3 可插拔后端（占位）。

    生产用法：加载 bge-m3，对 L3.label + L1.key_points 做 encode，dense+sparse 混合打分。
    关键修正：embed **短索引文本**而非整块 L0 碎片；命中后由 Retriever 沿 l0_pointer 取片段。
    未接入时调用抛 NotImplementedError，避免静默走错路径。
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        raise NotImplementedError(
            "BGEEmbedder 未接入：请实现 encode+hybrid_retrieve，"
            "且只对 L3.label/L1.key_points 做 embedding（不要对整个 L0 碎片做向量）。"
        )

    def similarity(self, query: str, doc: str) -> float:  # pragma: no cover
        raise NotImplementedError
