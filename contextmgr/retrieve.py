"""contextmgr —— 检索（L3 排序 → 沿指针 descend 到精确 L0 片段，预算感知填充）。

检索语义严格遵循设计：
- 只在 L3/L1（短索引）上匹配 query，命中后**沿 l0_pointer 回落 L0 原文片段**；
- 默认只返回 L0（Truth），契合「L0 is Truth」原则；
- `include_levels=True` 时返回 `(L3, L1, L0)` 三元组，供逐级 descend 导航（agent 拿到 L1 函数签名 + 行号再做是否 descend L0 的决策）；
- 贪婪按分数+优先级填充，直到 token 预算耗尽；会话/资料库/代码统一参与。
"""

from __future__ import annotations

from .index import Index
from .store import FragmentStore
from .tokenize import estimate_tokens
from .types import Fragment, L1Structured, L3Index, Source


class Retriever:
    def __init__(self, store: FragmentStore, index: Index):
        self.store = store
        self.index = index

    def retrieve(self, query: str, budget_tokens: int, top_k: int = 8,
                 sources: set[Source] | None = None,
                 include_levels: bool = False) -> list:
        """返回命中的 L0 片段（精确原文），按预算贪婪填充。

        include_levels=True 时返回 list[tuple[L3Index, L1Structured | None, Fragment]]，
        供 agent 做「L3 → L1 函数签名+行号 → L0 完整代码」逐级导航。
        """
        ranked = self.index.search(query, top_k=top_k)
        out: list = []
        used = 0
        for l3 in ranked:
            if sources and l3.source not in sources:
                continue
            frag = self.store.l0_from_pointer(l3.l0_pointer)
            if frag is None:
                continue
            # 把排序分数回写到 L0 片段：下游（LayeredKB 跨层重排 / 注入层）需要相关性信号。
            # score 是检索期瞬态字段，不落盘、不参与等价比较；缺分时保持默认 0.0。
            try:
                frag.score = float(getattr(l3, "score", 0.0) or 0.0)
            except (TypeError, ValueError):
                frag.score = 0.0
            cost = frag.tokens or estimate_tokens(frag.text)
            if used + cost > budget_tokens and out:
                break
            if include_levels:
                l1 = self.store._l1.get(l3.fid)
                out.append((l3, l1, frag))
            else:
                out.append(frag)
            used += cost
        return out
