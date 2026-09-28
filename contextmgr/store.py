"""contextmgr —— L0 片段存储 + 派生 L1/L2/L3 缓存。

Store 只认 Fragment（L0 = Truth）。L1/L2/L3 都是可重建的派生缓存。
内存结构；持久化（SQLite / 文件）是可选扩展，不在本包范围内。
"""

from __future__ import annotations

from .types import Fragment, L1Structured, L2Visual, L3Index, Source


class FragmentStore:
    def __init__(self) -> None:
        self._frags: dict[str, Fragment] = {}
        self._l1: dict[str, L1Structured] = {}
        self._l2: dict[str, L2Visual] = {}
        self._l3: dict[str, L3Index] = {}

    # ---- 写入 ----
    def add_fragment(self, frag: Fragment) -> None:
        if frag.tokens <= 0:
            from .tokenize import estimate_tokens
            frag.tokens = estimate_tokens(frag.text)
        self._frags[frag.fid] = frag

    def add_derived(self, l1: L1Structured, l2: L2Visual, l3: L3Index) -> None:
        self._l1[l1.fid] = l1
        self._l2[l2.fid] = l2
        self._l3[l3.fid] = l3

    # ---- 读取 ----
    def get_fragment(self, fid: str) -> Fragment | None:
        return self._frags.get(fid)

    def get_l3(self, fid: str) -> L3Index | None:
        return self._l3.get(fid)

    def fragment_count(self, source: Source | None = None) -> int:
        if source is None:
            return len(self._frags)
        return sum(1 for f in self._frags.values() if f.source == source)

    def all_l3(self) -> list[L3Index]:
        return list(self._l3.values())

    def all_fragments(self) -> list[Fragment]:
        return list(self._frags.values())

    def l0_from_pointer(self, pointer: tuple[str, int, int]) -> Fragment | None:
        """沿 L3 回指针取精确 L0 片段（descend 终点）。"""
        fid = pointer[0]
        return self._frags.get(fid)

    def clear(self) -> None:
        self._frags.clear()
        self._l1.clear()
        self._l2.clear()
        self._l3.clear()
