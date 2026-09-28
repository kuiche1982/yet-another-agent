"""contextmgr —— ContextManager 编排层。

把 L0–L3 分级、三源（Library/Session/Code）、检索 descend、三种压缩策略
收口成一个可直接用的对话上下文管理器。model-free（默认 BM25 + Keyword/Code 蒸馏），
生产可换 BGE-M3 / LLM 蒸馏（见 embedder.py / distill.py 的接口占位）。

与 swe_agent/management.py 的 ContextManager 关系：
- 现有 harness 的 ContextManager 只做「长度感知对话 buffer 压缩」；
- 本类在其之上补齐「资料库/代码 RAG 回落 + L3 索引分级」；
- 接入缝：harness 可在 before_step 调 `cm.build_context(query, budget)` 拿组装好的 messages，
  或把本类作为 `RunState.cm` 的后端（详见 docs/contextmgr_dev.md）。
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

from .compress import CompressionStrategy, compress
from .distill import CodeDistiller, Distiller, KeywordDistiller, split_code_fragments
from .embedder import BM25Embedder, Embedder
from .index import Index
from .retrieve import Retriever
from .store import FragmentStore
from .tokenize import estimate_tokens
from .types import Fragment, Source

_LIB_PRI = 2
_SESS_PRI = 5  # 会话片段高优先级


class ContextManager:
    def __init__(self, budget_tokens: int = 4096,
                 strategy: CompressionStrategy = CompressionStrategy.COMPRESS,
                 embedder: Embedder | None = None,
                 library_distiller: Distiller | None = None,
                 code_distiller: Distiller | None = None,
                 symbol_provider: "SymbolProvider | None" = None):
        """symbol_provider：代码切分用的符号提供者，默认 AST（纯 stdlib）。

        传 `LspSymbolProvider()` 走 LSP `documentSymbol`（多语言 + 真实行号）。
        LSP 不可用时它自己会静默回落 —— 但那需要真实文件，见 `ingest_code(path=)`。
        """
        self.budget_tokens = budget_tokens
        self.strategy = strategy
        self.embedder = embedder or BM25Embedder()
        self.store = FragmentStore()
        self.index = Index(self.store, self.embedder)
        self.lib_distiller = library_distiller or KeywordDistiller()
        self.code_distiller = code_distiller or CodeDistiller()
        self.symbol_provider = symbol_provider
        self.buffer: list[dict] = []   # 实时会话（OpenAI 风格 messages）
        self._seq = itertools.count(1)
        # 真相日志（L0 truth）：永远记原文，供 recall / 「会话历史不丢失」。
        self._truth: list[dict] = []
        # 压缩结果缓存：buffer 未变且预算相同则直接复用，避免「下次请求又重新压缩」。
        self._buffer_version: int = 0
        self._cached_compressed: list[dict] | None = None
        self._cached_budget: int = -1

    # ---------- 写入：三源 L0 ----------
    def append(self, role: str, content: str) -> "ContextManager":
        """当前会话消息 → 写入 buffer，并建一条高优先级 Session L0 片段（参与检索）。"""
        self.buffer.append({"role": role, "content": content})
        self._truth.append({"role": role, "content": content})
        self._buffer_version += 1
        self._cached_compressed = None  # buffer 变了 → 缓存失效
        if role == "system":
            return self  # system 不单独建片段，buffer 自带
        fid = f"session#{next(self._seq)}"
        frag = Fragment(fid=fid, source=Source.SESSION, text=content,
                        priority=_SESS_PRI, tokens=estimate_tokens(content))
        self.index.index_fragment(frag, self.lib_distiller)
        return self

    def add(self, msg: dict) -> "ContextManager":
        return self.append(msg.get("role", "user"), msg.get("content", ""))

    def set_system(self, content: str) -> "ContextManager":
        if self.buffer and self.buffer[0].get("role") == "system":
            self.buffer[0]["content"] = content
        else:
            self.buffer.insert(0, {"role": "system", "content": content})
        return self

    def ingest_library(self, text: str, doc_id: str = "doc") -> int:
        """文档导入资料库（L0: Source=Library），自动 L1/L2/L3。按空行切块。

        同时记录每块在原文中的行号区间（lineno/end_lineno），使注入层能保留
        「出处文件:行号」——与 Code 源对齐，便于 agent 精确回溯原文位置。
        """
        chunks = [c for c in text.split("\n\n") if c.strip()]
        n = 0
        cursor = 0
        for i, c in enumerate(chunks):
            chunk = c.strip()
            # 在原文档中顺序定位该块（cursor 单调递增，避免重复块相互覆盖）
            pos = text.find(chunk, cursor)
            if pos < 0:
                pos = cursor
            lineno = text.count("\n", 0, pos) + 1
            end_lineno = lineno + chunk.count("\n")
            cursor = pos + len(chunk)
            fid = f"{doc_id}#p{i}"
            frag = Fragment(fid=fid, source=Source.LIBRARY, text=chunk,
                            origin=doc_id, priority=_LIB_PRI, tokens=estimate_tokens(chunk),
                            lineno=lineno, end_lineno=end_lineno)
            self.index.index_fragment(frag, self.lib_distiller)
            n += 1
        return n

    def ingest_code(self, text: str, name: str = "module") -> int:
        """代码导入（L0: Source=Code），按符号切每定义一个片段，Code.L3 脑图参与检索。

        若构造时传入 `symbol_provider`（如 LspSymbolProvider）则用它切分
        （多语言 + 真实行号）；否则默认 ASTSymbolProvider（纯 stdlib，仅 Python）。
        两种 provider 切不出符号时都退化为整文件一个片段。

        注：内存态 `text` 无真实文件路径，LSP provider 需要 didOpen 的 uri，
        故走 LSP 时请用 `ingest_code_file(path=)`。
        """
        frags = split_code_fragments(text, name, fid_prefix=name,
                                     provider=self.symbol_provider, path=None)
        for f in frags:
            self.index.index_fragment(f, self.code_distiller)
        return len(frags)

    def ingest_code_file(self, path: str, name: str | None = None) -> int:
        """从磁盘读一个真实代码文件并切分（LSP provider 需要真实 path 才能 didOpen）。

        path 经 `symbol_provider.symbols(..., path=)` 传给 LSP，从而能解析跨模块 /
        self.method() 调用与真实行号。无 LSP 或切不出符号时自动回落 AST / 整文件片段。
        """
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        name = name or os.path.basename(path)
        frags = split_code_fragments(text, name, fid_prefix=name,
                                     provider=self.symbol_provider, path=path)
        for f in frags:
            self.index.index_fragment(f, self.code_distiller)
        return len(frags)

    # ---------- 检索 ----------
    def retrieve(self, query: str, budget_tokens: int | None = None, top_k: int = 8,
                 sources: set[Source] | None = None,
                 include_levels: bool = False) -> list:
        """descend 检索。

        include_levels=False（默认）：返回 L0 片段列表（仅 Truth）。
        include_levels=True：返回 [(L3, L1, L0)] 三元组，供「L3 标签 → L1 函数签名+行号 → L0 完整代码」
        三段式导航（详见 retrieve.py docstring）。
        """
        budget = budget_tokens if budget_tokens is not None else self.budget_tokens // 2
        return Retriever(self.store, self.index).retrieve(
            query, budget, top_k=top_k, sources=sources, include_levels=include_levels,
        )

    # ---------- 组装下一轮上下文 ----------
    def build_context(self, query: str, budget_tokens: int | None = None) -> list[dict]:
        """按 token 预算自动 RAG + 剪裁，返回组装好的 messages（供下一轮对话）。"""
        budget = budget_tokens or self.budget_tokens
        system_msg = next((m for m in self.buffer if m.get("role") == "system"), None)
        sys_tokens = estimate_tokens(system_msg["content"]) if system_msg else 0

        # RAG：资料库 + 代码参考（会话本身在 buffer 里，不重复拉）
        refs = self.retrieve(query, budget_tokens=max(0, budget - sys_tokens - 256),
                             sources={Source.LIBRARY, Source.CODE})
        rag_tokens = sum(f.tokens for f in refs)
        rag_block = []
        if refs:
            rag_text = "\n---\n".join(f"[{f.origin or f.source.value}] {f.text}" for f in refs)
            rag_block = [{"role": "system",
                          "content": f"[资料库/代码参考 · 沿 L3 索引回落 L0]\n{rag_text}"}]

        if self.strategy is CompressionStrategy.AGGRESSIVE:
            last_user = next((m for m in reversed(self.buffer) if m.get("role") == "user"), None)
            out = []
            if system_msg:
                out.append(system_msg)
            out.extend(rag_block)
            if last_user:
                out.append(last_user)
            return out

        remaining = budget - sys_tokens - rag_tokens
        # 命中缓存（buffer 未变且预算相同）→ 直接复用，避免「下次请求又重新压缩」。
        if (self._cached_compressed is not None
                and self._cached_version == self._buffer_version
                and self._cached_budget == max(0, remaining)):
            compressed = self._cached_compressed
        else:
            compressed = compress(self.buffer, max(0, remaining), self.strategy)
            self._cached_compressed = compressed
            self._cached_budget = max(0, remaining)
            self._cached_version = self._buffer_version
        body = [m for m in compressed if m.get("role") != "system"]
        out = []
        if system_msg:
            out.append(system_msg)
        out.extend(rag_block)
        out.extend(body)
        return out

    def compress_if_needed(self, budget_tokens: int | None = None) -> "ContextManager":
        """原地压缩会话 buffer（harness before_step 中间件可调用）。缓存压缩结果。"""
        budget = budget_tokens or self.budget_tokens
        self.buffer = compress(self.buffer, budget, self.strategy)
        # 压缩后 buffer 即「压缩过的 messages」：存好，下次请求直接复用，不再重新压缩。
        self._cached_compressed = self.buffer
        self._cached_budget = budget
        self._cached_version = self._buffer_version
        return self

    # ---------- 召回：从完整会话史（_truth，原文未压缩）按需取回详细信息 ----------
    def recall(self, query: str, budget_tokens: int | None = None) -> list[dict]:
        """model-free 召回：在 _truth（完整会话史，原文未压缩）里按关键词重叠取回相关信息。

        这是「压缩过后仍能在后续需要时 recall 详细信息」的能力来源——buffer 被压缩/裁剪，
        但 _truth 永远保留原文，这里按 query 把被压缩掉的历史细节重新取回。

        算法已下沉 `contextmgr.prepare.recall_from_truth`（与 harness 侧同一份实现，
        避免两处各写一遍打分逻辑而漂移）。
        """
        if not self._truth:
            return []
        budget = budget_tokens or (self.budget_tokens // 2)
        from .prepare import recall_from_truth
        return recall_from_truth(self._truth, query, budget)

    # ---------- 会话持久化（关闭「火车票缺口」） ----------
    def save_session(self, cache_dir: str) -> "ContextManager":
        """落盘会话：buffer + Session 片段。进程退出前调一次。"""
        from .persist import save_session as _save
        _save(self, cache_dir)
        return self

    def load_session(self, cache_dir: str) -> "ContextManager":
        """加载会话：恢复 buffer + Session 片段（L3 由 L1 重建）。重启后调一次。"""
        from .persist import load_session as _load
        _load(self, cache_dir)
        return self

    # ---------- 观测 ----------
    def stats(self) -> dict:
        return {
            "fragments": self.store.fragment_count(),
            "by_source": {s.value: self.store.fragment_count(s) for s in Source},
            "buffer_messages": len(self.buffer),
            "buffer_tokens": sum(estimate_tokens(m.get("content", "")) for m in self.buffer),
        }
