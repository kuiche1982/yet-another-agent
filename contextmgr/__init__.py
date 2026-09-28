"""contextmgr —— 分级记忆 + ContextManager（model-free 核心）。

记忆物料分级（与 harness_design_map.md 的「认知三层」是不同轴，勿混）：
- L0 Fragment  = Truth of Source（原始片段；唯一真相）
- L1 L1Structured = 结构化蒸馏（代码源与 L0 合并）
- L2 L2Visual = mermaid 可视化（agent 自导航；渲染失败不丢信息）
- L3 L3Index  = 一行索引 + 回 L0 精确指针（检索入口）

三源：Library（文档） / Session（会话，高优先级） / Code（代码，AST 脑图）。
检索：query 匹配 L3/L1（短索引）→ 沿 l0_pointer 精确 descend 到 L0。
压缩：SlidingWindow / Compress / Aggressive 三策略，全部 model-free、确定性。
"""

from __future__ import annotations

from .compress import CompressionStrategy, compress, compress_two_tier
from .distill import CodeDistiller, Distiller, KeywordDistiller, split_code_fragments
from .embedder import BM25Embedder, BGEEmbedder, Embedder
from .index import Index
from .inject import InjectOptions, clip, format_local_search, rag_block, ref_location
from .manager import ContextManager
from .prepare import prepare_messages, recall_from_truth
from .retrieve import Retriever
from .store import FragmentStore
from .tokenize import estimate_messages, estimate_tokens, tokenize_words
from .types import Fragment, L1Structured, L2Visual, L3Index, Source

__all__ = [
    "Source", "Fragment", "L1Structured", "L2Visual", "L3Index",
    "ContextManager", "FragmentStore", "Index", "Retriever",
    "Distiller", "KeywordDistiller", "CodeDistiller", "split_code_fragments",
    "Embedder", "BM25Embedder", "BGEEmbedder",
    "CompressionStrategy", "compress", "compress_two_tier",
    "estimate_tokens", "estimate_messages", "tokenize_words",
    "InjectOptions", "ref_location", "clip", "rag_block", "format_local_search",
    "prepare_messages", "recall_from_truth",
]
