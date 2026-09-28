"""contextmgr —— token 估算（model-free、确定性）。

默认用字符启发式（中英文混排约 4 字符/token），可选 tiktoken（若环境可用）。
单测不依赖任何外部 tokenizer，保证可复现。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

try:  # tiktoken 可选；缺省回退到确定性启发式
    import tiktoken  # type: ignore

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - 取决于环境
    _ENC = None

# 中英文混合的轻量分词模式：CJK 按字切，其余按标识符切。
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数（确定性）。"""
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:  # pragma: no cover
            pass
    # 启发式：CJK 按字计，其他按空白分词；取较保守估计
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    rest = len(text) - cjk
    return max(1, cjk + rest // 4)


def estimate_messages(messages: Iterable[Dict[str, Any]]) -> int:
    """估算一组 OpenAI 风格 messages 的总 token 数。

    口径与 `compress` / `prepare_messages` 的预算判定**必须一致**，故收口于此，
    避免各文件各写一份 `sum(estimate_tokens(m.get("content", "")) for m in ...)` 漂移。
    content 缺失或为 None → 计 0（tool-call assistant 消息即属此类）。
    """
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def tokenize_words(text: str) -> Set[str]:
    """轻量分词（中英文混合，确定性）：用于 model-free 的关键词重叠打分。

    CJK 逐字、其余取 `[a-zA-Z0-9_]+`，统一转小写后返回集合（调用方取交集大小）。
    """
    return set(_WORD_RE.findall((text or "").lower()))
