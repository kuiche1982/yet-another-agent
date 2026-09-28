#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""contextmgr —— 注入层（provenance / 切片 / 注入块组装）

本模块是「检索出来的片段 → 注入给模型的文本」这一段的**唯一实现**，原住在
`swe_agent/management.py`（`_ref_location` / `ContextManager._clip` / `_rag_block` /
`local_search` 的格式化部分），现下沉到内容管理层。理由见 `docs/contextmgr_design.md`：
RAG 参考的组装归属 contextmgr（`build_context`），harness 只该管协议合规与 buffer。

三条硬边界（下沉时必须守住，否则等于把雷搬进新家）：
1. **不 import `swe_agent.config`**：所有参数由调用方在调用期注入（`InjectOptions`）。
   若在此处 `from swe_agent.config import RAG_INJECT_TOP_N`，值会被冻结进本模块命名空间
   → monkeypatch / e2e 改 env 全部失效。
2. **不触网络**：模型摘取走**可注入后端**（`summarize=Callable`），默认 `None` → 完全跳过。
   「模型是否已加载」的判断留在 harness（由注入的 callable 内部完成），本模块不认识任何模型。
   这守住 design doc §8：contextmgr 单测保持 model-free / tool-free。
3. **不持有模块级可变状态**：本模块只有纯函数 + 一个 frozen dataclass，天然会话隔离安全。

出处行号（provenance）由代码写入、不经模型 —— 模型即便出错也不会污染出处。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

from .tokenize import estimate_tokens
from .types import Source

logger = logging.getLogger(__name__)

ELLIPSIS = "\n…（中间省略）…\n"
READ_FILE = "read_file"


@dataclass(frozen=True)
class InjectOptions:
    """注入层参数（由 harness 在调用期从 config 取值填入，勿在模块内捕获 config）。"""

    max_chars: int = 2400              # 单条正文上限（超出走摘取/截断）
    allow_summary: bool = False        # 是否允许模型摘取（需同时注入 summarize 后端）
    summary_min_tokens: int = 0        # 低于此 token 数不做摘取（不值得一次往返）
    summary_max_calls: int = 0         # 单次注入的摘取调用上限
    summary_model: str = ""            # 摘取用模型名（仅透传给后端）
    snippet_chars: int = 600           # local_search 工具输出的单条片段上限
    top_n: int = 5                     # 被动注入精选条数


# ======================================================================
# provenance：片段 → (出处, read_file 指针)
# ======================================================================
def ref_location(kb, f) -> tuple[str, str]:
    """解析片段的真实出处与 read_file 指针（注入块与 local_search 共用，保证口径一致）。

    - Code 源：origin 即相对 WORKSPACE 的路径，可直接 read_file；
    - Library 源：origin 仅 "kb:<stem>"，须经 KB 的 stem→真实路径索引还原；
    - 有行号 → 给 `path:start-end` 与 offset/limit 指针；无行号 → 退化为按路径读全文。
    """
    origin = getattr(f, "origin", "") or ""
    lineno = getattr(f, "lineno", 0) or 0
    end = getattr(f, "end_lineno", lineno) or lineno
    path = origin
    if getattr(f, "source", None) is not Source.CODE:
        real = ""
        resolver = getattr(kb, "_lib_realpath", None) if kb is not None else None
        if callable(resolver):
            try:
                real = resolver(origin) or ""
            except Exception:
                real = ""
        path = real or origin
    if lineno:
        return (f"{path}:{lineno}-{end}",
                f"{READ_FILE} {path} offset={lineno} limit={max(1, end - lineno + 1)}")
    return path, f"{READ_FILE} {path}"


# ======================================================================
# 切片：抽取式截断（确定性、不调模型）
# ======================================================================
def clip(text: str, max_chars: int) -> tuple[str, bool]:
    """抽取式截断（保真、确定性、不调模型）：保头 + 保尾，返回 (文本, 是否截断)。

    优先按**行边界**截断（保留完整行，语义不残缺）；首行本身就超预算时（压缩过的长行）
    回退字符切分。头部权重 0.6 / 尾部 0.4 —— 文档结论与失败摘要常在末尾。
    """
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head_budget = max(1, int(max_chars * 0.6))
    tail_budget = max(1, max_chars - head_budget)
    lines = text.splitlines()
    if len(lines) > 1:
        head_lines, used = [], 0
        for ln in lines:
            if used + len(ln) + 1 > head_budget:
                break
            head_lines.append(ln)
            used += len(ln) + 1
        if head_lines:                      # 首行能整行放下才走行切分
            tail_lines, used = [], 0
            for ln in reversed(lines[len(head_lines):]):
                if used + len(ln) + 1 > tail_budget:
                    break
                tail_lines.append(ln)
                used += len(ln) + 1
            head_txt = "\n".join(head_lines).rstrip()
            tail_txt = "\n".join(reversed(tail_lines)).lstrip()
            return head_txt + ELLIPSIS + tail_txt, True
    head = max(1, int(max_chars * 0.6))
    tail = max(1, max_chars - head)
    return text[:head].rstrip() + ELLIPSIS + text[-tail:].lstrip(), True


# ======================================================================
# 注入块组装（被动注入：正文 + 出处行号）
# ======================================================================
def rag_block(refs: Sequence, query: str, opts: InjectOptions,
              kb=None, summarize: Optional[Callable[[str, str, str], str]] = None,
              summarize_ready: Optional[Callable[[str], bool]] = None) -> str:
    """把检索结果打包成注入块：每条带「出处文件:行号」+ 正文（必要时摘取/截断）。

    - 相关性：上游已跨层全局归一化排序（高分在前），此处只做展示。
    - 正文：默认直接注入 L0 原文片段（agent 无需再 read_file）；单条超 max_chars 时，
      若有 `summarize` 后端且参数允许，先做「针对问题的摘取式总结」，失败再抽取式截断。
    - **出处行号恒由代码写入** → 永不丢失、不会被模型编造。
    - `summarize(query, text, model) -> str`：由 harness 注入（内含「模型就绪」判断）；
      未注入 = 完全不调用模型（model-free）。
    - `summarize_ready(model) -> bool`：可选前置门控（如「本地模型是否已加载」）。
      在**计入调用次数之前**判定 —— 与旧实现逐字对齐（未就绪时不消耗 max_calls 配额）。
    """
    max_chars = max(200, int(opts.max_chars))
    min_tok = max(0, int(opts.summary_min_tokens))
    max_calls = max(0, int(opts.summary_max_calls))
    calls = 0

    lines = [f"我找到了如下相关信息（本地知识库/代码，按相关性排序，共 {len(refs)} 条；"
             f"正文为原文片段，出处含文件与行号，可直接引用，无需再逐个读文件）："]
    for i, f in enumerate(refs, 1):
        loc, pointer = ref_location(kb, f)
        score = getattr(f, "score", 0.0) or 0.0
        raw = (getattr(f, "text", "") or "").strip()
        body, clipped = clip(raw, max_chars)
        if (clipped and opts.allow_summary and summarize is not None
                and calls < max_calls and estimate_tokens(raw) >= min_tok
                and (summarize_ready is None or summarize_ready(opts.summary_model))):
            calls += 1
            try:
                summ = summarize(query, raw, opts.summary_model) or ""
            except Exception as e:      # 摘取是旁支，失败必须回退抽取式截断
                logger.debug('RAG 摘取失败（回退抽取式截断）：%s', e)
                summ = ""
            # 忠实性闸门：非空、且确实比原文更短（防模型把内容写长或编造）才采用
            if summ and len(summ) < len(raw) and estimate_tokens(summ) < estimate_tokens(raw):
                body, clipped = summ, False
        if clipped:
            lines.append(f"### [{i}] {loc} · 相关性 {score:.2f}\n{body}\n"
                         f"（已截断，读全文：{pointer}）")
        else:
            lines.append(f"### [{i}] {loc} · 相关性 {score:.2f}\n{body}")
    return "\n\n".join(lines)


# ======================================================================
# local_search 工具输出格式（主动检索：分层标注 + 相关性 + 指针 + 片段）
# ======================================================================
def format_local_search(results: Iterable, *, kb=None, scope: str = "all",
                        offset: int = 0, snippet_chars: int = 600) -> str:
    """把 [(layer, fragment)] 渲染成工具返回文本（与被动注入块同口径的出处/指针）。"""
    results = list(results)
    snip = max(80, int(snippet_chars))
    out = [f"本地检索结果（scope={scope}，按相关性降序，第 {offset + 1}-{offset + len(results)} 条）："]
    for idx, (layer, f) in enumerate(results, offset + 1):
        loc, pointer = ref_location(kb, f)
        score = getattr(f, "score", 0.0) or 0.0
        body = clip((getattr(f, "text", "") or "").strip(), snip)[0]
        out.append(f"{idx}. [{layer}] {loc} · 相关性 {score:.2f}\n   → {pointer}\n   {body}")
    return "\n".join(out)


def select_refs(refs: Optional[Iterable], top_n: int) -> list:
    """按全局相关性精选前 top_n 条（注入正文模式的前提，打破「N 条指针 → N 次 read_file」的 1:1 耦合）。"""
    return list(refs or [])[:max(0, int(top_n))]
