"""contextmgr —— 对话历史压缩（三种策略，全部 model-free、确定性）。

策略（对应需求 #6）：
- SLIDING_WINDOW：保留末尾窗口，丢弃最旧；system 永远保留在最前。
- COMPRESS：保留 system + 首部 + 尾部，丢弃中间（用一行占位说明压缩了多少轮）。
- AGGRESSIVE：极简极小最激进——丢弃整段历史，仅留 system + 最近一条用户消息
  （RAG 注入由 ContextManager 在外部拼装，本函数只负责「砍历史」）。

所有函数纯函数式：输入 messages，输出 messages，不触模型、不触工具。

📌 与 `swe_agent/management.py` 的 `ContextManager` 关系：
   harness 运行时 `prepare_messages` / `compress_if_needed` 最终都收口到本模块的
   `compress()`。本文件是「压缩真相源」，F1/F2/F3 的修复都落在这里。
"""

from __future__ import annotations

import enum
from typing import Callable, Optional

from .tokenize import estimate_messages


class CompressionStrategy(str, enum.Enum):
    SLIDING_WINDOW = "sliding_window"
    COMPRESS = "compress"
    AGGRESSIVE = "aggressive"


def _count(messages: list[dict]) -> int:
    return estimate_messages(messages)


def _group_by_user(messages: list[dict]) -> list[list[dict]]:
    """按 user 边界把 body 切成「轮」组：每组从一条 user 起到「下一条 user 前」为止
    （含期间的 assistant / tool / system），形如 [user, system?, tool, tool, user)。

    组是滑动窗口与压缩的最小**不可拆**单元 —— 保证 `assistant(tool_calls)` 与其
    `tool` 结果永远同组，绝不会被单条裁剪拆散（否则出现「有 tool 回执无 tool_calls
    声明」的协议断裂，导致下一轮 provider 报 400）。body 首条若非 user 则归入首个组。

    注意：F1 的「轮内子单元降级」不改变本函数的语义（其契约被单测锁定），
    子单元拆分在 `_split_turn_subunits` 里独立实现。
    """
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for m in messages:
        if m.get("role") == "user" and cur:
            groups.append(cur)
            cur = []
        cur.append(m)
    if cur:
        groups.append(cur)
    return groups


def _flatten(groups: list[list[dict]]) -> list[dict]:
    out: list[dict] = []
    for g in groups:
        out.extend(g)
    return out


def _split_turn_subunits(group: list[dict]) -> list[list[dict]]:
    """把一个「轮」组拆成可独立丢弃的**子单元**（F1 用）。

    拆分规则（保协议完整：assistant(tool_calls) 与其 tool 回执永不分离）：
    - 独立的 `user` 消息 = 一个子单元（会话问题，体积小，尽量保留）。
    - `assistant`(+其后的 `tool`/`system`) = 一个子单元（答案/工具结果，体积大，优先丢）。

    降级时从最旧端逐个丢「assistant 子单元」、保留 user 子单元，直到预算满足或只剩 user；
    这样即使最后一轮整体超预算，也至少保住「用户问了什么」，不会整段清空成只剩 system
    （agent 失忆）。user 子单元之间互不依赖，单独保留不违反协议。
    """
    subs: list[list[dict]] = []
    cur_user: Optional[dict] = None
    cur_asst: Optional[list[dict]] = None

    def _flush():
        nonlocal cur_user, cur_asst
        if cur_asst is not None:
            subs.append(cur_asst)
            cur_asst = None
        if cur_user is not None:
            subs.append([cur_user])
            cur_user = None

    for m in group:
        r = m.get("role")
        if r == "user":
            _flush()
            cur_user = m
        elif r == "assistant":
            if cur_asst is not None:
                subs.append(cur_asst)
            cur_asst = [m]
        elif r == "tool":
            if cur_asst is not None:
                cur_asst.append(m)
            else:
                # 孤立 tool（不应出现于规范输入）→ 自成子单元，随最旧端丢弃
                subs.append([m])
        else:  # system 等组内杂项 → 挂到当前 assistant 子单元；无则独立
            if cur_asst is not None:
                cur_asst.append(m)
            else:
                subs.append([m])
    _flush()
    return subs


def _hard_degrade(groups: list[list[dict]], budget_tokens: int,
                  system: str) -> list[dict]:
    """🔴 F1 最后手段（model-free 兜底）：整组丢弃 → 仅最旧剩余组轮内子单元降级。

    三步走，且只在「必要时」降级，避免误伤本可整组保留的 tool 协议：
    ① 整组从最旧端丢弃（保最新），至少留 1 组——解决「连最新一轮都装不下时前几轮被整组清掉」。
    ② 剩余整组若已装得下 → 原样返回（assistant(tool_calls)+tool 整组保留，协议不断裂）。
    ③ 仅当剩下那一组仍超预算 → 轮内子单元降级：从最旧端丢「assistant(+tool) 子单元」、
       保留 user 子单元；若 user 也超预算再丢最旧 user（至少留 1 个）。

    永不产生孤立 tool（整子单元同进退），且保证不出现「输出只剩 system」的失忆态
    （只要原文里存在过任何 user 消息，就至少保住一条）。
    """
    # ① 整组丢弃（注意重建列表，避免与外部 head_g/tail_g 别名串扰）
    groups = list(groups)
    while len(groups) > 1 and _count(_flatten(groups)) > budget_tokens:
        groups = groups[1:]

    # ② 剩余整组若装得下 → 直接原样返回（tool 协议完整保留，不降级）
    flat = _flatten(groups)
    if _count(flat) <= budget_tokens:
        return ([{"role": "system", "content": system}] if system else []) + flat

    # ③ 仅最旧剩余组（通常只剩 1 组）仍超预算 → 轮内子单元降级
    subs: list[list[dict]] = []
    for g in groups:
        subs.extend(_split_turn_subunits(g))

    # 先丢最旧的 assistant 子单元（保留所有 user 问题）
    i = 0
    while _count(_flatten(subs)) > budget_tokens and i < len(subs):
        if subs[i][0].get("role") == "assistant":
            subs = subs[:i] + subs[i + 1:]
            i = 0
            continue
        i += 1

    # 仍超预算（user 本身太大）→ 丢最旧 user 子单元，至少留 1 个
    while _count(_flatten(subs)) > budget_tokens and len(subs) > 1:
        subs = subs[1:]

    return ([{"role": "system", "content": system}] if system else []) + _flatten(subs)


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    if messages and messages[0].get("role") == "system":
        return messages[0].get("content", ""), messages[1:]
    return "", messages


def compress(messages: list[dict], budget_tokens: int,
             strategy: CompressionStrategy = CompressionStrategy.COMPRESS,
             window: int = 6,
             compress_backend: Optional[Callable] = None) -> list[dict]:
    system, body = _split_system(messages)

    if strategy is CompressionStrategy.AGGRESSIVE:
        # 仅留最近一条 user 消息（system 之外）
        last_user = next((m for m in reversed(body) if m.get("role") == "user"), None)
        out = []
        if system:
            out.append({"role": "system", "content": system})
        if last_user:
            out.append(last_user)
        return out

    if strategy is CompressionStrategy.SLIDING_WINDOW:
        # 滑动窗口：保留尾部窗口，但按「轮」(user→下一条 user 前) 整组对齐与整组删，
        # 绝不单条砍。先按 window 条消息定位尾部边界，再向前对齐到组起点（避免切在 turn 中间）。
        groups = _group_by_user(body)
        if window and window > 0 and len(body) > window:
            start = len(body) - window
            while start > 0 and body[start].get("role") != "user":
                start -= 1
            groups = _group_by_user(body[start:])
        # 预算裁剪：整组从最旧端丢弃（保最新一轮 tool 协议完整），绝不单条删。
        # 至少保留 1 组，避免「连最新一轮都装不下」时被清空成只剩 system（F1）。
        while len(groups) > 1 and _count(_flatten(groups)) > budget_tokens:
            groups.pop(0)
        kept = _flatten(groups)
        base = ([{"role": "system", "content": system}] if system else []) + kept
        # F1 兜底：若最新一组仍超预算，轮内子单元降级（丢 assistant 保 user）
        if _count(base) > budget_tokens:
            return _hard_degrade(groups, budget_tokens, system)
        return base

    # COMPRESS：保留首部组 + 尾部组，中间组折叠为占位；超预算则整组砍 tail/head（不拆 turn）
    if _count(body) <= budget_tokens:
        return ([{"role": "system", "content": system}] if system else []) + body
    # head_n/tail_n 为「组数」（非消息数）：window//6≈旧 head_n=2 条消息、window//3≈旧 tail_n=3 条，
    # 既贴近旧尺寸，又保证每组是完整 turn（user→assistant(tool_calls)→tool）。
    head_n, tail_n = max(1, window // 6), max(1, window // 3)
    groups = _group_by_user(body)
    n = len(groups)
    if head_n + tail_n >= n:
        # 无中间段可丢弃 → 整段保留（若仍超预算，靠下方整组预算裁剪收敛）
        head_g, tail_g = groups, []
    else:
        head_g = groups[:head_n]
        tail_g = groups[-tail_n:]
        # 防御：head 与 tail 重叠（极端小 n）→ 取整段
        if head_g and tail_g and head_g[-1] is tail_g[0]:
            head_g, tail_g = groups, []
    dropped_msgs = max(0, len(body) - sum(len(g) for g in head_g) - sum(len(g) for g in tail_g))
    sysmsg = {"role": "system", "content": system}

    def assemble(hg, tg, placeholder):
        out = ([sysmsg] if system else []) + _flatten(hg)
        if placeholder is not None:
            out.append(placeholder)
        out.extend(_flatten(tg))
        return out

    placeholder = None
    if dropped_msgs > 0:
        placeholder = {"role": "system",
                       "content": f"[… 中间 {dropped_msgs} 条消息已压缩，按 L3 索引按需 RAG 回落 …]"}
    cand = assemble(head_g, tail_g, placeholder)
    if _count(cand) <= budget_tokens:
        return cand
    # 候选（head 组 + 占位 + tail 组）仍超预算：
    # ① 配置了语义压缩后端 → 升级大上下文模型压缩（不丢 user、不砍 tool），直接 return；
    # ② 未配置后端（model-free）→ 轮内子单元降级兜底（F1），绝不返回「只剩 system」。
    if compress_backend is not None:
        try:
            return compress_backend([sysmsg] + _flatten(head_g) + _flatten(tail_g), None)
        except Exception:
            pass
    return _hard_degrade(groups, budget_tokens, system)


def compress_two_tier(messages: list[dict], budget_tokens: int,
                      compress_backend: Optional[Callable] = None) -> list[dict]:
    """两级（三级）压缩收口：model-free → 仍超则滑动窗口 → 仍超且有后端则语义压缩。

    从 `swe_agent/management.ContextManager._compress_two_tier` **原样下沉**——该逻辑
    是「预算收敛策略」而非 harness 配置，故归 contextmgr；`compress_backend` 由调用方
    注入（harness 侧注入 LFM/副驾压缩通道，纯 contextmgr 用法传 None 保持 model-free）。

    每级都从**原始 messages** 重新压（而非在上一级产物上再压），保证策略可回退、
    结果可复现。
    """
    out = compress(list(messages), budget_tokens, CompressionStrategy.COMPRESS,
                   compress_backend=compress_backend)
    if _count(out) > budget_tokens:
        out = compress(list(messages), budget_tokens, CompressionStrategy.SLIDING_WINDOW)
    if compress_backend is not None and _count(out) > budget_tokens:
        try:
            out = compress_backend(out, None)
        except Exception:
            pass
    return out
