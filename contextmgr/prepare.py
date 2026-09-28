"""contextmgr —— `prepare_messages` 组装算法（自 `swe_agent/management.py` 下沉）。

职责边界（与 harness 的约定，勿越界）：

- 本模块只做**内容算法**：RAG 决策 → 预算裁剪/压缩 → 组装 → 出向清理。
- 凡属 **harness 配置**的东西一律由调用方**注入**，本模块不持有也不读取：
  - 预算/阈值数值 → 由 `model_context_length` 推导（调用方传窗口）；
  - 检索后端（LayeredKB）→ 注入 `rag_refs`；
  - 注入块组装（含模型摘取）→ 注入 `rag_block`；
  - 压缩后端（LFM/副驾）→ 注入 `compress_fn`；
  - 模型/工具的**调用次数**计数器（guard / fences）**不在此处** —— 那是 harness 的
    循环防护策略，归 `swe_agent/management.ContextManager.guard`。
- 本模块 `不 import swe_agent.config`（否则 import 期冻结值，monkeypatch / e2e env 失效）、
  不触网络、不触模型；所有注入项缺省 = 恒等行为，便于 model-free 单测。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .tokenize import estimate_messages, tokenize_words


def recall_from_truth(truth: List[Dict[str, Any]], query: str, budget: int,
                      ) -> List[Dict[str, Any]]:
    """model-free 召回：在**真相日志**（原文，未压缩）里按关键词重叠取回相关信息。

    这是「压缩过后仍能在后续需要时 recall 详细信息」的能力来源——工作集被压缩/裁剪，
    但真相日志永远保留原文，这里按 query 把被压缩掉的历史细节重新取回。

    打分：与 query 的 token 交集大小降序；同分按自身 token 数**升序**（小片段优先，
    更利于塞进预算）。不比较 dict 本身（避免 `TypeError: '<' not supported`）。
    """
    if not truth:
        return []
    q = tokenize_words(query)
    if not q:
        return []
    scored = []
    for m in truth:
        c = m.get("content", "") if isinstance(m.get("content"), str) else ""
        if not c:
            continue
        overlap = len(q & tokenize_words(c))
        if overlap:
            scored.append((overlap, estimate_messages([m]), m))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out, used = [], 0
    for _, _, m in scored:
        add = estimate_messages([m])
        if used + add > budget:
            continue
        out.append(m)
        used += add
    return out


def prepare_messages(
    buffer: List[Dict[str, Any]],
    *,
    model_context_length: int = 64000,
    user_input: str = "",
    recall_query: str = "",
    truth: Optional[List[Dict[str, Any]]] = None,
    rag_refs: Optional[Callable[[str, int], list]] = None,
    rag_block: Optional[Callable[[list, str, int], str]] = None,
    compress_fn: Optional[Callable[[list, int], list]] = None,
    recall_fn: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    sanitize: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """组装发往模型的一轮 messages（**只读** buffer，返回组装视图）。

    - `model_context_length`：目标模型上下文窗口（外部唯一需要告知的量）。
    - `user_input`：当前轮用户输入；命中 RAG 则拼「我找到了如下相关信息」块（由注入的
      `rag_block` 决定形态），未命中（无后端/无料）自动跳过（不强制）。
    - 决策顺序全部内部 auto：RAG → 预算裁剪（注入的 `compress_fn`，如两级压缩）→ 组装
      → （opt-in）从真相日志按需召回 → `sanitize` 出向清理。
    - 对 `buffer` 只读：不缩 buffer（guard 仍按全量历史算，计数归 harness）。

    🔴 不变量（勿简化）：`tool_call` 循环中**绝不**在末尾重复注入 user input + RAG。
    仅当「当前用户轮恰好是 buffer 末尾那条 user 消息」（首轮/新用户轮）或「`user_input`
    与 buffer 末条 user 不一致」（外部强制 query）时，才把 `(input + RAG)` 拼到末尾。
    详见 `tests/test_contextmgr_prepare_messages.py::test_prepare_messages_no_user_duplicate_after_tool_call`。
    """
    budget = int(model_context_length * 0.8)  # 留 20% 给输出
    # 分离 system 与 body（body 不再含 system，避免组装时 system 重复）
    body = list(buffer)
    system = None
    if body and body[0].get("role") == "system":
        system = body.pop(0)
    sys_tokens = estimate_messages([system]) if system else 0
    rag_budget = int(budget * 0.4)
    hist_budget = max(0, budget - sys_tokens - rag_budget)

    current_text = user_input
    # 找到 buffer 末尾最后一条 user 消息的内容（用于判断当前用户轮是否就在末尾）
    last_user_content = ""
    for m in reversed(body):
        if m.get("role") == "user":
            c = m.get("content", "")
            last_user_content = c if isinstance(c, str) else ""
            break
    # 当前用户轮是否就是 buffer 末尾的那条 user 消息？
    # 是 → 正常首轮/新用户轮：从 body 剥掉，改由末尾以 (input + RAG) 重新拼回；
    # 否 → 处于 tool_call 循环中（末尾是 tool/assistant）：保留原位、不在末尾重复注入，
    #      否则上一轮 user input 会被重复塞入、且每轮都追加 RAG（用户反馈的 bug）。
    current_is_terminal = bool(
        current_text and body
        and body[-1].get("role") == "user"
        and body[-1].get("content") == current_text
    )
    if current_is_terminal:
        body.pop()

    # 是否需要在末尾拼「当前用户轮」：(input + RAG) 或裸 input
    inject_current = bool(
        current_text and (current_is_terminal or current_text != last_user_content))

    # ① RAG 决策（检索后端由调用方注入；未注入 = 不检索）
    rag_user_msg = None
    if inject_current and rag_refs is not None:
        refs = rag_refs(current_text, rag_budget)
        if refs:
            block = rag_block(refs, current_text, rag_budget) if rag_block else ""
            rag_user_msg = {
                "role": "user",
                "content": current_text + "\n\n" + block,
            }

    # ② 压缩/滑动窗口/二级语义压缩决策（只读 body，body 不含 system）
    if compress_fn is not None and estimate_messages(body) > hist_budget:
        body = compress_fn(body, hist_budget)
        # 压缩器会对传入 body 重新插入 system 占位（内容为空，因 system 已单独抽出）；
        # 组装时 system 由本函数统一置顶，故剥离压缩结果里的 system 消息，避免重复 system。
        body = [m for m in body if m.get("role") != "system"]

    # ③ 组装：system + 压缩后多角色历史 + user(input+RAG)
    out: List[Dict[str, Any]] = []
    if system:
        out.append(system)
    out.extend(body)
    if rag_user_msg is not None:
        out.append(rag_user_msg)
    elif inject_current:
        out.append({"role": "user", "content": current_text})

    # ④ 按需召回（opt-in）：当 recall_query 给定且发往模型的工作集（body）比完整会话史
    #    （真相日志，原文未压缩）短时，说明历史被压缩/裁剪过，从真相日志取回与被压缩掉的
    #    历史相关的详细信息，作为末尾 system 块注入。默认关闭——关闭时行为与改动前完全
    #    一致（既有的 sliding 协议不变量测试不受影响）。
    truth_list = truth if truth is not None else []
    if recall_query and len(truth_list) > len(body):
        try:
            fn = recall_fn
            if fn is None:
                fn = lambda q, b: recall_from_truth(truth_list, q, b)  # noqa: E731
            recalled = fn(recall_query, rag_budget)
            if recalled:
                block = "\n\n".join(
                    f"[{m.get('role', '?')}] {m.get('content', '')}" for m in recalled)
                out.append({"role": "system",
                            "content": f"[历史补充·按需召回（原文，未压缩）]\n{block}"})
        except Exception:
            pass

    # 出向清理（如剥 reasoning_content / tool 轮 content 归零）——策略由调用方注入。
    if sanitize is not None:
        out = [sanitize(m) for m in out]
    return out
