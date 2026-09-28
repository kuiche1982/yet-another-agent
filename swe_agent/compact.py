#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/compact.py —— 对话压缩（参考 claude-code autoCompact）

压缩摘要生成走标准 OpenAI 文本补全（models.chat_text_messages，走 glm/lmstudio
的 content 通道），不再依赖已废弃的 rapid-mlx plaintext 路径。
其余估算/阈值/触发逻辑保持不变。
"""

import re
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
from . import config as C
from . import models as M
from . import roles_config as RC
from swe_agent.log import logger


# 对话压缩阈值
MANUAL_COMPACT_REQUESTED: Dict[str, bool] = {"flag": False}


_CJK_RE = re.compile(r"[一-鿿]")


def estimate_tokens(messages) -> int:
    """粗略估算 messages 的 token 数（无需真实分词器）。"""
    total = 0
    for m in messages:
        c = m.get("content", "") or ""
        if isinstance(c, list):
            c = " ".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
        cjk = len(_CJK_RE.findall(c))
        other = len(c) - cjk
        total += int(cjk * 1.3 + other / 3.5) + 4
    return total


def get_auto_compact_threshold() -> int:
    return C.CONTEXT_WINDOW - C.AUTOCOMPACT_BUFFER


COMPACT_SYSTEM_PROMPT = """你是一个对话压缩助手。你【只能输出纯文本总结，禁止调用任何工具】。

你会收到一段完整的开发对话记录（含用户需求、你的规划、代码文件内容、工具执行结果、测试输出等）。
请生成一份结构化总结，保留「继续完成该开发任务」所必需的全部技术细节，使得拿到这份总结的人无需回看原文也能继续工作。
"""


def get_compact_prompt(instructions: str = "") -> str:
    base = """请把上面的对话压缩成一份详细总结，重点关注用户的明确需求和你之前的动作。
这份总结要完整保留继续开发所需的技术细节、代码模式与架构决策。
你的总结请包含以下小节：
1. 主要需求与意图 2. 关键技术概念 3. 文件与代码段 4. 错误与修复
5. 问题解决 6. 所有用户消息 7. 待办任务 8. 当前工作 9. 可选的下一步
输出格式：先 <analysis> 块，再 <summary> 块。
"""
    if instructions:
        base += f"\n额外压缩侧重指令（用户/agent 指定）：\n{instructions}\n"
    return base


COMPACT_SUMMARY_TEMPLATE = (
    "【以下是对话压缩后的上下文摘要，请基于它继续完成任务】\n"
    "================ 对话摘要开始 ================\n"
    "{summary}\n"
    "================ 对话摘要结束 ================\n"
)


def extract_summary(text: str) -> str:
    m = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    stripped = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return stripped.strip() or text.strip()


def _trim_for_summary(conv, budget):
    trimmed, total = [], 0
    for m in reversed(conv):
        t = estimate_tokens([m])
        if trimmed and total + t > budget:
            break
        trimmed.insert(0, m)
        total += t
    return trimmed


def _generate_summary(conv, instructions, model_override=None):
    budget = max(C.CONTEXT_WINDOW - 4000, 4000)
    conv = _trim_for_summary(conv, budget)
    msgs = [{"role": "system", "content": COMPACT_SYSTEM_PROMPT}]
    msgs += conv
    msgs.append({"role": "user", "content": get_compact_prompt(instructions)})
    # 压缩后端模型：显式收口到 RoleConfig.model_override（根治旧「副驾静默换模型」bug）。
    # 走统一 Agent（TEXT 模式）→ LFM 的 load/unload 由 Agent 自己的 pre_loop/post_loop 钩子
    # 自包含管理（压缩完即卸 LFM），绝不和工作模型（如 qwen）同时驻留显存。
    # 不再使用 SW.SidecarCompressSession 的「unload 主模型→load LFM→还原」交换逻辑——
    # 那会造成「先 load 工作模型、压缩时又卸又装」的反序与冗余（见 supervisor 重构时序说明）。
    target = model_override or C.SIDECAR_COMPRESS_MODEL
    agent = RC.make_agent("compact", loop=RC.single_loop(), model_override=target)
    out = agent.run(msgs)

    if not out or out.startswith("llm_error"):
        raise RuntimeError(out or "compaction model returned empty")
    return extract_summary(out)


def compact_conversation(messages, instructions: str = "", is_auto: bool = False,
                         model_override=None):
    system = None
    conv = list(messages)
    if conv and conv[0].get("role") == "system":
        system = conv[0]
        conv = conv[1:]
    if len(conv) < 2:
        return messages, {"skipped": True, "reason": "对话过短，无需压缩", "is_auto": is_auto}
    pre_tokens = estimate_tokens(messages)
    summary = _generate_summary(conv, instructions, model_override=model_override)
    summary_msg = {"role": "user",
                   "content": COMPACT_SUMMARY_TEMPLATE.format(summary=summary)}
    new_messages = ([system] if system else []) + [summary_msg]
    post_tokens = estimate_tokens(new_messages)
    if post_tokens >= pre_tokens:
        return messages, {"skipped": True, "reason": "摘要未比原文更短，跳过压缩",
                          "pre": pre_tokens, "post": post_tokens, "is_auto": is_auto}
    return new_messages, {"pre": pre_tokens, "post": post_tokens,
                          "is_auto": is_auto, "summary_len": len(summary)}


_auto_failures = 0


def _stdin_monitor() -> None:
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        if line.strip().lower() in ("/compact", "/c"):
            MANUAL_COMPACT_REQUESTED["flag"] = True
            logger.info('%s', '\n[压缩] 检测到 /compact，将在本轮推理前压缩对话……')


def start_stdin_monitor() -> None:
    # 无人值守模式（UNATTENDED_MODE=1）：不启动 stdin 监听线程，避免后台进程下
    # sys.stdin.readline() 因 TTY 残留/管道未关闭而阻塞。手动 /compact 在 batch 下无意义。
    if C.UNATTENDED_MODE:
        return
    t = threading.Thread(target=_stdin_monitor, daemon=True)
    t.start()


def maybe_auto_compact(messages, model_override=None):
    """若估算 tokens 超过阈值且未熔断，则自动压缩。model_override 强制指定压缩模型。"""
    global _auto_failures
    if not C.AUTO_COMPACT_ENABLED:
        return messages
    tokens = estimate_tokens(messages)
    threshold = get_auto_compact_threshold()
    if tokens < threshold:
        return messages
    if _auto_failures >= C.MAX_AUTO_COMPACT_FAILURES:
        return messages
    logger.info('%s', f'\n[压缩] 自动触发：估算 tokens≈{tokens} ≥ 阈值 {threshold}，开始压缩对话……')
    try:
        new_msgs, stats = compact_conversation(messages, is_auto=True,
                                               model_override=model_override)
        _auto_failures = 0
        logger.info('%s', f"[压缩] 自动完成：{stats['pre']} → {stats['post']} tokens（节省 {stats['pre'] - stats['post']}）")
        return new_msgs
    except Exception as e:
        _auto_failures += 1
        logger.info('%s', f'[压缩] 自动压缩失败：{e}（熔断计数 {_auto_failures}/{C.MAX_AUTO_COMPACT_FAILURES}）')
        return messages
