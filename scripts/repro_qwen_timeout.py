#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repro_qwen_timeout.py —— 复现「qwen 间歇性返回 None」是否由【客户端超时但 server 仍在跑】引起。

做法：
  1. 复用 harness 的【真实 executor system】（config.CONTRACT_OVERRIDE_BANNER + SYSTEM_PROMPT）
     与【真实 executor tools】（registry.glm_tools("executor")），构造一个贴近 guess 任务的
     「大上下文 + 长生成」请求（max_tokens=4096 + 一段长 pytest 失败 dump 堆上下文）。
  2. 先用【短超时】（默认 25s）发一次：若 SDK 触发超时异常 → 客户端早于 server 断开。
  3. 若超时，再用【长超时】（默认 600s）发【同一请求】，测 server 实际完成耗时。
     - 若 server 实际耗时 >> 短超时 → 证实「客户端超时但 server 端还在跑」（用户假设成立）；
       此时把【客户端断连时刻】拿去对照 LM Studio server 日志里该请求的完成时刻即可闭环。
     - 若短超时内就返回 200（含空响应）→ 说明这次是「server 秒回空（软空响应）」，不是超时，
       与 03_guess.log 的历史证据一致，反证「超时」假设。
  4. 完整请求样本落盘 /tmp/repro_qwen_request.json，便于保存/复现。

不依赖后台电池进程；会短暂占用本地 qwen 做长生成（可能数十秒），对正在跑的电池仅有轻微延迟影响。
"""

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openai import OpenAI  # noqa: E402
from swe_agent.log import logger

BASE = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen2.5.1-coder-7b-instruct")
SHORT_TO = float(os.environ.get("REPRO_SHORT_TIMEOUT", "25"))
LONG_TO = float(os.environ.get("REPRO_LONG_TIMEOUT", "600"))


def probe_loaded():
    try:
        import urllib.request
        with urllib.request.urlopen(BASE + "/models", timeout=5) as r:
            d = json.loads(r.read().decode())
        return [m.get("id") for m in d.get("data", [])]
    except Exception as e:
        return f"(探测失败:{e})"


def build_messages():
    from swe_agent.config import SYSTEM_PROMPT, CONTRACT_OVERRIDE_BANNER  # noqa
    workspace = str(REPO / "agent_sandbox")
    contract = (
        "方案概述：在 agent_sandbox 中实现数字猜谜功能，包括 make_target 和 guess 函数，并提供 CLI 入口。\n"
        "文件白名单：\n  - src/guess.py：实现数字猜谜核心逻辑；接口：make_target(seed)；guess(target, x)；main()\n"
        "  - test_guess.py：pytest 测试；用 from src.guess import make_target, guess\n"
        "验收点：\n  - [1] 猜中返回 'hit'\n  - [2] target>x 时返回 'low'\n  - [3] target<x 时返回 'high'\n  - [4] CLI 可运行\n"
    )
    system = (
        CONTRACT_OVERRIDE_BANNER + "\n" + SYSTEM_PROMPT +
        f"\n\n# 工作环境\n当前工作目录（绝对路径）：{workspace}\n"
        "工作区初始状态：空目录（从零开始）。\n\n"
        "# 顶层 Planner 契约（必须遵守）\n" + contract
    )
    # 大上下文：模拟已完成几轮 + 一段长 pytest 失败 dump，把上下文堆到 ~10k 字符
    big_blob = (
        "=== pytest -q 输出 ===\n"
        + ("FAILED test_guess.py::test_hit - assert guess(50, 50) == 'hit'\n" * 80)
        + "\n\n当前 src/guess.py 内容（疑似占位/写反）：\n"
        + ("def make_target(seed):\n    return 42\n" * 40)
    )
    history = [
        {"role": "user", "content": "请实现 guess.py，先写 make_target 和 guess，再写测试。"},
        {"role": "tool", "content": "written src/guess.py (42 lines)"},
        {"role": "user", "content": "跑测试看看结果"},
        {"role": "tool", "content": big_blob},
        {"role": "user", "content": (
            "请修复测试失败，并补全 test_guess.py，确保全部通过。"
            "给出完整的 src/guess.py 与 test_guess.py 实现，并补充若干边界测试。")},
    ]
    return [{"role": "system", "content": system}] + history


def main():
    from swe_agent import registry as R  # noqa
    loaded = probe_loaded()
    logger.info('%s', f'[probe] 已加载模型：{loaded}')
    if isinstance(loaded, list) and MODEL not in loaded:
        logger.info('%s', f'[probe] ⚠️ {MODEL} 不在位，复现可能拿到 400 No models loaded（非超时），请先确保 qwen 在位。')

    messages = build_messages()
    tools = R.ToolRegistry.glm_tools("executor")
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
        "parallel_tool_calls": False,
        "tools": tools,
        "tool_choice": "auto",
    }
    Path("/tmp/repro_qwen_request.json").write_text(json.dumps(
        {"base": BASE, "payload": payload, "short_timeout": SHORT_TO,
         "note": "将该请求的发起时间/内容对照 LM Studio server 日志中对应请求的完成时间，可验证「客户端超时但 server 还在跑」"},
        ensure_ascii=False, indent=2), encoding="utf-8")

    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    logger.info('%s', f'[repro] model={MODEL}  messages={len(messages)}  tools={len(tools)}  context_chars≈{total_chars}  max_tokens=4096')
    logger.info('%s', f'[repro] 短超时={SHORT_TO}s 发起请求（若 server 完成晚于该值即触发客户端超时）…')

    c_short = OpenAI(base_url=BASE, api_key=KEY, timeout=SHORT_TO)
    t0 = time.time()
    timed_out = False
    short_dur = 0.0
    try:
        resp = c_short.chat.completions.create(**payload)
        short_dur = time.time() - t0
        msg = resp.choices[0].message
        n_tc = len(getattr(msg, "tool_calls", None) or [])
        logger.info('%s', f"[repro] 短超时内返回（未超时）：耗时 {short_dur:.1f}s，tool_calls={n_tc}，content_len={len(msg.content or '')}")
        logger.info('%s', f'[repro] → 这次没有复现超时；若返回为空（tool_calls=0 且 content 空）则是【软空响应】，与 03_guess 历史一致，非超时。')
    except Exception as e:
        timed_out = True
        short_dur = time.time() - t0
        disconnect_ts = time.strftime("%H:%M:%S", time.localtime(t0 + short_dur))
        logger.info('%s', f'[repro] 短超时触发客户端断开：{type(e).__name__}: {str(e)[:200]}')
        logger.info('%s', f'[repro] 客户端于 {disconnect_ts}（发起后 {short_dur:.1f}s）断开连接')

    if timed_out:
        logger.info('%s', f'[repro] 再用长超时 {LONG_TO}s 发【同一请求】，测 server 实际完成耗时…')
        c_long = OpenAI(base_url=BASE, api_key=KEY, timeout=LONG_TO)
        t1 = time.time()
        try:
            resp2 = c_long.chat.completions.create(**payload)
            server_dur = time.time() - t1
            msg2 = resp2.choices[0].message
            n_tc2 = len(getattr(msg2, "tool_calls", None) or [])
            logger.info('%s', f"[repro] server 实际耗时 {server_dur:.1f}s 才返回（tool_calls={n_tc2}，content_len={len(msg2.content or '')}）")
            logger.info('%s', f'[repro] 结论：客户端 {short_dur:.1f}s 断开 << server 需 {server_dur:.1f}s 完成 → 【客户端超时但 server 仍在跑】假设成立。')
            logger.info('%s', f'[repro] 对照：拿客户端断连时刻 {disconnect_ts} 去 LM Studio server 日志找该请求，其完成时刻应晚于 {disconnect_ts} 约 {server_dur - short_dur:.0f}s。')
        except Exception as e2:
            logger.info('%s', f'[repro] 长超时也失败：{type(e2).__name__}: {e2}（server 端可能也崩/卡死，非单纯超时）')

    logger.info('%s', f'[repro] 请求样本已落盘：/tmp/repro_qwen_request.json')


if __name__ == "__main__":
    main()
