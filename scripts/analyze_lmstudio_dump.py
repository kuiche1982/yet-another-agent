#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_lmstudio_dump.py —— 从 LMSTUDIO_DUMP 落盘的请求日志里，提取每次 qwen 调用的
真实输入，并【重点列出导致 None / 超时的 error 条目对应的完整 messages】。

这直接回答「qwen 返回 None 时对应的输入是什么」：把 error 条目的 messages 原样打出，
即可拿去对照 LM Studio server 日志（按 ts 时间对齐），定位到底是超时还是软空、卡在哪个上下文。

用法：
    python scripts/analyze_lmstudio_dump.py [path] [MAX_MSG_CHARS]
默认 path=logs/lmstudio_requests.jsonl，MAX_MSG_CHARS=1500（单条 message 截断长度）。
"""
import json
import os
import sys
from pathlib import Path
from swe_agent.log import logger

DEFAULT_PATH = str(Path(__file__).resolve().parent.parent / "logs" / "lmstudio_requests.jsonl")


def load(p: str):
    recs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def _show_content(content, max_chars: int):
    if isinstance(content, list):  # multimodal / 结构化 content
        parts = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
            else:
                parts.append(str(blk))
        content = "\n".join(parts)
    if not isinstance(content, str):
        return f"({type(content).__name__})"
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + f"...(truncated, total {len(content)} chars)"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    max_chars = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("MAX_MSG_CHARS", "1500"))
    if not Path(path).exists():
        logger.info('%s', f'(文件不存在：{path})')
        return
    recs = load(path)
    if not recs:
        logger.info('%s', f'(空：{path} 还没有记录，电池可能刚启动)')
        return

    req = [r for r in recs if r.get("tag") == "request"]
    err = [r for r in recs if r.get("tag") == "error"]
    logger.info('%s', f'总记录={len(recs)}  request={len(req)}  error(对应 None/超时)={len(err)}')
    sizes = sorted((r.get("total_chars", 0) for r in req), reverse=True)
    if sizes:
        logger.info('%s', f'request 上下文字符数：max={sizes[0]}  median={sizes[len(sizes) // 2]}  min={sizes[-1]}')

    if not err:
        logger.error('%s', '(尚无 error/None/超时 记录 —— 本次跑还没出现 qwen 返回 None)')
        return

    logger.info('%s', f'\n===== {len(err)} 条 error（None/超时）对应的完整输入 =====')
    for i, e in enumerate(err, 1):
        logger.info('%s', f"\n--- error #{i}  ts={e.get('ts')}  model={e.get('model')}  n_msgs={e.get('n_messages')}  chars={e.get('total_chars')} ---")
        logger.info('%s', f"  错误: {e.get('error')}")
        logger.info('%s', f"  tools: {e.get('tools')}")
        for m in e.get("messages", []):
            role = m.get("role")
            tc = m.get("tool_calls")
            if tc:
                names = " | ".join(t.get("function", {}).get("name", "?") for t in tc)
                logger.info('%s', f'  [{role}] (tool_calls={len(tc)}): {names}')
                continue
            logger.info('%s', f"  [{role}] {_show_content(m.get('content') or '', max_chars)}")


if __name__ == "__main__":
    main()
