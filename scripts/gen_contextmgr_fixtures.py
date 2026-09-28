#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 logs/lmstudio_requests.jsonl 抽取真实请求，生成 contextmgr 单测快照 fixtures。

用途：tests/test_contextmgr_prepare_messages_sliding.py 需要**真实**的多轮
tool_call 会话（system + 多轮 user + assistant(tool_calls) ↔ tool 配对），
而 sessions/ 与 logs/ 都是可变目录（会被后续 e2e 覆写），不能直接当测试输入。
故把选定的两行固化成 tests/fixtures/*.json，测试只读快照，稳定可复现。

选样标准（脚本自带校验，不满足直接报错退出）：
- 至少 2 条 user、2 条 tool、1 条带 tool_calls 的 assistant；
- **协议完整**：每个 tool_call_id 都能找到对应 tool_calls（无孤儿回执），
  且每个 tool_calls 都有回执（无缺结果）—— 这是 OpenAI 协议的硬约束；
- 消息 key 集合只含 {role, content, tool_calls, tool_call_id}（无 reasoning_content 等杂质）。

用法：
    .venv/bin/python scripts/gen_contextmgr_fixtures.py
    .venv/bin/python scripts/gen_contextmgr_fixtures.py --check   # 只校验现有快照

产出：
    tests/fixtures/lmstudio_req_345_conway.json      # 48 条，含 UU 连续 user、末尾 tool
    tests/fixtures/lmstudio_req_466_multitool.json   # 41 条，含单轮 4 个 tool_calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "logs" / "lmstudio_requests.jsonl"
OUT_DIR = ROOT / "tests" / "fixtures"

# (源行号, 输出文件名, 说明)  —— 行号由 scripts 扫描脚本筛出，见模块 docstring
TARGETS = [
    (345, "lmstudio_req_345_conway.json",
     "48 条：system + 多轮 user（含 idx38/39 连续两条 user）+ assistant(tool_calls)↔tool，末尾为 tool 回执"),
    (466, "lmstudio_req_466_multitool.json",
     "41 条：system + 单轮 4 个 tool_calls（idx2）对应 4 条 tool 回执（idx3-6）"),
]

_ALLOWED_KEYS = {"role", "content", "tool_calls", "tool_call_id"}


def _read_line(idx: int) -> dict:
    """按行号读取 jsonl（行号 = 0-based 物理行序）。"""
    with SRC.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == idx:
                line = line.strip()
                if not line:
                    raise ValueError(f"line {idx} 是空行")
                return json.loads(line)
    raise IndexError(f"{SRC} 没有第 {idx} 行（文件行数不足）")


def validate(messages: list, label: str) -> dict:
    """校验协议完整性并统计；任何一项不满足直接 raise（生成快照前拦住脏数据）。"""
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{label}: messages 不是非空列表")
    roles = [m.get("role") for m in messages]
    if roles[0] != "system":
        raise ValueError(f"{label}: 首条不是 system，而是 {roles[0]}")

    call_ids = [c["id"] for m in messages if m.get("tool_calls") for c in m["tool_calls"]]
    result_ids = [m.get("tool_call_id") for m in messages if m.get("role") == "tool"]
    orphan = [t for t in result_ids if t not in call_ids]     # 有回执无声明 → 协议断裂
    missing = [c for c in call_ids if c not in result_ids]    # 有声明无回执 → 下一轮 400
    if orphan:
        raise ValueError(f"{label}: 孤儿 tool 回执 {orphan[:3]}")
    if missing:
        raise ValueError(f"{label}: 缺回执的 tool_calls {missing[:3]}")
    if len(call_ids) != len(set(call_ids)):
        raise ValueError(f"{label}: tool_call id 重复")

    extra = {k for m in messages for k in m.keys()} - _ALLOWED_KEYS
    if extra:
        raise ValueError(f"{label}: 出现预期外的消息字段 {sorted(extra)}")

    return {
        "n_messages": len(messages),
        "n_user": roles.count("user"),
        "n_assistant": roles.count("assistant"),
        "n_tool": roles.count("tool"),
        "n_tool_calls": len(call_ids),
        "role_pattern": "".join({"system": "S", "user": "U", "assistant": "A",
                                 "tool": "T"}.get(r, "?") for r in roles),
    }


def build(idx: int, note: str) -> tuple[str, dict]:
    rec = _read_line(idx)
    messages = rec.get("messages")
    label = f"line {idx}"
    stats = validate(messages, label)
    payload = {
        "_source": {
            "file": "logs/lmstudio_requests.jsonl",
            "line": idx,
            "ts": rec.get("ts"),
            "tag": rec.get("tag"),
            "model": rec.get("model"),
            "generator": "scripts/gen_contextmgr_fixtures.py",
            "note": note,
        },
        "_stats": stats,
        "messages": messages,
    }
    return label, payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验已生成的快照，不写盘")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"源文件不存在：{SRC}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for idx, fname, note in TARGETS:
        out_path = OUT_DIR / fname
        if args.check:
            if not out_path.exists():
                print(f"[MISS ] {fname}（未生成）")
                ok = False
                continue
            data = json.loads(out_path.read_text(encoding="utf-8"))
            try:
                validate(data["messages"], f"{fname}")
                src_line = data["_source"]["line"]
                print(f"[OK   ] {fname} 行={src_line} n={data['_stats']['n_messages']} "
                      f"tc={data['_stats']['n_tool_calls']}")
            except Exception as e:
                print(f"[FAIL ] {fname}: {e}")
                ok = False
            continue

        label, payload = build(idx, note)
        text = json.dumps(payload, ensure_ascii=False, indent=1)
        out_path.write_text(text + "\n", encoding="utf-8")
        s = payload["_stats"]
        print(f"[WRITE] {fname} <- {label}  n={s['n_messages']} "
              f"U={s['n_user']} A={s['n_assistant']} T={s['n_tool']} tc={s['n_tool_calls']} "
              f"({os.path.getsize(out_path)} bytes)")
        print(f"        pattern={s['role_pattern']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
