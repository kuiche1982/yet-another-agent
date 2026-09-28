#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""聚焦验证 GLM 的 structured output：打印完整原始响应，并尝试多版 schema，
找到能让 GLM 稳定产出扁平 {"tasks":[string]} 的写法；最后用该 schema 回测 LM Studio 确认可统一。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swe_agent import config as C
import requests

PROMPT = (
    "你是一个资深软件工程师的规划器。请为下面的开发任务制定一个分步计划。\n"
    "任务：用 Python 实现康威生命游戏（Conway's Game of Life）的命令行程序，"
    "包含网格初始化、演化规则、渲染输出，并配套单元测试。\n"
    "只输出计划，不要写代码，不要解释。"
)
SYS = "你是规划器，必须按要求返回结构化 JSON。"

# 几版候选 schema
VARIANTS = {
    "A_strict_no_addprop": {  # 当前版（GLM 返回空）
        "type": "json_schema",
        "json_schema": {
            "name": "plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "tasks": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["tasks"],
            },
        },
    },
    "B_strict_addprop": {  # 加 additionalProperties:false
        "type": "json_schema",
        "json_schema": {
            "name": "plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "tasks": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["tasks"],
                "additionalProperties": False,
            },
        },
    },
    "C_no_strict_addprop": {  # 去掉 strict，仅 json_schema + addprop
        "type": "json_schema",
        "json_schema": {
            "name": "plan",
            "schema": {
                "type": "object",
                "properties": {
                    "tasks": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["tasks"],
                "additionalProperties": False,
            },
        },
    },
    "D_json_object": {  # 退化为 json_object（无 schema，靠 prompt 约束）
        "type": "json_object",
    },
}


def call_glm(rf, label):
    payload = {
        "model": C.GLM_MODEL,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "response_format": rf,
    }
    t = time.time()
    try:
        r = requests.post(
            f"{C.GLM_BASE_URL}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {C.GLM_API_TOKEN}"},
            timeout=120,
        )
        print(f"\n--- GLM 变体 {label}  HTTP {r.status_code} in {time.time()-t:.1f}s ---")
        j = r.json()
        # 完整原始响应（截断）
        raw = json.dumps(j, ensure_ascii=False)
        print("RAW:", raw[:600])
        msg = (j.get("choices") or [{}])[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        print("CONTENT:", repr(content[:300]))
        if content:
            try:
                obj = json.loads(content)
                tasks = obj.get("tasks")
                ok = isinstance(tasks, list) and all(isinstance(x, str) for x in tasks)
                print("  -> tasks 形状:", "flat_list[str]" if ok else str([type(x).__name__ for x in tasks]) if isinstance(tasks, list) else type(tasks).__name__)
            except Exception as e:
                print("  -> JSON 解析失败:", e)
    except Exception as e:
        print(f"\n--- GLM 变体 {label} EXC: {type(e).__name__}: {e}")


def call_lmstudio(rf, label):
    payload = {
        "model": "qwen2.5.1-coder-7b-instruct",
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "response_format": rf,
    }
    t = time.time()
    try:
        r = requests.post(
            f"{C.LMSTUDIO_BASE_URL}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {C.LMSTUDIO_API_KEY}"},
            timeout=120,
        )
        print(f"\n--- LMStudio 变体 {label}  HTTP {r.status_code} in {time.time()-t:.1f}s ---")
        j = r.json()
        content = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        print("CONTENT:", content[:300])
        try:
            obj = json.loads(content)
            tasks = obj.get("tasks")
            ok = isinstance(tasks, list) and all(isinstance(x, str) for x in tasks)
            print("  -> tasks 形状:", "flat_list[str]" if ok else "NOT-flat")
        except Exception as e:
            print("  -> JSON 解析失败:", e)
    except Exception as e:
        print(f"\n--- LMStudio 变体 {label} EXC: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("########## GLM 各 schema 变体 ##########")
    for label, rf in VARIANTS.items():
        call_glm(rf, label)
    # 选一个能用的（优先 B），回测 LMStudio 确认统一
    print("\n\n########## 用 B_strict_addprop 回测 LMStudio ##########")
    call_lmstudio(VARIANTS["B_strict_addprop"], "B_strict_addprop")
