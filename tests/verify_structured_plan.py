#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""
verify_structured_plan.py —— 定向验证（不跑全量 harness）

验证点：用「同一份 json_schema」通过 response_format 约束 GLM(planner 大脑)
和 LM Studio 的 qwen(executor 兜底) 的「plan」输出，是否都能稳定产出
harness 期望的扁平格式：{"tasks": ["字符串描述", ...]}（tasks 是字符串列表，
不是嵌套对象）。

这是之前 qwen 崩溃的根因对照：之前走 tool calling，qwen 把 plan 写成
{tasks:[{step,description,actions:[]}]}（嵌套 dict）导致 _verify_task_deliverables
对 dict 跑正则而崩溃。结构化输出能否强制它变扁平，是本次要验证的。

用法：
    .venv/bin/python3.12 verify_structured_plan.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swe_agent import config as C  # 仅取 BASE_URL / TOKEN

# ---------- 一份共享的扁平 plan schema ----------
PLAN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "依次要执行的任务，每个是一句纯文本描述",
                }
            },
            "required": ["tasks"],
        },
    },
}

PROMPT = (
    "你是一个资深软件工程师的规划器。请为下面的开发任务制定一个分步计划。\n"
    "任务：用 Python 实现康威生命游戏（Conway's Game of Life）的命令行程序，"
    "包含网格初始化、演化规则、渲染输出，并配套单元测试。\n"
    "只输出计划，不要写代码，不要解释。"
)


def _shape_ok(tasks):
    """harness 期望：tasks 是「非嵌套的纯字符串列表」。"""
    if not isinstance(tasks, list):
        return False, f"type={type(tasks).__name__}"
    if not tasks:
        return False, "empty list"
    all_str = all(isinstance(t, str) for t in tasks)
    return all_str, ("flat_list[str]" if all_str else "NOT-flat: " + str([type(t).__name__ for t in tasks]))


def probe_glm():
    import requests
    if not C.GLM_API_TOKEN:
        return None, "no GLM_API_TOKEN"
    payload = {
        "model": C.GLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是规划器，必须按要求返回结构化 JSON。"},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "response_format": PLAN_SCHEMA,
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
        print(f"[GLM] HTTP {r.status_code} in {time.time()-t:.1f}s")
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        return content, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def probe_lmstudio_qwen():
    import requests
    payload = {
        "model": "qwen2.5.1-coder-7b-instruct",
        "messages": [
            {"role": "system", "content": "你是规划器，必须按要求返回结构化 JSON。"},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "response_format": PLAN_SCHEMA,
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
        print(f"[LMStudio-qwen] HTTP {r.status_code} in {time.time()-t:.1f}s")
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        return content, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main():
    results = {}
    for name, fn in (("GLM(planner)", probe_glm), ("LMStudio-qwen(executor)", probe_lmstudio_qwen)):
        print(f"\n========== {name} ==========")
        content, err = fn()
        if err:
            print(f"  ❌ {err}")
            results[name] = {"ok": False, "error": err}
            continue
        print(f"  raw content: {content[:400]}")
        try:
            obj = json.loads(content)
            tasks = obj.get("tasks")
        except Exception as e:
            print(f"  ⚠️ 不是合法 JSON: {e}")
            results[name] = {"ok": False, "error": f"json-parse: {e}", "raw": content}
            continue
        ok, info = _shape_ok(tasks)
        print(f"  tasks 形状: {info}")
        print(f"  解析到的任务数: {len(tasks) if isinstance(tasks, list) else 'N/A'}")
        if isinstance(tasks, list):
            for i, t in enumerate(tasks[:6], 1):
                print(f"    {i}. {t if isinstance(t, str) else type(t).__name__}")
        results[name] = {"ok": ok, "shape": info, "tasks": tasks}

    # ---------- 汇总 ----------
    print("\n================ 汇总 ================")
    all_ok = True
    for name, res in results.items():
        if res.get("ok"):
            print(f"  ✅ {name:28} 扁平格式 OK  ({res['shape']})")
        else:
            all_ok = False
            print(f"  ❌ {name:28} 失败: {res.get('error') or res.get('shape')}")
    print("\n结论:", "两份后端都能用结构化输出统一产出扁平 plan 格式 ✅"
          if all_ok else "存在不一致/失败，需进一步调 schema ❌")


if __name__ == "__main__":
    main()
