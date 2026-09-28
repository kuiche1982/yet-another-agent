#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""GLM 收尾验证：用更接近官方文档的 json_schema 信封 + json_object 带 tasks 键，确认 GLM 能否产出扁平 plan。"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swe_agent import config as C
import requests

PROMPT = ("你是一个资深软件工程师的规划器。请为下面的开发任务制定一个分步计划。\n"
          "任务：用 Python 实现康威生命游戏命令行程序，含网格初始化、演化规则、渲染、单元测试。\n"
          "只输出计划，不要写代码，不要解释。")
SYS = "你是规划器，必须只返回一个 JSON 对象，结构为 {\"tasks\": [\"任务1\",\"任务2\",...]}，tasks 是字符串列表，不要嵌套对象。"

E_doc_exact = {  # 官方文档风格：json_schema + description，无 strict
    "type": "json_schema",
    "json_schema": {
        "name": "plan",
        "description": "分步开发计划",
        "schema": {
            "type": "object",
            "properties": {
                "tasks": {"type": "array", "items": {"type": "string"},
                          "description": "依次要执行的任务，纯文本描述"}
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
    },
}
F_json_object_tasks = {  # json_object + prompt 约束用 tasks 键
    "type": "json_object",
}


def call_glm(rf, label):
    payload = {"model": C.GLM_MODEL,
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user", "content": PROMPT}],
               "temperature": 0.2, "max_tokens": 1024, "response_format": rf}
    t = time.time()
    try:
        r = requests.post(f"{C.GLM_BASE_URL}/chat/completions", json=payload,
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {C.GLM_API_TOKEN}"}, timeout=120)
        j = r.json()
        content = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        print(f"\n--- GLM {label}  HTTP {r.status_code} in {time.time()-t:.1f}s ---")
        print("CONTENT[:280]:", repr(content[:280]))
        if content.strip().startswith("```"):
            print("  ⚠️ 返回了 markdown 围栏 -> response_format 未被采纳（GLM 退回普通文本）")
        try:
            obj = json.loads(content)
            tasks = obj.get("tasks")
            ok = isinstance(tasks, list) and all(isinstance(x, str) for x in tasks)
            print("  -> 解析: tasks 形状 =", "flat_list[str] ✅" if ok else f"非扁平/缺 tasks 键 (keys={list(obj.keys())})")
        except Exception as e:
            print("  -> JSON 解析失败:", e)
    except Exception as e:
        print(f"\n--- GLM {label} EXC: {type(e).__name__}: {e}")


if __name__ == "__main__":
    call_glm(E_doc_exact, "E_doc_exact_json_schema")
    call_glm(F_json_object_tasks, "F_json_object_tasks_key")
