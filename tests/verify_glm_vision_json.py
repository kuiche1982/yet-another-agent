#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""
验证 glm-4.1v-thinking-flashx / glm-4.6v 是否支持 response_format=json_object，
并检验它们能否产出 Planner 契约（flat tasks）。

直接打 zhipu OpenAI 兼容接口，复用 config 里的 token/base_url，
捕获 HTTP 状态、原始 content、JSON 解析结果，并过一遍 _normalize_plan 模拟 harness 解析。
"""
import json
import sys

sys.path.insert(0, "~/kuiwork/workdir2/litertlm")

from swe_agent import config as C
from swe_agent.llm_glm import _extract_json_object
from swe_agent.roles import _normalize_plan

import requests

PLANNER_SYSTEM = """你是顶层软件架构 Planner。把开发任务转化为结构化执行契约。
只输出一个 JSON 对象，禁止解释文字、禁止 markdown 代码块。Schema：
{"summary":"一句话概述","modules":[{"path":"相对路径","purpose":"职责","public":[]}],
"interface":[],"forbidden":[],"tasks":[{"step":"有序步骤","deliverables":[],"hidden_tests":[]}]}"""


def probe(model, use_rf):
    user = "【开发任务】\n用 Python 写一个函数 count_neighbors(grid, r, c)，统计二维网格中某格周围 8 邻域的存活邻居数。\n\n【当前工作区文件清单】\n（空目录，从零开始）"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    if use_rf:
        payload["response_format"] = {"type": "json_object"}
    try:
        resp = requests.post(
            f"{C.GLM_BASE_URL}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {C.GLM_API_TOKEN}"},
            timeout=C.GLM_TIMEOUT,
        )
        status = resp.status_code
        if status != 200:
            return {"ok": False, "status": status, "error": resp.text[:600]}
        data = resp.json()
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        usage = data.get("usage") or {}
        return {"ok": True, "status": status, "content": content, "finish_reason": finish,
                "usage": usage, "raw_msg": msg}
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        return {"ok": False, "status": status, "error": str(e)[:600]}


def evaluate(model, use_rf):
    print("=" * 70)
    print(f"MODEL={model}  response_format={use_rf}")
    print("=" * 70)
    r = probe(model, use_rf)
    if not r["ok"]:
        print(f"  ✗ HTTP {r['status']}: {r['error']}")
        return
    content = r["content"]
    print(f"  HTTP 200 | finish_reason={r['finish_reason']} | usage={r['usage']}")
    print(f"  --- raw content (前 500 字) ---")
    print("  " + content[:500].replace("\n", "\n  "))
    # 提取 JSON
    obj = _extract_json_object(content)
    if obj is None:
        print("  ✗ 无法解析为 JSON 对象（response_format 未生效或输出被包裹/含杂文）")
        return
    print(f"  ✓ 解析为 JSON 对象，顶层键: {list(obj.keys())}")
    norm = _normalize_plan(obj)
    if norm is None:
        print("  ✗ 结构不合格（缺 tasks/modules）")
        return
    n_tasks = len(norm["tasks"])
    print(f"  ✓ 通过 _normalize_plan：{len(norm['modules'])} 模块 / {n_tasks} 任务")
    for i, t in enumerate(norm["tasks"][:6], 1):
        print(f"     {i}. {t['desc'][:60]}")
    print()


def main():
    models = ["glm-4.1v-thinking-flashx", "glm-4.6v"]
    for m in models:
        # 同时测「带 response_format」与「不带」对照
        evaluate(m, True)
        evaluate(m, False)
    print("完成。")


if __name__ == "__main__":
    main()
