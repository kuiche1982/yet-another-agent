#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # -*- coding: utf-8 -*-
    """定向验证：GLM 在 tools + response_format=json_object 同时存在时，返回的是 tool_calls 还是 JSON？
       仅读探针，不改主流程。"""
    import json, os, sys, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from swe_agent import config as C
    import requests

    SYS = "你是规划器，必须只返回一个 JSON 对象，结构为 {\"tasks\": [\"任务1\",\"任务2\"]}，tasks 是字符串列表。"
    PROMPT = "为「用 Python 实现康威生命游戏命令行程序」制定分步计划。只输出计划。"

    # 复刻用户 curl：tools + response_format=json_object 共存
    TOOLS = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名称"}},
                "required": ["city"],
            },
        },
    }]

    payload = {
        "model": C.GLM_MODEL,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": PROMPT}],
        "temperature": 0.2, "max_tokens": 1024,
        "tools": TOOLS, "tool_choice": "auto",
        "response_format": {"type": "json_object"},
    }
    t = time.time()
    try:
        r = requests.post(f"{C.GLM_BASE_URL}/chat/completions", json=payload,
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {C.GLM_API_TOKEN}"}, timeout=120)
        j = r.json()
        msg = ((j.get("choices") or [{}])[0].get("message") or {})
        tc = msg.get("tool_calls")
        content = msg.get("content") or ""
        print(f"HTTP {r.status_code} in {time.time()-t:.1f}s")
        print("has tool_calls:", bool(tc))
        if tc:
            print("  tool_calls[0].function.name:", tc[0]["function"]["name"], "args=", tc[0]["function"]["arguments"][:120])
        print("content[:280]:", repr(content[:280]))
        # 是否仍是扁平 tasks？
        try:
            obj = json.loads(content)
            print("  -> 解析: keys =", list(obj.keys()), "| tasks 扁平 =",
                  isinstance(obj.get("tasks"), list) and all(isinstance(x, str) for x in obj["tasks"]))
        except Exception as e:
            print("  -> content 非 JSON:", e)
    except Exception as e:
        print(f"EXC after {time.time()-t:.1f}s: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()


def test_main():
    main()
