#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # -*- coding: utf-8 -*-
    """定位 glm-4.1v-thinking-flashx 400 的根因：是否是 max_tokens 超限。"""
    import sys
    sys.path.insert(0, "~/kuiwork/workdir2/litertlm")
    from swe_agent import config as C
    import requests

    SYSTEM = "你是顶层软件架构 Planner，只输出一个 JSON 对象。Schema: {\"summary\":\"\",\"modules\":[],\"interface\":[],\"forbidden\":[],\"tasks\":[{\"step\":\"\",\"deliverables\":[],\"hidden_tests\":[]}]}"
    USER = "用 Python 写一个函数 count_neighbors(grid, r, c)。"

    for mt in [1024, 4096, 8192, 32768]:
        payload = {
            "model": "glm-4.1v-thinking-flashx",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
            "temperature": 0.2, "max_tokens": mt,
            "response_format": {"type": "json_object"},
        }
        try:
            r = requests.post(f"{C.GLM_BASE_URL}/chat/completions", json=payload,
                              headers={"Content-Type": "application/json", "Authorization": f"Bearer {C.GLM_API_TOKEN}"},
                              timeout=C.GLM_TIMEOUT)
            ok = r.status_code == 200
            print(f"max_tokens={mt:6d} -> HTTP {r.status_code}  {'OK' if ok else r.text[:160]}")
        except Exception as e:
            print(f"max_tokens={mt:6d} -> EXC {e}")
    print("完成。")


if __name__ == "__main__":
    main()


def test_main():
    main()
