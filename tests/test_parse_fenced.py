#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # -*- coding: utf-8 -*-
    """parse_actions 解析回归测试（标准 OpenAI toolcall 产物）。

    原生工具调用路径下，传输层（glm / lmstudio）已把模型输出规约为合法 JSON 数组字符串
    （[{"action": ...}, ...]）。本模块只做轻量解析：去 ```json 围栏、单对象自动包数组，
    不再保留 plaintext 那套「三引号 / XML / 宽松正则」容错（已随 rapid-mlx 一并废弃）。
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from swe_agent.actions import parse_actions

    PASS, FAIL = 0, 0

    def check(name, cond, detail=""):
        nonlocal PASS, FAIL
        if cond:
            PASS += 1
            print(f"  ok   {name}")
        else:
            FAIL += 1
            print(f"  FAIL {name}  {detail}")

    # ---- 合法 JSON 数组（原生 tool_calls 的返回形态） ----
    SAMPLE_C = '[{"action":"write_file","path":"a.py","content":"print(1)\\n"}]'
    print("== 合法 JSON 数组（原生 tool_calls 路径） ==")
    acts = parse_actions(SAMPLE_C)
    check("解析出 write_file", len(acts) == 1 and acts[0].get("action") == "write_file", str(acts))
    check("content 转义正确", acts and acts[0].get("content") == "print(1)\n", repr(acts[0].get("content") if acts else None))

    # ---- 单对象（无数组括号）自动包数组 ----
    print("== 单对象自动包数组 ==")
    acts = parse_actions('{"action":"read_file","path":"b.py"}')
    check("单对象被包装为 1 个动作", len(acts) == 1 and acts[0].get("action") == "read_file", str(acts))

    # ---- ```json 围栏被剥离 ----
    print("== ```json 围栏剥离 ==")
    acts = parse_actions('```json\n[{"action":"shell","command":"ls"}]\n```')
    check("围栏内动作被解析", len(acts) == 1 and acts[0].get("action") == "shell", str(acts))

    # ---- 空串返回空列表 ----
    print("== 空串 ==")
    check("空串返回 []", parse_actions("") == [])
    check("纯空白返回 []", parse_actions("   \n") == [])

    print(f"\n{'ALL PASS' if FAIL == 0 else 'HAS FAILURES'}: {PASS} ok / {FAIL} fail")
    assert FAIL == 0, f"{FAIL} checks failed"


if __name__ == "__main__":
    main()


def test_main():
    main()
