#!/usr/bin/env python3
"""探针2：hardened judge prompt —— 看 Spark 能否被 prompt 救回（空/退化必须 no）。
复用 probe1 已验证可解析的多行 json_schema 形态。
"""
import json, os, urllib.request, time

URL = "http://localhost:1234/v1/chat/completions"
MODEL = os.environ.get("JUDGE_MODEL", "Spark-X2.5-1.7B")
JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_result",
        "strict": "true",
        "schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string", "enum": ["yes", "no", "notsure"]},
                "reason": {"type": "string"},
            },
            "required": ["result", "reason"],
            "additionalProperties": False,
        },
    },
}

ANALYZER_GOOD = ("## 概念解释\n斐波那契数列是第 n 项等于前两项之和的整数序列，递推式 F(n)=F(n-1)+F(n-2)，"
    "边界 F(0)=0、F(1)=1。\n## 需求说明\n任务要求用 Python 在 src/fib.py 实现 fib(n) 返回前 n 项列表，"
    "fib(0)=[]、fib(1)=[1]、fib(2)=[1,1]，并提供 main() CLI 入口；测试根目录 test_fib.py，"
    "用 from src.fib import fib 导入，覆盖 fib(0)/fib(1)/fib(10) 基准值，pytest 全绿。")
ANALYZER_BAD = ""
ANALYZER_FLUFF = "已完成调研，内容详见上文。"
TESTER_GOOD = ("交付文件：src/fib.py 实现 fib(n) 递推返回列表、main() CLI；test_fib.py 含 7 个 pytest 函数"
    "(test_fib_zero/one/two/ten/recurrence/invalid_input/main_runs)，覆盖基准与递推，运行 7 passed。")
TESTER_BAD = "实现文件为空，测试文件为空，没有任何 pytest 函数。"


def judge_prompt(role, content):
    if role == "analyzer":
        return ("你是严格的调研质量审校。判断以下调研结果是否包含有效的调研信息。标准：1. 有概念解释；2. 有需求说明。"
                "【铁律】若内容为空、仅为占位符或客套话（如'已完成''详见上文'但无具体内容）、"
                "或没有实质的概念解释与需求说明，必须回答 no，不得给 yes。只有确实同时包含两者才回答 yes。"
                "只输出结构化结果。\n\n【调研结果】\n" + content)
    return ("你是严格的测试质量审校。判断以下交付是否合格。标准：1. 是否包含有效的单元测试；2. 是否包含有效的实现。"
            "【铁律】若实现文件或测试文件为空、测试数量为0、或只是占位符或客套话，必须回答 no，不得给 yes 或 notsure。"
            "只有确实包含有效非空单测与非空实现才回答 yes。只输出结构化结果。\n\n【交付】\n" + content)


def call(temp, role, content):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": judge_prompt(role, content)}],
               "temperature": temp, "max_tokens": 1024, "response_format": JUDGE_SCHEMA,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read().decode())
    raw = resp["choices"][0]["message"].get("content") or ""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        val = json.loads(cleaned)
        ok = isinstance(val, dict) and val.get("result") in ("yes", "no", "notsure") and isinstance(val.get("reason"), str)
        return {"ok": ok, "res": val.get("result"), "reason": (val.get("reason") or "")[:80]}
    except Exception as e:  # noqa
        return {"ok": False, "res": "PARSE_FAIL", "reason": str(e)[:60]}


CASES = [("analyzer", "good(应yes)", ANALYZER_GOOD), ("analyzer", "empty(应no)", ANALYZER_BAD),
         ("analyzer", "fluff(应no)", ANALYZER_FLUFF), ("tester", "good(应yes)", TESTER_GOOD),
         ("tester", "empty(应no)", TESTER_BAD)]
print(f"== 探针2( hardened prompt ) 模型 {MODEL} ==")
for role, label, content in CASES:
    print(f"\n### {role} / {label}")
    for temp in (0.0, 1.1):
        r = call(temp, role, content)
        print(f"  temp={temp:<4} {'OK' if r['ok'] else 'FAIL'} result={r['res']} reason={r['reason']!r}")
