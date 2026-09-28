#!/usr/bin/env python3
"""实证探针：弱模型(Spark-X2.5-1.7B)能否稳定返回 json_schema 结构化 judge 结果。
不修改任何项目源码，仅打 LM Studio localhost:1234/v1/chat/completions。
"""
import json, os, urllib.request, urllib.error, time

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

ANALYZER_GOOD = (
    "## 概念解释\n斐波那契数列是第 n 项等于前两项之和的整数序列，递推式 F(n)=F(n-1)+F(n-2)，"
    "边界 F(0)=0、F(1)=1。\n## 需求说明\n任务要求用 Python 在 src/fib.py 实现 fib(n) 返回前 n 项列表，"
    "fib(0)=[]、fib(1)=[1]、fib(2)=[1,1]，并提供 main() CLI 入口；测试根目录 test_fib.py，"
    "用 from src.fib import fib 导入，覆盖 fib(0)/fib(1)/fib(10) 基准值，pytest 全绿。"
)
ANALYZER_BAD = ""  # 空调研
TESTER_GOOD = (
    "交付文件：src/fib.py 实现 fib(n) 递推返回列表、main() CLI；test_fib.py 含 7 个 pytest 函数"
    "(test_fib_zero/one/two/ten/recurrence/invalid_input/main_runs)，覆盖基准与递推，运行 7 passed。"
)
TESTER_BAD = "实现文件为空，测试文件为空，没有任何 pytest 函数。"


def judge_prompt(role: str, content: str) -> str:
    if role == "analyzer":
        return ("请判断以下调研结果是否包含有效的调研信息。标准：1. 有概念解释；2. 有需求说明。"
                "只调用结构化输出，不要写额外文字。\n\n【调研结果】\n" + content)
    return ("请判断以下交付是否合格。标准：1. 是否包含有效的单元测试；2. 是否包含有效的实现；"
            "不能是空文件、不能是空测试。只调用结构化输出，不要写额外文字。\n\n【交付】\n" + content)


def call(temp: float, role: str, content: str):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": judge_prompt(role, content)}],
        "temperature": temp,
        "max_tokens": 1024,
        "response_format": JUDGE_SCHEMA,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode("utf-8"))
        dt = time.time() - t0
        msg = resp["choices"][0]["message"]
        fr = resp["choices"][0].get("finish_reason")
        raw = msg.get("content") or ""
        # 有的引擎把 json 包在 ```json 里，做一次清理
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        parsed_ok, parse_err, schema_ok, val = False, "", False, None
        try:
            val = json.loads(cleaned)
            parsed_ok = True
            schema_ok = (
                isinstance(val, dict)
                and val.get("result") in ("yes", "no", "notsure")
                and isinstance(val.get("reason"), str)
            )
        except Exception as e:  # noqa
            parse_err = f"{type(e).__name__}: {e}"
        return {"http": 200, "dt": round(dt, 1), "finish": fr, "raw_head": raw[:120],
                "parsed_ok": parsed_ok, "parse_err": parse_err, "schema_ok": schema_ok, "val": val}
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        body = e.read().decode("utf-8", "replace")[:300]
        return {"http": e.code, "dt": round(dt, 1), "finish": None,
                "raw_head": body, "parsed_ok": False, "parse_err": f"HTTP {e.code}",
                "schema_ok": False, "val": None}
    except Exception as e:  # noqa
        return {"http": -1, "dt": round(time.time() - t0, 1), "finish": None,
                "raw_head": f"{type(e).__name__}: {e}"[:200], "parsed_ok": False,
                "parse_err": str(e)[:200], "schema_ok": False, "val": None}


CASES = [
    ("analyzer", "good(应yes)", ANALYZER_GOOD),
    ("analyzer", "empty(应no)", ANALYZER_BAD),
    ("tester", "good(应yes)", TESTER_GOOD),
    ("tester", "empty(应no)", TESTER_BAD),
]
TEMPS = [0.0, 0.5, 1.1]

print(f"== 探针模型: {MODEL}  endpoint: {URL} ==")
for role, label, content in CASES:
    print(f"\n### {role} / {label}")
    for temp in TEMPS:
        r = call(temp, role, content)
        mark = "OK" if (r["parsed_ok"] and r["schema_ok"]) else "FAIL"
        res = r["val"].get("result") if r["val"] else "-"
        print(f"  temp={temp:<4} http={r['http']} {mark} parse={r['parsed_ok']} schema={r['schema_ok']} "
              f"result={res} finish={r['finish']} dt={r['dt']}s raw={r['raw_head']!r}")
        if not r["schema_ok"] and r["parse_err"]:
            print(f"         parse_err={r['parse_err']}")
