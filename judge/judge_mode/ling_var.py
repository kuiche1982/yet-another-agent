#!/usr/bin/env python3
"""聚焦方差测试：Ling-3.0-Tiny (带思考) max_tokens=8192 下 json_schema 输出是否稳定。
analyzer-good 用例连打 3 次；解析容错：剥 <think>...</think>，抓首个{到末个}。
"""
import json, os, re, urllib.request, time

URL = "http://localhost:1234/v1/chat/completions"
MODEL = os.environ.get("JUDGE_MODEL", "Ling-3.0-Tiny")
SCHEMA = {"type": "json_schema", "json_schema": {"name": "judge_result", "strict": "true",
    "schema": {"type": "object", "properties": {"result": {"type": "string", "enum": ["yes", "no", "notsure"]},
    "reason": {"type": "string"}}, "required": ["result", "reason"], "additionalProperties": False}}}
CONTENT = ("## 概念解释\n斐波那契数列是第 n 项等于前两项之和的整数序列，递推式 F(n)=F(n-1)+F(n-2)，"
    "边界 F(0)=0、F(1)=1。\n## 需求说明\n任务要求用 Python 在 src/fib.py 实现 fib(n) 返回前 n 项列表，"
    "fib(0)=[]、fib(1)=[1]、fib(2)=[1,1]，并提供 main() CLI 入口；测试根目录 test_fib.py，"
    "用 from src.fib import fib 导入，覆盖 fib(0)/fib(1)/fib(10) 基准值，pytest 全绿。")
PROMPT = ("你是严格的调研质量审校。判断以下调研结果是否包含有效的调研信息。标准：1. 有概念解释；2. 有需求说明。"
    "【铁律】若内容为空、仅为占位符或客套话、或没有实质的概念解释与需求说明，必须回答 no。只有确实同时包含两者才回答 yes。"
    "只输出结构化结果。\n\n【调研结果】\n" + CONTENT)


def extract_json(raw: str):
    s = raw.strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s), True, ""
    except Exception as e:  # noqa
        pass
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(s[a:b+1]), True, "slice"
        except Exception as e:  # noqa
            return None, False, f"slice_fail:{e}"
    return None, False, "no_brace"


def call(temp):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
               "temperature": temp, "max_tokens": 1024, "response_format": SCHEMA,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read().decode())
        dt = round(time.time() - t0, 1)
        fr = resp["choices"][0].get("finish_reason")
        raw = resp["choices"][0]["message"].get("content") or ""
        val, ok, how = extract_json(raw)
        res = val.get("result") if isinstance(val, dict) else "-"
        return {"ok": ok, "finish": fr, "len": len(raw), "dt": dt, "res": res, "how": how, "head": raw[:90]}
    except Exception as e:  # noqa
        return {"ok": False, "finish": "ERR", "len": -1, "dt": round(time.time() - t0, 1), "res": "-", "how": str(e)[:50], "head": ""}


print(f"== Ling 方差测试(max_tokens=8192): {MODEL} ==")
for attempt in range(3):
    r = call(1.1)
    print(f"  attempt {attempt+1}: {'JSON_OK' if r['ok'] else 'FAIL'} finish={r['finish']} result={r['res']} "
          f"len={r['len']} dt={r['dt']}s how={r['how']} raw={r['head']!r}")
