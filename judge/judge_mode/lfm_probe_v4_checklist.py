#!/usr/bin/env python3
"""judge 探针 v4 —— 针对 v3 的两个失败点做 prompt 修复，并扩到 10 用例。

v3 失败点：
  P3 "调研做完了，看前面就行。"  → 1.2B 判 yes（客套话漏判，致命：放行坏产出）
  P6 长文陷阱（只有概念解释、零需求说明）→ 1.2B 与 Spark 双双判 yes（被专业感欺骗）

修复思路 D4：把「两条标准都要满足」写成强制 checklist，要求逐项确认后再给结论，
确认过程写进 reason 字段（不破坏 JSON 结构）。

同时打印 LM Studio 响应的真实 model 字段 —— 防止「请求 A 模型、实际 B 模型回答」的静默顶替。

用法：项目根目录执行
  ./.venv/bin/python judge/judge_mode/lfm_probe_v4_checklist.py
"""
import json
import time
import urllib.error
import urllib.request

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

SYS_BASE = """你是严格的评审Judge。
规则：
1. 禁止输出任何思考过程、禁止输出标签，禁止解释推理草稿。
2. 只输出严格JSON，不要额外前言、后记、markdown```标记。
3. 返回结构固定：{"result":"yes/no/notsure","reason":"简短理由"}
4. 所有判断理由放在reason字段，不要写在别处。
"""

# D4 新增：强制逐项 checklist，禁止"看起来专业就放行"
SYS_CHECKLIST = SYS_BASE + """
5. 判定前必须逐项确认评审标准里的每一条，缺任何一条都必须判 no。
6. 内容长、术语多、看起来专业，都不等于达标；只看标准里要求的要素是否真的存在。
"""

STD_A = "评审标准：①有概念解释 ②有需求说明。两条必须同时满足，缺一条即判 no。"
STD_T = "评审标准：①有有效的单元测试 ②有有效的实现。两条必须同时满足，缺一条即判 no。"
IRON_A = "【铁律】若内容为空、仅为占位符或客套话（如'已完成''详见上文''看前面就行'等无实质内容的表述）、或没有实质的概念解释与需求说明，必须回答 no，不得给 yes。"
IRON_T = "【铁律】若实现文件或测试文件为空、测试数量为0、测试只是空壳占位、或只是占位符或客套话，必须回答 no，不得给 yes 或 notsure。"

# ---------------- 10 个用例（v3 的 8 个 + 2 个新增） ----------------
A_GOOD_ORIG = ("## 概念解释\n斐波那契数列是第 n 项等于前两项之和的整数序列，递推式 F(n)=F(n-1)+F(n-2)，"
               "边界 F(0)=0、F(1)=1。\n## 需求说明\n任务要求用 Python 在 src/fib.py 实现 fib(n) 返回前 n 项列表，"
               "fib(0)=[]、fib(1)=[1]、fib(2)=[1,1]，并提供 main() CLI 入口；测试根目录 test_fib.py，"
               "用 from src.fib import fib 导入，覆盖 fib(0)/fib(1)/fib(10) 基准值，pytest 全绿。")
T_EMPTY_ORIG = "实现文件为空，测试文件为空，没有任何 pytest 函数。"
A_GOOD_PARA = ("斐波那契数列中每个数都由它前面两个数相加得到，起点是 0 和 1。本次要交付的东西很明确："
               "写一个 Python 模块，里面的函数接收 n 并返回长度为 n 的数列，n 为 0 时给空列表，"
               "n 为 1 时给 [1]；另外要能在命令行直接跑。测试放在仓库根目录，直接 import 被测函数，"
               "至少覆盖 n=0、n=1、n=10 三种情形，要求 pytest 全绿。")
A_EMPTY_PARA = "（此处没有任何内容）"
A_FLUFF_PARA = "调研做完了，看前面就行。"
T_GOOD_PARA = ("src/fib.py 里实现了求斐波那契数列的函数，并带有命令行入口；"
               "仓库根目录的测试文件里写了七个测试函数，全部通过。")
T_EMPTY_PARA = "没写任何代码，也没编任何测试。"
A_TRAP = ("## 概念解释\n斐波那契数列最早出现在印度数学著作中，由比萨的莱昂纳多（斐波那契）在 1202 年"
          "《计算之书》里借兔子繁殖问题引入欧洲。它与黄金分割密切相关：相邻两项之比趋近于"
          " (1+sqrt(5))/2 ≈ 1.618。该序列在植物叶序、向日葵种子排列、蜜蜂家系树中广泛出现，"
          "在算法领域常用作递归与动态规划的教学样例，也可用矩阵快速幂在 O(log n) 内求解。\n")
# P7 新增：有实现，但测试是空壳（pass 占位）→ 应 no
T_SHELL = ("src/fib.py 已实现 fib(n) 与 main()；test_fib.py 里写了三个测试函数，"
           "函数体都是 pass，没有任何断言。")
# P8 新增：只有需求说明，没有概念解释 → 应 no
A_NEED_ONLY = ("## 需求说明\n任务要求用 Python 在 src/fib.py 实现 fib(n) 返回前 n 项列表，"
               "fib(0)=[]、fib(1)=[1]，测试放根目录，pytest 全绿。")

CASES = [
    ("analyzer", "锚点/good原文(应yes)", A_GOOD_ORIG, "yes"),
    ("tester", "锚点/empty原文(应no)", T_EMPTY_ORIG, "no"),
    ("analyzer", "P1 good换措辞(应yes)", A_GOOD_PARA, "yes"),
    ("analyzer", "P2 empty换措辞(应no)", A_EMPTY_PARA, "no"),
    ("analyzer", "P3 fluff换措辞(应no)", A_FLUFF_PARA, "no"),
    ("tester", "P4 good换措辞(应yes)", T_GOOD_PARA, "yes"),
    ("tester", "P5 empty换措辞(应no)", T_EMPTY_PARA, "no"),
    ("analyzer", "P6 长文陷阱(应no)", A_TRAP, "no"),
    ("tester", "P7 测试空壳(应no)", T_SHELL, "no"),
    ("analyzer", "P8 只有需求(应no)", A_NEED_ONLY, "no"),
]

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_result", "strict": "true",
        "schema": {"type": "object",
                   "properties": {"result": {"type": "string", "enum": ["yes", "no", "notsure"]},
                                  "reason": {"type": "string"}},
                   "required": ["result", "reason"], "additionalProperties": False},
    },
}


def strict_parse(raw):
    try:
        v = json.loads((raw or "").strip())
    except Exception:
        return "PARSE_FAIL"
    if not isinstance(v, dict) or "result" not in v or "reason" not in v:
        return "PARSE_FAIL"
    r = v.get("result")
    if not isinstance(r, str) or r.strip().lower() not in ("yes", "no", "notsure"):
        return "PARSE_FAIL"
    return r.strip().lower()


def user_text(kind, content):
    std, iron = (STD_A, IRON_A) if kind == "analyzer" else (STD_T, IRON_T)
    return f"{std}{iron}\n待评审产出：\n{content}\n"


def run_mlx(name, path, modes):
    model, tok = load(path, tokenizer_config={"trust_remote_code": True})
    out = {}
    for mode, sys_p in modes.items():
        print(f"\n===== {name} [{mode}]  (mlx_lm 直连) =====")
        hit = 0
        for kind, label, content, exp in CASES:
            p = tok.apply_chat_template(
                [{"role": "system", "content": sys_p},
                 {"role": "user", "content": user_text(kind, content)}],
                tokenize=False, add_generation_prompt=True)
            t0 = time.time()
            raw = generate(model, tok, prompt=p, max_tokens=160,
                           sampler=make_sampler(temp=0.0), verbose=False)
            dt = round(time.time() - t0, 1)
            got = strict_parse(raw)
            ok = got == exp
            hit += ok
            print(f"  {label:<22} exp={exp:<4} {'OK  ' if ok else 'MISS'} "
                  f"{'' if ok else got + ' <-- 崩':<12} dt={dt}s  raw={raw[:64]!r}")
        print(f"  --> {mode} 合计 {hit}/{len(CASES)}")
        out[mode] = hit
    del model, tok
    return out


def run_lmstudio(model_id, modes):
    out = {}
    for mode, sys_p in modes.items():
        print(f"\n===== 请求 {model_id} [{mode}]  (LM Studio + json_schema) =====")
        hit, served = 0, set()
        for kind, label, content, exp in CASES:
            payload = {"model": model_id,
                       "messages": [{"role": "system", "content": sys_p},
                                    {"role": "user", "content": user_text(kind, content)}],
                       "temperature": 0.0, "max_tokens": 1024, "response_format": JUDGE_SCHEMA}
            req = urllib.request.Request(LMSTUDIO_URL, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read().decode())
                dt = round(time.time() - t0, 1)
                raw = resp["choices"][0]["message"].get("content") or ""
                served.add(resp.get("model", "?"))
            except Exception as e:  # noqa
                dt, raw = round(time.time() - t0, 1), f"{type(e).__name__}: {e}"[:120]
            got = strict_parse(raw)
            ok = got == exp
            hit += ok
            print(f"  {label:<22} exp={exp:<4} {'OK  ' if ok else 'MISS'} "
                  f"{'' if ok else got + ' <-- 崩':<12} dt={dt}s  raw={raw[:64]!r}")
        print(f"  --> {mode} 合计 {hit}/{len(CASES)}   [实际服务模型: {served or '?'}]")
        out[mode] = hit
    return out


def main():
    modes = {"D3基线": SYS_BASE, "D4checklist": SYS_CHECKLIST}
    print("== 探针 v4：checklist 修复 + 扩到 10 用例 ==")
    print("严格口径：解析失败/枚举非法 → PARSE_FAIL（≠期望即判错）\n")
    print("== 汇总 ==")
    res = {}
    res["LM Studio 通道"] = run_lmstudio("Spark-X2.5-1.7B", modes)
    res["LFM2.5-1.2B 直连"] = run_mlx("LFM2.5-1.2B-Instruct",
                                       "models/LFM2.5-1.2B-Instruct-MLX-8bit", modes)
    print("\n== 汇总 ==")
    for eng, d in res.items():
        for mode, hit in d.items():
            print(f"  {eng:<22} {mode:<14} {hit}/{len(CASES)}")


if __name__ == "__main__":
    main()
