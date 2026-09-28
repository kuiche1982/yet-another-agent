#!/usr/bin/env python3
"""judge 小模型探针 v2 —— 按「豆包建议」口径重跑本地 mlx 小模型能否顶替 Ling。

与 v1 的差异（豆包口径，逐条可证伪）：
  1. sampler = make_sampler(temp=0.0) 纯 greedy，不带 top_p/top_k/min_p；
  2. 关闭 thinking 靠 system prompt 约束（mlx-lm 无 API 开关）；
  3. 严格 json.loads 校验，失败即 Judge 失败 → 按 notsure 计，不做正则修补；
  4. 失败/或 notsure 时，temp=0.3 重试 1 次（豆包的重试档）。

变量：
  D1 = 豆包原案：system prompt 前置 + 裸 prompt 直传（不套 chat_template）
  D2 = 同一 system prompt 走 chat_template（system/user 分角色）
  D3 = D2 + 线上 config.JUDGE_PROMPTS 的「空/占位符/客套话→必须 no」铁律

用法：项目根目录执行
  ./.venv/bin/python judge/judge_mode/lfm_probe_v2.py
  PROBE_MODELS=350m ./.venv/bin/python judge/judge_mode/lfm_probe_v2.py   # 只跑 350M
"""
import gc
import json
import os
import re
import time

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS = {
    "350m": "models/LFM2.5-350M-MLX-4bit",
    "1.2b": "models/LFM2.5-1.2B-Instruct-MLX-8bit",
    "0.5b": "models/qwen2.5-0.5B-Instruct-MLX-4bit",
}

# ---------------- 豆包的 system prompt（原文照搬） ----------------
JUDGE_SYSTEM_PROMPT = """你是严格的评审Judge。
规则：
1. 禁止输出任何思考过程、禁止输出标签，禁止解释推理草稿。
2. 只输出严格JSON，不要额外前言、后记、markdown```标记。
3. 返回结构固定：{"result":"yes/no/notsure","reason":"简短理由"}
4. 所有判断理由放在reason字段，不要写在别处。
"""

# 线上 config.JUDGE_PROMPTS 的铁律（实测：弱模型不加此铁律会把退化输入误判 yes/notsure）
IRON_RULE_ANALYZER = "【铁律】若内容为空、仅为占位符或客套话（如'已完成''详见上文'但无具体内容）、或没有实质的概念解释与需求说明，必须回答 no，不得给 yes。"
IRON_RULE_TESTER = "【铁律】若实现文件或测试文件为空、测试数量为0、或只是占位符或客套话，必须回答 no，不得给 yes 或 notsure。"

STD = {
    "analyzer": "评审标准：1. 是否包含有效的调研信息；标准：①有概念解释 ②有需求说明。",
    "tester": "评审标准：1. 是否包含有效的单元测试；2. 是否包含有效的实现。",
    "phone": "评审标准：判断待评审内容里的字符串是否为合法中国大陆手机号（11位数字且以1开头）。",
}

A_GOOD = ("## 概念解释\n斐波那契数列是第 n 项等于前两项之和的整数序列，递推式 F(n)=F(n-1)+F(n-2)，"
          "边界 F(0)=0、F(1)=1。\n## 需求说明\n任务要求用 Python 在 src/fib.py 实现 fib(n) 返回前 n 项列表，"
          "fib(0)=[]、fib(1)=[1]、fib(2)=[1,1]，并提供 main() CLI 入口；测试根目录 test_fib.py，"
          "用 from src.fib import fib 导入，覆盖 fib(0)/fib(1)/fib(10) 基准值，pytest 全绿。")
A_FLUFF = "已完成调研，内容详见上文。"
T_GOOD = ("交付文件：src/fib.py 实现 fib(n) 递推返回列表、main() CLI；test_fib.py 含 7 个 pytest 函数"
          "(test_fib_zero/one/two/ten/recurrence/invalid_input/main_runs)，覆盖基准与递推，运行 7 passed。")
T_BAD = "实现文件为空，测试文件为空，没有任何 pytest 函数。"
PHONE = "Is the following string a valid phone number? 13521871956"

# (kind, label, content, expected)
CASES = [
    ("analyzer", "good(应yes)", A_GOOD, "yes"),
    ("analyzer", "empty(应no)", "", "no"),
    ("analyzer", "fluff(应no)", A_FLUFF, "no"),
    ("tester", "good(应yes)", T_GOOD, "yes"),
    ("tester", "empty(应no)", T_BAD, "no"),
    ("phone", "豆包原用例(应yes)", PHONE, "yes"),
]

MAX_TOKENS = 160


def build_D1(kind, content):
    """豆包原案：system prompt 与标准、内容拼成一个裸 prompt 直传。"""
    return f"{JUDGE_SYSTEM_PROMPT}\n{STD[kind]}\n待评审产出：\n{content}\n"


def build_chat(tok, kind, content, iron_rule=None):
    """D2/D3：system 与 user 分角色，套 chat_template。"""
    user = STD[kind]
    if iron_rule:
        user += iron_rule
    user += f"\n待评审产出：\n{content}\n"
    return tok.apply_chat_template(
        [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def strict_parse(raw):
    """豆包口径：只 json.loads + 字段/枚举硬校验，不做任何修补。失败 → PARSE_FAIL(等价 notsure)。"""
    try:
        v = json.loads(raw.strip())
    except Exception:
        return "PARSE_FAIL", False
    if not isinstance(v, dict) or "result" not in v or "reason" not in v:
        return "PARSE_FAIL", False
    r = v.get("result")
    if not isinstance(r, str) or r.strip().lower() not in ("yes", "no", "notsure"):
        return "PARSE_FAIL", False
    return r.strip().lower(), True


def loose_parse(raw):
    """v1 口径（仅作对照，证明两种口径差距）：剥围栏 + 抽 {...} + 正则兜底。"""
    s = raw.strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[1] if len(parts) > 1 else s
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        v = json.loads(s)
        if isinstance(v, dict) and str(v.get("result", "")).lower() in ("yes", "no", "notsure"):
            return str(v["result"]).lower()
    except Exception:
        pass
    a, b = s.find("{"), s.rfind("}")
    if a >= 0 and b > a:
        try:
            v = json.loads(s[a:b + 1])
            if isinstance(v, dict) and str(v.get("result", "")).lower() in ("yes", "no", "notsure"):
                return str(v["result"]).lower()
        except Exception:
            pass
    m = re.search(r"\b(yes|no|notsure)\b", s, re.I)
    return m.group(1).lower() if m else "PARSE_FAIL"


def score(got, exp):
    """严格口径：PARSE_FAIL 视为 notsure（harness 行为），notsure≠期望即判错。"""
    return got == exp


def run_model(name, path, modes, reps):
    t0 = time.time()
    model, tok = load(path, tokenizer_config={"trust_remote_code": True})
    print(f"\n===== {name}  ({path})  load={round(time.time() - t0, 1)}s =====")

    for mode in modes:
        rows, hit, strict_ok, dts = [], 0, 0, []
        for kind, label, content, exp in CASES:
            if mode == "D1":
                p = build_D1(kind, content)
            elif mode == "D2":
                p = build_chat(tok, kind, content)
            else:
                p = build_chat(tok, kind, content,
                               IRON_RULE_ANALYZER if kind == "analyzer" else
                               IRON_RULE_TESTER if kind == "tester" else None)

            got, got_loose, raws, d = [], [], [], []
            for _ in range(reps):
                t1 = time.time()
                raw = generate(model, tok, prompt=p, max_tokens=MAX_TOKENS,
                               sampler=make_sampler(temp=0.0), verbose=False)
                d.append(round(time.time() - t1, 1))
                raws.append(raw)
                g, ok = strict_parse(raw)
                got.append(g); got_loose.append(loose_parse(raw))
                strict_ok += ok
            # 重试档：严格口径失败/或 notsure 才跑 temp=0.3
            retry = ""
            if got[0] in ("PARSE_FAIL", "notsure"):
                raw2 = generate(model, tok, prompt=p, max_tokens=MAX_TOKENS,
                                sampler=make_sampler(temp=0.3), verbose=False)
                g2, _ = strict_parse(raw2)
                retry = f" → retry@0.3: {g2}"
                if score(g2, exp):
                    got[0] = g2
            ok = score(got[0], exp)
            hit += ok
            dts += d
            rows.append(f"  {mode} {kind + '/' + label:<20} exp={exp:<9} got={got[0]:<11} "
                        f"loose={got_loose[0]:<11} {'OK  ' if ok else 'MISS'} dt={max(d)}s{retry}  "
                        f"raw={raws[0][:60]!r}")
        print("\n".join(rows))
        print(f"  --> {mode}: 正确 {hit}/{len(CASES)}  严格JSON合规 {strict_ok}/{len(CASES) * reps}  "
              f"延迟 max={max(dts)}s")
    del model, tok
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()


def sampler_claim_check(model, tok):
    """证伪豆包 claims#1：temp=0 时 top_p/top_k/min_p 是否真的无效（输出应逐字相同）。"""
    p = build_chat(tok, "phone", PHONE)
    a = generate(model, tok, prompt=p, max_tokens=MAX_TOKENS,
                 sampler=make_sampler(temp=0.0), verbose=False)
    b = generate(model, tok, prompt=p, max_tokens=MAX_TOKENS,
                 sampler=make_sampler(temp=0.0, top_p=0.9, top_k=10, min_p=0.2), verbose=False)
    print("\n[claims#1] temp=0 时 top_p/top_k/min_p 是否无效："
          f"{'成立（输出逐字相同）' if a == b else '不成立（输出不同）'}")
    print(f"   纯 greedy : {a[:90]!r}")
    print(f"   带采样参数: {b[:90]!r}")


def main():
    sel = os.environ.get("PROBE_MODELS", "350m,1.2b,0.5b").split(",")
    modes = os.environ.get("PROBE_MODES", "D1,D2,D3").split(",")
    reps = int(os.environ.get("PROBE_REPS", "2"))
    print(f"== 探针 v2（豆包口径）models={sel} modes={modes} reps={reps} ==")
    print("严格口径：解析失败/枚举非法 → PARSE_FAIL（harness 等价 notsure，≠期望即判错）")
    for key in sel:
        if key not in MODELS:
            continue
        run_model(key, MODELS[key], modes, reps)
        if key == sel[0]:
            # claims#1 只在首个模型上验证一次
            m, t = load(MODELS[key], tokenizer_config={"trust_remote_code": True})
            sampler_claim_check(m, t)
            del m, t
            gc.collect()


if __name__ == "__main__":
    main()
