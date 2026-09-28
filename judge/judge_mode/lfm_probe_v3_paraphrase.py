#!/usr/bin/env python3
"""judge 探针 v3 —— 扰动测试：证伪「LFM2.5-1.2B 的 4/6 是真语义判别」这一假设。

v2 里 1.2B 拿到 4/6 + 严格 JSON 合规 12/12。但它判对的 tester/empty 原文含字面「为空」，
判错的 analyzer/empty 是真·空字符串（连个「空」字都没有）→ 怀疑是关键词匹配而非语义判别。

扰动手法：**语义完全不变，只换措辞，专门剥掉/替换掉原用例里的表面关键词**。
若模型是真语义判别，扰动后 verdict 应保持不变；若崩，则 v2 的 4/6 是过拟合到用例措辞的假象。

对照组（同为 1-2B 小模型，唯一变量是通道）：
  - LFM2.5-1.2B / 350M : mlx_lm 直连，prompt 约束（无约束解码）
  - Spark-X2.5-1.7B    : LM Studio + response_format=json_schema（引擎约束解码）

用法：项目根目录执行
  ./.venv/bin/python judge/judge_mode/lfm_probe_v3_paraphrase.py
"""
import json
import os
import time
import urllib.error
import urllib.request

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

# ---- 与 v2 D3 完全一致的 system prompt + 标准 + 铁律（控制变量：两边 prompt 逐字相同）----
JUDGE_SYSTEM_PROMPT = """你是严格的评审Judge。
规则：
1. 禁止输出任何思考过程、禁止输出标签，禁止解释推理草稿。
2. 只输出严格JSON，不要额外前言、后记、markdown```标记。
3. 返回结构固定：{"result":"yes/no/notsure","reason":"简短理由"}
4. 所有判断理由放在reason字段，不要写在别处。
"""
STD = {
    "analyzer": "评审标准：1. 是否包含有效的调研信息；标准：①有概念解释 ②有需求说明。",
    "tester": "评审标准：1. 是否包含有效的单元测试；2. 是否包含有效的实现。",
}
IRON = {
    "analyzer": "【铁律】若内容为空、仅为占位符或客套话（如'已完成''详见上文'但无具体内容）、"
                "或没有实质的概念解释与需求说明，必须回答 no，不得给 yes。",
    "tester": "【铁律】若实现文件或测试文件为空、测试数量为0、或只是占位符或客套话，"
              "必须回答 no，不得给 yes 或 notsure。",
}

# ---------------- 原始锚点用例（复现 v2 结果，确认基线一致） ----------------
A_GOOD_ORIG = ("## 概念解释\n斐波那契数列是第 n 项等于前两项之和的整数序列，递推式 F(n)=F(n-1)+F(n-2)，"
               "边界 F(0)=0、F(1)=1。\n## 需求说明\n任务要求用 Python 在 src/fib.py 实现 fib(n) 返回前 n 项列表，"
               "fib(0)=[]、fib(1)=[1]、fib(2)=[1,1]，并提供 main() CLI 入口；测试根目录 test_fib.py，"
               "用 from src.fib import fib 导入，覆盖 fib(0)/fib(1)/fib(10) 基准值，pytest 全绿。")
T_EMPTY_ORIG = "实现文件为空，测试文件为空，没有任何 pytest 函数。"

# ---------------- 扰动用例：语义不变，剥掉关键词 ----------------
# P1 原始是带 "## 概念解释/## 需求说明" 标题 → 改成散文，无标题关键词，语义完整
A_GOOD_PARA = ("斐波那契数列中每个数都由它前面两个数相加得到，起点是 0 和 1。本次要交付的东西很明确："
               "写一个 Python 模块，里面的函数接收 n 并返回长度为 n 的数列，n 为 0 时给空列表，"
               "n 为 1 时给 [1]；另外要能在命令行直接跑。测试放在仓库根目录，直接 import 被测函数，"
               "至少覆盖 n=0、n=1、n=10 三种情形，要求 pytest 全绿。")
# P2 原始是真·空字符串 → 换成一句"什么都没写"的声明（有字，但语义仍是"无内容"）
A_EMPTY_PARA = "（此处没有任何内容）"
# P3 原始 "已完成调研，内容详见上文。"（含"已完成"/"详见上文"）→ 换成同义但无这些字面
A_FLUFF_PARA = "调研做完了，看前面就行。"
# P4 原始含 "pytest" 字样 → 换成不含该词的等价描述
T_GOOD_PARA = ("src/fib.py 里实现了求斐波那契数列的函数，并带有命令行入口；"
               "仓库根目录的测试文件里写了七个测试函数，全部通过。")
# P5 原始 "实现文件为空，测试文件为空"（含"为空"字面）→ 换成不含"为空"的等价表述
T_EMPTY_PARA = "没写任何代码，也没编任何测试。"
# P6 反向陷阱：内容长且专业，但只有概念、零需求说明 → 应 no（判 yes = 被长度/专业感欺骗）
A_TRAP = ("## 概念解释\n斐波那契数列最早出现在印度数学著作中，由比萨的莱昂纳多（斐波那契）在 1202 年"
          "《计算之书》里借兔子繁殖问题引入欧洲。它与黄金分割密切相关：相邻两项之比趋近于"
          " (1+sqrt(5))/2 ≈ 1.618。该序列在植物叶序、向日葵种子排列、蜜蜂家系树中广泛出现，"
          "在算法领域常用作递归与动态规划的教学样例，也可用矩阵快速幂在 O(log n) 内求解。\n")

CASES = [
    ("analyzer", "锚点/good原文(应yes)", A_GOOD_ORIG, "yes"),
    ("tester", "锚点/empty原文(应no)", T_EMPTY_ORIG, "no"),
    ("analyzer", "P1 good换措辞(应yes)", A_GOOD_PARA, "yes"),
    ("analyzer", "P2 empty换措辞(应no)", A_EMPTY_PARA, "no"),
    ("analyzer", "P3 fluff换措辞(应no)", A_FLUFF_PARA, "no"),
    ("tester", "P4 good换措辞(应yes)", T_GOOD_PARA, "yes"),
    ("tester", "P5 empty换措辞(应no)", T_EMPTY_PARA, "no"),
    ("analyzer", "P6 长文陷阱(应no)", A_TRAP, "no"),
]

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


def strict_parse(raw):
    """严格口径（豆包口径）：只 json.loads + 字段/枚举硬校验，不修补。失败 → PARSE_FAIL。"""
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
    return f"{STD[kind]}{IRON[kind]}\n待评审产出：\n{content}\n"


# ---------------- 通道 A：mlx_lm 直连 ----------------
def run_mlx(name, path):
    model, tok = load(path, tokenizer_config={"trust_remote_code": True})
    print(f"\n===== {name}  (mlx_lm 直连, prompt 约束) =====")
    hit = 0
    for kind, label, content, exp in CASES:
        p = tok.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
             {"role": "user", "content": user_text(kind, content)}],
            tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        raw = generate(model, tok, prompt=p, max_tokens=160,
                       sampler=make_sampler(temp=0.0), verbose=False)
        dt = round(time.time() - t0, 1)
        got = strict_parse(raw)
        ok = got == exp
        hit += ok
        mark = "OK  " if ok else "MISS"
        if not ok:
            got = f"{got}  <-- 崩"
        print(f"  {label:<22} exp={exp:<4} {mark} dt={dt}s  raw={raw[:72]!r}")
    print(f"  --> 合计 {hit}/{len(CASES)}")
    del model, tok
    return hit


# ---------------- 通道 B：LM Studio + json_schema ----------------
def run_lmstudio(model_id):
    print(f"\n===== {model_id}  (LM Studio + json_schema 引擎约束解码) =====")
    hit = 0
    for kind, label, content, exp in CASES:
        payload = {
            "model": model_id,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                         {"role": "user", "content": user_text(kind, content)}],
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": JUDGE_SCHEMA,
        }
        req = urllib.request.Request(
            LMSTUDIO_URL, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read().decode())
            dt = round(time.time() - t0, 1)
            raw = resp["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:
            dt = round(time.time() - t0, 1)
            raw = e.read().decode("utf-8", "replace")[:200]
        except Exception as e:  # noqa
            dt = round(time.time() - t0, 1)
            raw = f"{type(e).__name__}: {e}"[:200]
        got = strict_parse(raw)
        ok = got == exp
        hit += ok
        mark = "OK  " if ok else "MISS"
        if not ok:
            got = f"{got}  <-- 崩"
        print(f"  {label:<22} exp={exp:<4} {mark} dt={dt}s  raw={raw[:72]!r}")
    print(f"  --> 合计 {hit}/{len(CASES)}")
    return hit


def main():
    print("== 探针 v3：扰动测试（语义不变、剥掉关键词）==")
    print("严格口径：解析失败/枚举非法 → PARSE_FAIL（≠期望即判错）\n")
    scores = {}
    try:
        scores["Spark-1.7B(LM Studio)"] = run_lmstudio("Spark-X2.5-1.7B")
    except Exception as e:  # noqa
        print(f"[跳过 LM Studio 通道] {type(e).__name__}: {e}")
    scores["LFM2.5-1.2B(mlx_lm)"] = run_mlx("LFM2.5-1.2B-Instruct", "models/LFM2.5-1.2B-Instruct-MLX-8bit")
    scores["LFM2.5-350M(mlx_lm)"] = run_mlx("LFM2.5-350M", "models/LFM2.5-350M-MLX-4bit")
    print("\n== 汇总 ==")
    for k, v in scores.items():
        print(f"  {k:<26} {v}/{len(CASES)}")


if __name__ == "__main__":
    main()
