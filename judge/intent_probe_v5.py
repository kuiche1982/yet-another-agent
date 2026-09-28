#!/usr/bin/env python3
"""LFM2.5-350M 轻量意图分类可行性探针 v5。

评估「350M 做意图初筛、confidence=low / 解析失败即升级大模型」这套方案是否成立。

三档解析（关键：上一轮 350M 100% 输出 ```json 围栏，裸解析恒失败）：
  strict  = 裸 json.loads          （豆包口径，不做任何修补）
  fence   = 剥 ```json 围栏后 loads （工程上安全的确定性容错，不引入幻觉）
  loose   = 正则抠 intent          （仅作对照，会引入误判风险）

用例来源：豆包给出的场景-意图候选表（直接用它自己的示例 query），
另加 长query / 模糊 / 多意图 / 闲聊 四组压力样本。

指标：
  - 各解析档的 JSON 合规率
  - intent top-1 准确率（可接受集合：候选表本身有重叠，见下方 ACCEPT 说明）
  - confidence 分布
  - 路由有效率 = P(confidence==high) × 该档准确率   ← 真正决定初筛价值的指标

用法：项目根目录执行
  ./.venv/bin/python judge/intent_probe_v5.py
"""
import json
import re
import time
from collections import Counter

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = "models/LFM2.5-350M-MLX-4bit"

# ---------------- 豆包的 system prompt（原文照搬） ----------------
INTENT_SYSTEM_PROMPT = """你是意图分类器。
约束：
1. 禁止输出思考过程，禁止标签，禁止markdown```标记。
2. 只输出裸JSON，不输出其他文字。
3. 返回格式固定：{"intent":"意图名称","confidence":"high/medium/low","reason":"简短依据"}
confidence说明：
high：意图明确无歧义
medium：有一定倾向，但存在其他可能性
low：模糊、多意图混杂、无法判断
可选意图集合：
code_generate,code_debug,code_explain,code_refactor,code_ask_api,arch_design,doc_write,
task_execute,config_modify,troubleshoot,query_status,prompt_optimize,
knowledge_qa,exercise_solve,summary_text,translate,essay_draft,
doc_summary,doc_compose,data_analysis,info_search,idea_consult,
life_qa,step_guide,fault_diagnose,plan_make,
knowledge_explain,exercise_make,text_evaluate,course_design,material_compose,
search,qa,summarize,chat,unknown
只从上面集合选intent字段的值，不要自己造意图。
"""

INTENT_SET = {
    "code_generate", "code_debug", "code_explain", "code_refactor", "code_ask_api",
    "arch_design", "doc_write", "task_execute", "config_modify", "troubleshoot",
    "query_status", "prompt_optimize", "knowledge_qa", "exercise_solve", "summary_text",
    "translate", "essay_draft", "doc_summary", "doc_compose", "data_analysis",
    "info_search", "idea_consult", "life_qa", "step_guide", "fault_diagnose",
    "plan_make", "knowledge_explain", "exercise_make", "text_evaluate",
    "course_design", "material_compose", "search", "qa", "summarize", "chat", "unknown",
}

# 候选表本身存在重叠（translate 出现两次；summary_text/doc_summary/summarize 三者重叠；
# knowledge_qa/qa、info_search/search 重叠），故期望值用「可接受集合」，避免低估模型。
# (query, acceptable_intent_set, group)
CASES = [
    # ---- 短 query：直接取自候选表自己的示例 ----
    ("写一个python异步http客户端", {"code_generate"}, "short"),
    ("这段代码为什么报index out of range", {"code_debug"}, "short"),
    ("帮我解释这一段go代码逻辑", {"code_explain"}, "short"),
    ("把这个函数改得更简洁高效", {"code_refactor"}, "short"),
    ("mlx-lm generate参数怎么配置", {"code_ask_api"}, "short"),
    ("设计一个harness judge模块", {"arch_design"}, "short"),
    ("帮我写SKILL.md文档片段", {"doc_write"}, "short"),
    ("调用agent帮我分析这份调研文档", {"task_execute"}, "short"),
    ("把judge最大重试次数改成2", {"config_modify"}, "short"),
    ("为什么tester judge一直返回notsure", {"troubleshoot"}, "short"),
    ("看下刚才的analyzer任务结果", {"query_status"}, "short"),
    ("帮我改写judge的system prompt", {"prompt_optimize"}, "short"),
    ("解释什么是MoE架构", {"knowledge_qa", "qa"}, "short"),
    ("求解这道数学题", {"exercise_solve"}, "short"),
    ("总结这篇文章核心要点", {"summary_text", "summarize"}, "short"),
    ("翻译这段英文", {"translate"}, "short"),
    ("帮我写一段议论文", {"essay_draft", "doc_compose"}, "short"),
    ("总结会议纪要", {"doc_summary", "summarize", "summary_text"}, "short"),
    ("写一份周报、汇报PPT大纲", {"doc_compose"}, "short"),
    ("分析这份账单数据", {"data_analysis"}, "short"),
    ("查一下阿里云账单API字段", {"info_search", "search"}, "short"),
    ("给一个推广方案建议", {"idea_consult"}, "short"),
    ("空调不启动是什么原因", {"life_qa", "fault_diagnose"}, "short"),
    ("教我更换插座步骤", {"step_guide"}, "short"),
    ("机器异响哪里出问题", {"fault_diagnose"}, "short"),
    ("帮我规划出行路线", {"plan_make"}, "short"),
    ("怎么给学生讲递归概念", {"knowledge_explain"}, "short"),
    ("生成10道练习题", {"exercise_make"}, "short"),
    ("点评这篇学生作文", {"text_evaluate"}, "short"),
    ("设计一堂课的大纲", {"course_design"}, "short"),
    ("生成课堂案例素材", {"material_compose"}, "short"),
    # ---- 长 query（>60 字）----
    ("我们线上有个服务最近一到高峰期响应就变慢，日志里能看到大量超时，我想先搞清楚可能的原因有哪些，"
     "再决定要不要加缓存或者扩容，你帮我梳理一下排查思路", {"troubleshoot", "arch_design", "qa"}, "long"),
    ("我在做一个本地小模型跑评测的项目，需要设计一套能自动打分并且把失败case归档的流水线，"
     "涉及模型调用、结果解析和报告生成几个环节，帮我规划下模块怎么拆", {"arch_design"}, "long"),
    # ---- 模糊 ----
    ("帮我看看这个", {"unknown"}, "vague"),
    ("那个东西怎么样了", {"unknown", "query_status"}, "vague"),
    ("随便聊聊", {"chat"}, "vague"),
    # ---- 多意图 ----
    ("帮我写个脚本读取日志并解释它的逻辑", {"code_generate", "code_explain"}, "multi"),
    ("翻译这段英文并总结一下要点", {"translate", "summarize", "summary_text"}, "multi"),
    # ---- 闲聊 ----
    ("今天天气不错", {"chat"}, "chat"),
    ("你好", {"chat"}, "chat"),
]


def build_raw(query):
    """豆包原案：system 与用户输入拼成一个裸 prompt 直传。"""
    return f"{INTENT_SYSTEM_PROMPT}\n用户输入：{query}\n"


def build_chat(tok, query):
    """system / user 分角色 + chat_template。"""
    return tok.apply_chat_template(
        [{"role": "system", "content": INTENT_SYSTEM_PROMPT},
         {"role": "user", "content": f"用户输入：{query}\n"}],
        tokenize=False, add_generation_prompt=True)


def parse(raw):
    """返回 (intent, confidence, level)；level ∈ strict / fence / loose / fail。"""
    if not raw or not raw.strip():
        return None, None, "fail"
    s = raw.strip()
    # strict
    try:
        v = json.loads(s)
        if isinstance(v, dict) and str(v.get("intent", "")) in INTENT_SET \
                and str(v.get("confidence", "")) in ("high", "medium", "low"):
            return v["intent"], v["confidence"], "strict"
    except Exception:
        pass
    # fence
    t = s
    if t.startswith("```"):
        parts = t.split("```", 2)
        t = parts[1] if len(parts) > 1 else t
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        v = json.loads(t)
        if isinstance(v, dict) and str(v.get("intent", "")) in INTENT_SET \
                and str(v.get("confidence", "")) in ("high", "medium", "low"):
            return v["intent"], v["confidence"], "fence"
    except Exception:
        pass
    # loose
    mi = re.search(r'"intent"\s*:\s*"([a-z_]+)"', t)
    mc = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', t)
    if mi and mi.group(1) in INTENT_SET:
        return mi.group(1), (mc.group(1) if mc else "low"), "loose"
    return None, None, "fail"


def run(mode, tok, model):
    print(f"\n===== 形态 {mode} =====")
    rows = []
    for query, accept, group in CASES:
        p = build_raw(query) if mode == "A_raw" else build_chat(tok, query)
        t0 = time.time()
        raw = generate(model, tok, prompt=p, max_tokens=96,
                       sampler=make_sampler(temp=0.0), verbose=False)
        dt = round(time.time() - t0, 2)
        intent, conf, level = parse(raw)
        hit = intent in accept if intent else False
        rows.append((query, group, accept, intent, conf, level, hit, dt, raw))
        flag = "OK  " if hit else ("FAIL" if level == "fail" else "MISS")
        print(f"  [{group:<5}] {query[:26]:<28} exp={'/'.join(sorted(accept))[:26]:<26} "
              f"got={str(intent):<18} conf={str(conf):<6} {level:<6} {flag} dt={dt}s")
    return rows


def summarize(rows, min_level):
    """min_level: 允许的最低解析档（strict < fence < loose）。"""
    order = {"strict": 0, "fence": 1, "loose": 2, "fail": 3}
    usable = [r for r in rows if order[r[5]] <= order[min_level]]
    if not usable:
        print(f"  [{min_level}档] 无可用结果")
        return
    hit = sum(1 for r in usable if r[6])
    conf_dist = Counter(r[4] for r in usable)
    groups = {}
    for r in usable:
        groups.setdefault(r[1], []).append(r[6])
    print(f"  [{min_level}档] 可用 {len(usable)}/{len(rows)}  准确率 {hit}/{len(usable)} "
          f"({hit / len(usable) * 100:.0f}%)")
    print(f"        confidence 分布: {dict(conf_dist)}")
    high = [r for r in usable if r[4] == "high"]
    if high:
        hh = sum(1 for r in high if r[6])
        print(f"        confidence=high 占 {len(high)}/{len(usable)} "
              f"({len(high) / len(usable) * 100:.0f}%)，其中正确 {hh}/{len(high)} "
              f"({hh / len(high) * 100:.0f}%)")
        print(f"        >>> 路由有效率(high且正确 / 全部) = {hh}/{len(rows)} "
              f"({hh / len(rows) * 100:.0f}%)")
    print("        分组准确率: " + "  ".join(
        f"{g}={sum(v)}/{len(v)}" for g, v in sorted(groups.items())))


def main():
    model, tok = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})
    print("== 意图分类探针 v5 (LFM2.5-350M-MLX-4bit) ==")
    print(f"用例 {len(CASES)} 个，temp=0.0 greedy\n")
    all_rows = {}
    for mode in ("A_raw", "B_chat"):
        rows = run(mode, tok, model)
        all_rows[mode] = rows
        print(f"\n  ---- {mode} 汇总 ----")
        for lvl in ("strict", "fence", "loose"):
            summarize(rows, lvl)
        dts = [r[7] for r in rows]
        print(f"        延迟 avg={sum(dts) / len(dts):.2f}s max={max(dts)}s")


if __name__ == "__main__":
    main()
