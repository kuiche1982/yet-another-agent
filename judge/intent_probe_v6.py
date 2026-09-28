#!/usr/bin/env python3
"""意图分类探针 v6 —— 修复 v5 暴露的死因：36 值枚举下 350M 有 32/40 自造意图。

v5 结论：剥围栏后 JSON 破损 0/40（格式没问题），但枚举非法 32/40，且 confidence 恒为 high
（兜底分支永不触发）。对比 judge 场景 3 值枚举能守住 → 假设：枚举集合大小是关键变量。

本探针验证三个修复方向：
  M1 = 36 值原集合（基线复现）
  M2 = 8 值粗粒度集合
  M3 = 8 值 + few-shot（3 例）
  M4 = 8 值 + 编号输出（只吐 intent_id 数字，降低生成难度）

核心指标：
  枚举合规率 = 输出的 intent 确实来自给定集合（这是"能不能用"的硬门槛）
  准确率      = 在合规结果中，intent 命中期望集合的比例
  confidence 分布 = 决定"升级大模型"的兜底分支能否触发

用法：项目根目录执行
  ./.venv/bin/python judge/intent_probe_v6.py
"""
import json
import re
import time
from collections import Counter

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = "models/LFM2.5-350M-MLX-4bit"

# ---------------- 8 值粗粒度集合 + 36→8 映射 ----------------
COARSE = ["code", "write", "analyze", "search", "config", "teach", "qa", "chat"]
MAP36_8 = {
    "code_generate": "code", "code_debug": "code", "code_explain": "code",
    "code_refactor": "code", "code_ask_api": "code", "arch_design": "code",
    "doc_write": "write", "doc_compose": "write", "essay_draft": "write",
    "material_compose": "write",
    "data_analysis": "analyze", "summary_text": "analyze", "doc_summary": "analyze",
    "summarize": "analyze", "translate": "analyze", "idea_consult": "analyze",
    "info_search": "search", "search": "search",
    "config_modify": "config", "prompt_optimize": "config", "troubleshoot": "config",
    "query_status": "config", "task_execute": "config",
    "knowledge_explain": "teach", "exercise_make": "teach", "text_evaluate": "teach",
    "course_design": "teach",
    "knowledge_qa": "qa", "qa": "qa", "exercise_solve": "qa", "life_qa": "qa",
    "fault_diagnose": "qa", "step_guide": "qa", "plan_make": "qa",
    "chat": "chat", "unknown": "chat",
}
SET36 = sorted(set(MAP36_8) | {"unknown"})

FEWSHOT = """
示例：
用户输入：写一个快速排序
输出：{"intent":"code","confidence":"high","reason":"要求生成代码"}
用户输入：今天心情不太好
输出：{"intent":"chat","confidence":"high","reason":"闲聊无明确任务"}
用户输入：帮我把这份表格做个统计
输出：{"intent":"analyze","confidence":"high","reason":"要求对数据做分析"}
"""

# M5：8 类每类各 1 个示例（8-shot 全覆盖）
FEWSHOT8 = """
示例：
用户输入：写一个快速排序
输出：{"intent":"code","confidence":"high","reason":"要求生成代码"}
用户输入：帮我写一封辞职信
输出：{"intent":"write","confidence":"high","reason":"要求撰写文本"}
用户输入：帮我把这份表格做个统计
输出：{"intent":"analyze","confidence":"high","reason":"要求对数据做分析"}
用户输入：查一下阿里云账单API有哪些字段
输出：{"intent":"search","confidence":"high","reason":"查询事实资料"}
用户输入：把judge最大重试次数改成2
输出：{"intent":"config","confidence":"high","reason":"修改系统配置"}
用户输入：怎么给学生讲明白递归
输出：{"intent":"teach","confidence":"high","reason":"教学讲解"}
用户输入：空调不启动是什么原因
输出：{"intent":"qa","confidence":"high","reason":"常识问答"}
用户输入：今天天气不错
输出：{"intent":"chat","confidence":"high","reason":"闲聊"}
"""


def sys_prompt(mode):
    if mode == "M1_36值":
        opts = ",".join(SET36)
        head = "你是意图分类器。\n约束：\n1. 禁止输出思考过程，禁止标签，禁止markdown```标记。\n" \
               "2. 只输出裸JSON，不输出其他文字。\n" \
               "3. 返回格式固定：{\"intent\":\"意图名称\",\"confidence\":\"high/medium/low\",\"reason\":\"简短依据\"}\n"
        return head + f"可选意图集合：{opts}\n只从上面集合选intent字段的值，不要自己造意图。\n"
    opts = ",".join(COARSE)
    head = "你是意图分类器。\n约束：\n1. 禁止输出思考过程，禁止标签，禁止markdown```标记。\n" \
           "2. 只输出裸JSON，不输出其他文字。\n"
    if mode == "M4_8值编号":
        numbered = "  ".join(f"{i + 1}={n}" for i, n in enumerate(COARSE))
        return head + "3. 返回格式固定：{\"intent_id\":<编号>,\"confidence\":\"high/medium/low\",\"reason\":\"简短依据\"}\n" \
               + f"可选意图编号：{numbered}\n只输出上面存在的编号，不要自己造编号。\n"
    body = head + "3. 返回格式固定：{\"intent\":\"意图名称\",\"confidence\":\"high/medium/low\",\"reason\":\"简短依据\"}\n" \
           + f"可选意图集合：{opts}\n只从上面集合选intent字段的值，不要自己造意图。\n"
    if mode == "M3_8值_fewshot":
        body += FEWSHOT
    if mode == "M5_8值_8shot":
        body += FEWSHOT8
    return body


# (query, acceptable_36_set, group)
CASES = [
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
    ("我们线上有个服务最近一到高峰期响应就变慢，日志里能看到大量超时，我想先搞清楚可能的原因有哪些，"
     "再决定要不要加缓存或者扩容，你帮我梳理一下排查思路", {"troubleshoot", "arch_design", "qa"}, "long"),
    ("我在做一个本地小模型跑评测的项目，需要设计一套能自动打分并且把失败case归档的流水线，"
     "涉及模型调用、结果解析和报告生成几个环节，帮我规划下模块怎么拆", {"arch_design"}, "long"),
    ("帮我看看这个", {"unknown"}, "vague"),
    ("那个东西怎么样了", {"unknown", "query_status"}, "vague"),
    ("随便聊聊", {"chat"}, "vague"),
    ("帮我写个脚本读取日志并解释它的逻辑", {"code_generate", "code_explain"}, "multi"),
    ("翻译这段英文并总结一下要点", {"translate", "summarize", "summary_text"}, "multi"),
    ("今天天气不错", {"chat"}, "chat"),
    ("你好", {"chat"}, "chat"),
]


def strip_fence(s):
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[1] if len(parts) > 1 else s
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    return s


def parse(raw, mode):
    """返回 (intent, confidence, enum_ok, json_ok)。"""
    s = strip_fence(raw)
    if not s:
        return None, None, False, False
    try:
        v = json.loads(s)
    except Exception:
        return None, None, False, False
    if not isinstance(v, dict):
        return None, None, False, False
    conf = str(v.get("confidence", "")).strip().lower()
    conf_ok = conf in ("high", "medium", "low")
    if mode == "M4_8值编号":
        raw_id = v.get("intent_id")
        try:
            idx = int(str(raw_id).strip()) - 1
        except Exception:
            return None, conf if conf_ok else None, False, True
        if 0 <= idx < len(COARSE):
            return COARSE[idx], conf if conf_ok else None, True, True
        return str(raw_id), conf if conf_ok else None, False, True
    intent = str(v.get("intent", "")).strip()
    valid = SET36 if mode == "M1_36值" else COARSE
    return intent, conf if conf_ok else None, intent in valid, True


def main():
    model, tok = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})
    print("== 意图探针 v6 (LFM2.5-350M)：枚举集合大小 / few-shot / 编号输出 ==")
    print(f"用例 {len(CASES)}，temp=0.0 greedy，解析一律先剥 ```json 围栏\n")
    for mode in ("M1_36值", "M2_8值", "M3_8值_fewshot", "M4_8值编号", "M5_8值_8shot"):
        sp = sys_prompt(mode)
        rows = []
        for query, accept36, group in CASES:
            accept8 = {MAP36_8[a] for a in accept36}
            p = tok.apply_chat_template(
                [{"role": "system", "content": sp},
                 {"role": "user", "content": f"用户输入：{query}\n"}],
                tokenize=False, add_generation_prompt=True)
            t0 = time.time()
            raw = generate(model, tok, prompt=p, max_tokens=96,
                           sampler=make_sampler(temp=0.0), verbose=False)
            dt = round(time.time() - t0, 2)
            intent, conf, enum_ok, json_ok = parse(raw, mode)
            hit = intent in accept8 if enum_ok else False
            rows.append((group, intent, conf, enum_ok, json_ok, hit, dt, raw))
        n = len(rows)
        json_ok = sum(1 for r in rows if r[4])
        enum_ok = sum(1 for r in rows if r[3])
        hit = sum(1 for r in rows if r[5])
        conf_dist = Counter(r[2] for r in rows if r[3])
        g = {}
        for r in rows:
            g.setdefault(r[0], []).append(r[5])
        print(f"--- {mode} ---")
        print(f"    JSON可解析 {json_ok}/{n}   枚举合规 {enum_ok}/{n} ({enum_ok / n * 100:.0f}%)   "
              f"准确率(合规中命中) {hit}/{enum_ok}"
              f"{f' ({hit / enum_ok * 100:.0f}%)' if enum_ok else ''}   "
              f"端到端 {hit}/{n} ({hit / n * 100:.0f}%)")
        print(f"    confidence 分布(合规样本) {dict(conf_dist)}")
        print(f"    分组端到端: " + "  ".join(f"{k}={sum(v)}/{len(v)}" for k, v in sorted(g.items())))
        bad = [r for r in rows if not r[3]]
        if bad:
            print(f"    自造意图样例: " + "; ".join(
                f"{str(r[1])[:18]!r}" for r in bad[:6]))
        print(f"    延迟 avg={sum(r[6] for r in rows) / n:.2f}s\n")


if __name__ == "__main__":
    main()
