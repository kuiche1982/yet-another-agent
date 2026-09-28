#!/usr/bin/env python3
"""意图探针 v7 —— 找 350M 的「可用边界」。

v6 已证：8 类细粒度路由端到端只有 20%，不可用。但 350M 本地 0.2s、零 API 成本，
只要它能拦下一部分流量，经济账就可能是正的。所以真正的问题不是「行不行」，
而是「任务难度降到多低才值得用」。

本探针测两个**真正有业务价值**的粗粒度二分（平衡测试集，避免全猜一类的假高分）：
  T1  code vs not_code —— 技术类请求初筛（决定是否拉起代码工具链）
  T2  chat vs task     —— 闲聊识别（闲聊直接回，不启动 agent 链路，省的是整条链路）
  T3  8 类（v6 的 M3 形态，作为已知对照）

核心指标（沿用 v6）：
  枚举合规率 / 准确率 / confidence 分布
  **危险率 = 合规但分类错误 且 confidence=high** ← 这部分不触发升级，会 confidently 路由错

用法：项目根目录执行
  ./.venv/bin/python judge/intent_probe_v7_boundary.py
"""
import json
import time
from collections import Counter

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = "models/LFM2.5-350M-MLX-4bit"

# ---------------- T1: code vs not_code（各 20） ----------------
T1 = [
    ("帮我写个快速排序", "code"),
    ("用python写个异步http客户端", "code"),
    ("这段代码为什么报index out of range", "code"),
    ("帮我看看这个函数哪里写错了", "code"),
    ("解释一下这段go代码的逻辑", "code"),
    ("把这个函数改得更简洁高效", "code"),
    ("给这段代码加上单元测试", "code"),
    ("mlx-lm的generate参数怎么配置", "code"),
    ("设计一个harness judge模块", "code"),
    ("用go实现一个限流器", "code"),
    ("帮我重构这个类的继承关系", "code"),
    ("这个sql为什么这么慢", "code"),
    ("写个脚本批量重命名文件", "code"),
    ("docker容器启动就退出怎么排查", "code"),
    ("把这个接口改成异步的", "code"),
    ("react的useEffect依赖数组怎么写", "code"),
    ("帮我看看这个报错日志", "code"),
    ("实现一个lru缓存", "code"),
    ("这个正则匹配不到我想匹配的内容", "code"),
    ("把这段python代码翻译成go", "code"),
    ("今天天气不错", "not_code"),
    ("帮我写一份周报", "not_code"),
    ("总结下这份会议纪要", "not_code"),
    ("翻译这段英文", "not_code"),
    ("解释什么是MoE架构", "not_code"),
    ("空调不启动是什么原因", "not_code"),
    ("教我更换插座的步骤", "not_code"),
    ("帮我规划出行路线", "not_code"),
    ("给一个推广方案建议", "not_code"),
    ("生成10道练习题", "not_code"),
    ("点评这篇学生作文", "not_code"),
    ("分析这份账单数据", "not_code"),
    ("查一下阿里云账单API字段", "not_code"),
    ("帮我写一段议论文", "not_code"),
    ("怎么给学生讲明白递归", "not_code"),
    ("设计一堂课的大纲", "not_code"),
    ("你好", "not_code"),
    ("把judge最大重试次数改成2", "not_code"),
    ("为什么tester judge一直返回notsure", "not_code"),
    ("帮我改写judge的system prompt", "not_code"),
]

# ---------------- T2: chat vs task（各 20） ----------------
T2 = [
    ("你好", "chat"),
    ("今天天气不错", "chat"),
    ("随便聊聊", "chat"),
    ("谢谢", "chat"),
    ("你叫什么名字", "chat"),
    ("讲个笑话", "chat"),
    ("晚安", "chat"),
    ("哈哈", "chat"),
    ("在吗", "chat"),
    ("好的", "chat"),
    ("有意思", "chat"),
    ("再见", "chat"),
    ("早上好", "chat"),
    ("辛苦了", "chat"),
    ("嗯嗯", "chat"),
    ("我明白了", "chat"),
    ("真棒", "chat"),
    ("你觉得呢", "chat"),
    ("无语", "chat"),
    ("哈哈哈哈哈太搞笑了", "chat"),
    ("帮我写个快速排序", "task"),
    ("这段代码为什么报错", "task"),
    ("帮我写一份周报", "task"),
    ("总结下这份会议纪要", "task"),
    ("翻译这段英文", "task"),
    ("解释什么是MoE架构", "task"),
    ("空调不启动是什么原因", "task"),
    ("帮我规划出行路线", "task"),
    ("分析这份账单数据", "task"),
    ("查一下阿里云账单API字段", "task"),
    ("生成10道练习题", "task"),
    ("把judge最大重试次数改成2", "task"),
    ("为什么tester judge一直返回notsure", "task"),
    ("教我更换插座的步骤", "task"),
    ("帮我改写judge的system prompt", "task"),
    ("给一个推广方案建议", "task"),
    ("设计一个harness judge模块", "task"),
    ("帮我看看这个", "task"),
    ("求解这道数学题", "task"),
    ("点评这篇学生作文", "task"),
]

COARSE8 = ["code", "write", "analyze", "search", "config", "teach", "qa", "chat"]
# T3 复用 v6 的 8 类用例（取 v6 的部分，标注 8 类期望）
T3 = [
    ("帮我写个快速排序", "code"), ("这段代码为什么报错", "code"),
    ("解释一下这段go代码的逻辑", "code"), ("把这个函数改得更简洁", "code"),
    ("mlx-lm的generate参数怎么配置", "code"), ("设计一个harness judge模块", "code"),
    ("帮我写一份周报", "write"), ("帮我写一段议论文", "write"),
    ("帮我写SKILL.md文档片段", "write"), ("生成课堂案例素材", "write"),
    ("分析这份账单数据", "analyze"), ("总结下这份会议纪要", "analyze"),
    ("翻译这段英文", "analyze"), ("总结这篇文章核心要点", "analyze"),
    ("查一下阿里云账单API字段", "search"), ("搜一下最近的AI新闻", "search"),
    ("把judge最大重试次数改成2", "config"), ("为什么tester judge一直返回notsure", "config"),
    ("看下刚才的analyzer任务结果", "config"), ("调用agent帮我分析这份调研文档", "config"),
    ("怎么给学生讲明白递归", "teach"), ("生成10道练习题", "teach"),
    ("点评这篇学生作文", "teach"), ("设计一堂课的大纲", "teach"),
    ("解释什么是MoE架构", "qa"), ("空调不启动是什么原因", "qa"),
    ("求解这道数学题", "qa"), ("帮我规划出行路线", "qa"),
    ("你好", "chat"), ("今天天气不错", "chat"),
]


def sys_for(labels, few):
    opts = ",".join(labels)
    s = ("你是意图分类器。\n约束：\n1. 禁止输出思考过程，禁止标签，禁止markdown```标记。\n"
         "2. 只输出裸JSON，不输出其他文字。\n"
         "3. 返回格式固定：{\"intent\":\"意图名称\",\"confidence\":\"high/medium/low\",\"reason\":\"简短依据\"}\n"
         f"可选意图集合：{opts}\n只从上面集合选intent字段的值，不要自己造意图。\n")
    s += few
    return s


FEW_T1 = """
示例：
用户输入：写个脚本批量改文件名
输出：{"intent":"code","confidence":"high","reason":"要求写代码"}
用户输入：帮我写一份周报
输出：{"intent":"not_code","confidence":"high","reason":"写作任务与代码无关"}
"""
FEW_T2 = """
示例：
用户输入：写一个快速排序
输出：{"intent":"task","confidence":"high","reason":"有明确任务"}
用户输入：今天心情不太好
输出：{"intent":"chat","confidence":"high","reason":"闲聊无明确任务"}
"""
FEW_T3 = """
示例：
用户输入：写一个快速排序
输出：{"intent":"code","confidence":"high","reason":"要求生成代码"}
用户输入：帮我写一封辞职信
输出：{"intent":"write","confidence":"high","reason":"要求撰写文本"}
用户输入：帮我把这份表格做个统计
输出：{"intent":"analyze","confidence":"high","reason":"要求对数据做分析"}
用户输入：今天天气不错
输出：{"intent":"chat","confidence":"high","reason":"闲聊"}
"""


FEW_T1_UNK = FEW_T1 + """
用户输入：那个东西怎么弄
输出：{"intent":"unknown","confidence":"low","reason":"表述模糊，无法判断是否与代码相关"}
"""
FEW_T2_UNK = FEW_T2 + """
用户输入：那个东西怎么样了
输出：{"intent":"unknown","confidence":"low","reason":"指代不明，无法判断是否有任务"}
"""
FEW_T3_UNK = FEW_T3 + """
用户输入：帮我看看这个
输出：{"intent":"unknown","confidence":"low","reason":"表述模糊，无法归类"}
"""


def strip_fence(s):
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[1] if len(parts) > 1 else s
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    return s


def run(tag, cases, labels, few, tok, model):
    """labels 含 unknown 时，unknown 视为「主动弃权」→ 走升级（安全路径）。"""
    sp = sys_for(labels, few)
    n = len(cases)
    json_ok = covered = hit = danger = unknown = nonenum = 0
    conf_dist, wrong_samples = Counter(), []
    dts = []
    for query, exp in cases:
        p = tok.apply_chat_template(
            [{"role": "system", "content": sp},
             {"role": "user", "content": f"用户输入：{query}\n"}],
            tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        raw = generate(model, tok, prompt=p, max_tokens=96,
                       sampler=make_sampler(temp=0.0), verbose=False)
        dts.append(time.time() - t0)
        s = strip_fence(raw)
        intent = conf = None
        jok = eok = False
        try:
            v = json.loads(s)
            jok = True
            if isinstance(v, dict):
                intent = str(v.get("intent", "")).strip()
                c = str(v.get("confidence", "")).strip().lower()
                conf = c if c in ("high", "medium", "low") else None
                eok = intent in labels
        except Exception:
            pass
        json_ok += jok
        if not eok:
            nonenum += 1                      # 解析失败/自造意图 → 升级（安全）
        elif intent == "unknown":
            unknown += 1                      # 主动弃权 → 升级（安全）
        else:
            covered += 1                      # 350M 实际拦下的流量
            conf_dist[conf] += 1
            if intent == exp:
                hit += 1
            else:
                wrong_samples.append((query, exp, intent, conf))
                if conf == "high":
                    danger += 1               # 合规但错 + high → 直接路由错（危险）
    safe = unknown + nonenum
    print(f"\n--- {tag}  (平衡集 {n} 例, 随机基线 {100 / (len(labels) - 1):.0f}%) ---")
    print(f"    JSON可解析 {json_ok}/{n}")
    print(f"    覆盖率(350M 实际拦下) {covered}/{n} ({covered / n * 100:.0f}%)   "
          f"其中正确 {hit}/{covered}" + (f" ({hit / covered * 100:.0f}%)" if covered else ""))
    print(f"    端到端正确 {hit}/{n} ({hit / n * 100:.0f}%)")
    print(f"    confidence 分布(已拦下样本) {dict(conf_dist)}")
    print(f"    [安全] 升级大模型(弃权{unknown}+不合规{nonenum}): {safe}/{n} ({safe / n * 100:.0f}%)")
    print(f"    [危险] 合规但错+high→直接路由错: {danger}/{n} ({danger / n * 100:.0f}%)")
    print(f"    延迟 avg={sum(dts) / n:.2f}s")
    if wrong_samples:
        print("    错例: " + "; ".join(
            f"{q[:14]}(应{e}→得{i}/{c})" for q, e, i, c in wrong_samples[:6]))
    return {"e2e": hit / n, "danger": danger / n, "cover": covered / n,
            "acc_in": (hit / covered) if covered else 0.0, "safe": safe / n}


def main():
    model, tok = load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})
    print("== 探针 v7：350M 的可用边界（粗粒度二分 vs 8 类细分）==")
    print("平衡测试集，避免「全猜一类」刷分；temp=0.0 greedy")
    print("对照重点：**强制二选一 vs 带 unknown 逃生口** —— 量化强制选择造成的分数失真\n")
    res = {}
    # ---- A 组：强制二选一（无 unknown，模型被迫选，v7 初版的设计缺陷）----
    run("A1 code/not_code（强制二选一）", T1, ["code", "not_code"], FEW_T1, tok, model)
    run("A2 chat/task（强制二选一）", T2, ["chat", "task"], FEW_T2, tok, model)
    # ---- B 组：带 unknown 逃生口（三选一）----
    res["B1 code/not_code+unknown"] = run(
        "B1 code/not_code/unknown（有逃生口）", T1, ["code", "not_code", "unknown"], FEW_T1_UNK, tok, model)
    res["B2 chat/task+unknown"] = run(
        "B2 chat/task/unknown（有逃生口）", T2, ["chat", "task", "unknown"], FEW_T2_UNK, tok, model)
    res["B3 8类+unknown（对照）"] = run(
        "B3 8 类细分 + unknown（v6 已知弱项）", T3, COARSE8 + ["unknown"], FEW_T3_UNK, tok, model)
    print("\n== 汇总（经济账）==")
    print(f"  {'任务':<26}{'覆盖率':>8}{'覆盖内准确率':>12}{'端到端':>8}{'安全升级':>10}{'危险路由':>10}")
    for k, v in res.items():
        print(f"  {k:<26}{v['cover'] * 100:>7.0f}%{v['acc_in'] * 100:>11.0f}%"
              f"{v['e2e'] * 100:>7.0f}%{v['safe'] * 100:>9.0f}%{v['danger'] * 100:>9.0f}%")


if __name__ == "__main__":
    main()
