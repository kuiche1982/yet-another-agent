#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比报告生成器（compare_report.py）
=================================
读取 compare_results.jsonl + 各格日志（compare_<model>_<task>.log），
生成一份 2×2 对比报告 compare_report.md。

brain 恒为 glm-4.7（planner + reviewer），双手为 ling-3.0-tiny / qwen-2.5-coder-7b。
"""
import json
import os
import re
from pathlib import Path
from swe_agent.log import logger

REPO = "~/kuiwork/workdir2/litertlm"
RES = os.path.join(REPO, "scripts", "compare_results.jsonl")
MODELS = ["ling", "qwen"]
TASKS = ["new", "bugfix"]


def load_results():
    out = {}
    if not os.path.exists(RES):
        return out
    with open(RES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[(d["model"], d["task"])] = d
    return out


def tail_log(model, task, n=25):
    p = os.path.join(REPO, "scripts", f"compare_{model}_{task}.log")
    if not os.path.exists(p):
        return ""
    lines = Path(p).read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def extract_highlights(model, task, r):
    """从日志里抽关键事实：planner 契约、pytest/hidden 结果、最终 verdict。"""
    log = tail_log(model, task, n=400)
    hi = {}
    m = re.search(r"pytest:\s*(\d+)\s*passed\s*/\s*(\d+)\s*failed", log)
    if m:
        hi["pytest"] = f"{m.group(1)} passed / {m.group(2)} failed"
    m = re.search(r"隐藏验收:\s*(\d+)\s*passed\s*/\s*(\d+)\s*failed", log)
    if m:
        hi["hidden"] = f"{m.group(1)} passed / {m.group(2)} failed"
    m = re.search(r"\[planner\].*?隐藏测试集\s*(\d+)\s*条", log)
    if m:
        hi["hidden_count"] = m.group(1)
    m = re.search(r"Agent 运行结束", log)
    if m:
        hi["ran_to_end"] = True
    return hi


def main():
    res = load_results()
    lines = []
    lines.append("# Ling vs Qwen 对比报告（brain = glm-4.5-flash）\n")
    lines.append(f"> 生成时间：{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**受控变量**：大脑恒为 glm-4.5-flash（planner + reviewer，因 glm-4.7 系列当前对长生成会超时，详见文末说明）；"
                 "双手为 `ling-3.0-tiny`（本地 rapid-mlx）与 `qwen-2.5-coder-7b`（本地 rapid-mlx）。\n")
    lines.append("**任务**：`new` = 从零构建康威生命游戏 CLI；"
                 "`bugfix` = 接手含 bug 的实现并修复（隐藏验收闸门 `test_acceptance_user.py` 戳穿 step 坐标写反）。\n")

    # ---- 2x2 表格 ----
    lines.append("\n## 结果总览（2×2）\n")
    lines.append("| 模型 | 任务 | 大脑 | 结论 | 轮数 | Executor调用 | GLM调用 | 耗时(s) |")
    lines.append("|------|------|------|------|------|--------------|----------|---------|")
    for model in MODELS:
        for task in TASKS:
            r = res.get((model, task))
            if not r:
                lines.append(f"| {model} | {task} | - | （无记录） | - | - | - | - |")
                continue
            brain = r.get("brain_used", r.get("brain", "glm-4.7"))
            if brain != "glm-4.7":
                brain_cell = f"⚠️ {brain}"
            else:
                brain_cell = brain
            lines.append(
                f"| {model} | {task} | {brain_cell} | {r['verdict']} | {r['rounds']} | "
                f"{r['executor_calls']} | {r['glm_calls']} | {r['duration_s']} |"
            )

    # ---- 逐格明细 ----
    lines.append("\n## 逐格明细\n")
    for model in MODELS:
        for task in TASKS:
            r = res.get((model, task))
            lines.append(f"\n### {model} × {task}\n")
            if not r:
                lines.append("_（该格无结果记录，可能进程异常）_\n")
                continue
            hi = extract_highlights(model, task, r)
            brain = r.get("brain_used", r.get("brain", "glm-4.7"))
            if brain != "glm-4.7":
                lines.append(f"- **⚠️ 大脑降级**：本格 planner 因 glm 调用超时静默降级为「本地自规划」，"
                             f"结果**不计入** glm-4.7 大脑对比，需重跑。")
            lines.append(f"- **结论**：`{r['verdict']}`")
            lines.append(f"- 轮数 {r['rounds']} ｜ Executor 调用 {r['executor_calls']} ｜ "
                         f"GLM 调用 {r['glm_calls']} ｜ 耗时 {r['duration_s']}s")
            if hi.get("pytest"):
                lines.append(f"- 自测 pytest：**{hi['pytest']}**")
            if hi.get("hidden"):
                lines.append(f"- 隐藏验收：**{hi['hidden']}**")
            lines.append("")

    lines.append("\n---\n_由 compare_report.py 自动生成。原始明细见 compare_results.jsonl 与各 compare_<model>_<task>.log。_\n")
    lines.append("\n## 关于「大脑」的说明\n")
    lines.append("- 原始需求为 glm-4.7 作为 planner+reviewer 大脑。实测 **glm-4.7（含 glm-4.7-flash）当前对「生成完整执行契约」这类长生成会卡死**（read timeout=600s 仍零输出；"
                 "而同账号下简单提问 0.7s 返回、中等生成 30s 出 1500 token，说明 API 可达，仅 glm-4.7 家族长生成被限流/挂起）。\n")
    lines.append("- 为保证对比可完成且 2×2 大脑一致，改用当前可用的 **glm-4.5-flash**（实测 49.6s 出有效契约：2 模块 / 3 任务 / 2 禁止项），planner+reviewer 均为 glm-4.5-flash。\n")
    lines.append("- 若后续 glm-4.7 恢复，可对 4 格用 glm-4.7 重跑以获得纯 glm-4.7 大脑的对比。\n")

    out_path = os.path.join(REPO, "reports", "compare_report.md")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info('%s', f'[report] 已生成 {out_path}')
    logger.info('%s', '\n'.join(lines))


if __name__ == "__main__":
    main()
