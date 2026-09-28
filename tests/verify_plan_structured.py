#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""定向验证：用真实 _make_plan 跑通 GLM / LM Studio qwen 两种 Planner，
确认 response_format=json_object 接线后都产出合法（可 _normalize_plan）的契约。
不跑全量 harness，只验证 plan 生成这一环。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swe_agent.config as C
import swe_agent.models as M
import swe_agent.roles as R

TASK = "用 Python 实现康威生命游戏命令行程序（含网格初始化、演化规则、字符渲染、单元测试）。"


def try_planner(label):
    t = time.time()
    try:
        plan = R._make_plan(TASK)
    except Exception as e:
        print(f"\n### {label}: EXC {type(e).__name__}: {e}")
        return
    dt = time.time() - t
    if plan is None:
        print(f"\n### {label}: 返回 None（降级本地自规划） in {dt:.1f}s  ❌")
        return
    tasks = plan.get("tasks") or []
    mods = plan.get("modules") or []
    print(f"\n### {label}: OK in {dt:.1f}s ✅")
    print(f"    summary: {plan.get('summary','')[:60]}")
    print(f"    modules={len(mods)} tasks={len(tasks)} forbidden={len(plan.get('forbidden',[]))}")
    for i, tk in enumerate(tasks[:6], 1):
        print(f"      {i}. {tk.get('desc','')[:70]}")
    # 关键：每个 task 必须是 dict 且含 desc（_m_plan 的崩溃根源是 str/dict 不一致）
    bad = [i for i, tk in enumerate(tasks) if not isinstance(tk, dict) or not str(tk.get('desc','')).strip()]
    print(f"    task 形状检查: {'全部合格 ✅' if not bad else f'异常下标 {bad} ❌'}")


if __name__ == "__main__":
    # 1) GLM planner（默认）
    C.PLANNER_MODEL = "glm-4.7"
    spec = M.role_spec("planner")
    print(f"[setup] GLM planner spec provider={spec['provider'] if spec else None}")
    try_planner("GLM planner (json_object)")

    # 2) LM Studio qwen planner（切换 catalog 条目）
    C.PLANNER_MODEL = "qwen-lm-tjson"
    spec = M.role_spec("planner")
    print(f"\n[setup] LM Studio qwen planner spec provider={spec['provider'] if spec else None}, "
          f"model_name={spec.get('model_name') if spec else None}")
    try_planner("LM Studio qwen planner (json_object)")
