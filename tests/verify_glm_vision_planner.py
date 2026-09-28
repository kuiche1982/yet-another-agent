#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # -*- coding: utf-8 -*-
    """model-free / tool-free 验证：GLM 顶层 Planner 的「契约装配管线」。

    原版端到端真打 zhipu API（glm-4.1v-thinking-flashx / glm-4.6v），需要 GLM_API_TOKEN
    且依赖网络，离线/CI 必红。改为离线验证：桩掉 roles_config.make_agent，让 planner 的
    run() 直接回灌合法契约样本，确认
    role_spec -> _planner_response_format -> roles._make_plan -> _normalize_plan
    的装配链路正确产出合法契约，不触 LLM / 不触网络 / 不读 token。

    注意桩点为何选 make_agent（而非 models.chat_text_escalating）：构造真实 Agent 会触发
    ModelManager.__init__ 对整个工作区做 AST 解析建知识库（contextmgr 链），离线单测会因此
    卡死数十秒；桩掉 make_agent 返回轻量假 Agent 既离线又不触该重型链路，验证意图等价
    （LLM 输出 = canned_plan，装配/抽取/归一化逻辑仍走真实代码）。
    """
    import json
    from unittest import mock

    from swe_agent import config as C
    from swe_agent import models as M
    from swe_agent import roles
    from swe_agent import roles_config as RC

    TASK = "用 Python 写一个函数 count_neighbors(grid, r, c)，统计二维网格中某格周围 8 邻域的存活邻居数。"

    # 合法 Planner 契约样本（覆盖 modules / interface / forbidden / tasks 字段）。
    canned_plan = json.dumps({
        "summary": "统计二维网格 8 邻域存活邻居数",
        "modules": [{"path": "grid.py", "purpose": "网格核心逻辑", "public": ["count_neighbors()"]}],
        "interface": ["count_neighbors(grid, r, c)"],
        "forbidden": ["禁止修改输入 grid"],
        "tasks": [
            {"step": "实现 count_neighbors", "deliverables": ["grid.py"], "status": "pending"},
            {"step": "写单测覆盖边界", "deliverables": ["test_grid.py"], "status": "pending"},
        ],
    })

    def fake_make_agent(role=None, loop=None, model_override=None, ctx=None, before_step=None):
        # 轻量假 Agent：run() 直接回灌样本契约，绕开真实 GLM 调用与 ModelManager 知识库摄入。
        # 仅实现 _make_plan 实际用到的方法（run），避免任何重型构造。
        class _FakePlannerAgent:
            def run(self, messages):
                return canned_plan
        return _FakePlannerAgent()

    saved_planner = C.PLANNER_MODEL
    try:
        # 用注册过的 zhipu 模型，确保 role_spec 不抛 ModelUnknown（glm-4.7 在 MODELS 中）。
        C.PLANNER_MODEL = "glm-4.7"
        spec = M.role_spec("planner")
        assert spec is not None, "planner 模型应在 MODELS 注册"
        assert spec["provider"] in ("zhipu", "lmstudio"), f"provider 异常: {spec.get('provider')}"
        assert roles._planner_response_format() is not None, "planner 应配置 response_format"

        # 桩掉 make_agent 后驱动真实 _make_plan 管线（含 _extract_json_object + _normalize_plan）。
        # 假 Agent 的 run() 等价回灌 canned_plan，装配/抽取/归一化逻辑仍走真实代码。
        with mock.patch.object(RC, "make_agent", fake_make_agent):
            plan = roles._make_plan(TASK)
        assert plan is not None, "_make_plan 在桩模型下应产出契约"
        assert len(plan["tasks"]) >= 1, "契约应含至少 1 个任务"
        assert plan["modules"][0]["path"] == "grid.py", "契约 modules 装配正确"
        print(f"✓ model-free 装配通过：{len(plan['modules'])} 模块 / {len(plan['tasks'])} 任务")
        print(f"  role_spec -> provider={spec['provider']}, fc={spec['fc'].value}")
    finally:
        C.PLANNER_MODEL = saved_planner
    print("完成（离线，未触 GLM token / 网络）。")


if __name__ == "__main__":
    main()


def test_main():
    main()
