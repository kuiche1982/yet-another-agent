#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # -*- coding: utf-8 -*-
    """
    离线单元测试：验证模型/角色解耦（#85）的核心逻辑，不触发任何网络/模型调用。

    覆盖：
    - 默认角色→模型→provider/fc 解析（结构不变式，不绑死具体模型 id）
    - _resolve_executor_arg 别名/直接 id 映射（local 已废弃，指向真实 lmstudio 执行器）
    - 运行时改 config.* 能被 live 读取（--executor 等价行为）
    - 降级路径：角色模型为空 → run_planner 返回 False（无网络）
    - 单一能力 NATIVE_TOOLS：所有角色 fc 归一，不再有 TEXT_JSON/JSON_MODE 分支
    """
    import sys
    sys.path.insert(0, "~/kuiwork/workdir2/litertlm")

    import swe_agent.config as C
    from swe_agent import models as M
    from swe_agent import roles
    from swe_agent.supervisor import _resolve_executor_arg

    fails = []

    def check(name, cond):
        print(("✅" if cond else "❌"), name)
        if not cond:
            fails.append(name)

    KNOWN_PROVIDERS = {"zhipu", "lmstudio"}

    # 1) 默认解析：结构不变式（不绑死具体模型 id，因 env/默认值可能随环境变化）
    for role in ("planner", "analyzer", "executor"):
        mid = M.role_model_id(role)
        prov = M.role_provider(role)
        fc = M.role_fc(role)
        check(f"默认 {role} 解析到已登记模型 id", mid in M.MODELS)
        check(f"默认 {role} provider ∈ {{zhipu,lmstudio}}", prov in KNOWN_PROVIDERS)
        check(f"默认 {role} fc == NATIVE_TOOLS", fc == M.FCCapability.NATIVE_TOOLS)
    check("FCCapability 仅剩 NATIVE_TOOLS（无 TEXT_JSON/JSON_MODE）",
          [m for m in dir(M.FCCapability) if not m.startswith("_") and m.isupper()] == ["NATIVE_TOOLS"])

    # 2) _resolve_executor_arg
    check("--executor glm → glm-4.7", _resolve_executor_arg("glm") == "glm-4.7")
    check("--executor lmstudio → liquid/lfm2.5-1.2b", _resolve_executor_arg("lmstudio") == "liquid/lfm2.5-1.2b")
    # local/plaintext 已废弃，别名指向真实 lmstudio 执行器（不再产生孤儿 id）
    check("--executor local → qwen2.5.1-coder-7b-instruct",
          _resolve_executor_arg("local") == "qwen2.5.1-coder-7b-instruct")
    check("--executor 直接传 catalog id 透传", _resolve_executor_arg("qwen2.5.1-coder-7b-instruct") == "qwen2.5.1-coder-7b-instruct")
    check("--executor 未知回退 qwen2.5.1-coder-7b-instruct", _resolve_executor_arg("not-a-model") == "qwen2.5.1-coder-7b-instruct")
    # 解析结果必须都是已登记目录 id（不再出现孤儿 qwen-2.5-coder-7b）
    check("所有别名解析结果都在 MODELS 目录",
          all(_resolve_executor_arg(v) in M.MODELS for v in ("glm", "lmstudio", "local", "not-a-model")))

    # 3) 运行时 live 覆盖（等价于 CLI 改 config）
    saved_exec = C.EXECUTOR_MODEL
    C.EXECUTOR_MODEL = "glm-4.7"
    check("运行时改 EXECUTOR_MODEL → provider zhipu", M.role_provider("executor") == "zhipu")
    C.EXECUTOR_MODEL = "liquid/lfm2.5-1.2b"
    check("运行时改 EXECUTOR_MODEL → provider lmstudio", M.role_provider("executor") == "lmstudio")
    C.EXECUTOR_MODEL = saved_exec

    # 4) 降级路径（无网络）：角色模型为空
    saved_p = C.PLANNER_MODEL
    C.PLANNER_MODEL = ""
    check("planner 为空 → run_planner 返回 False（降级）", roles.run_planner("做个任务") is False)
    C.PLANNER_MODEL = saved_p

    # 5) describe_config 可调用且不抛
    try:
        txt = M.describe_config()
        check("describe_config 可调用", "planner" in txt and "executor" in txt)
    except Exception as e:
        check(f"describe_config 不抛异常（{e}）", False)

    # 6) #82 结构化 plan（Planner 显式声明 → 校验直接读取，正则仅兜底）
    import tempfile
    from pathlib import Path
    from swe_agent import state as S
    from swe_agent.roles import _normalize_task, _normalize_plan, _apply_plan

    # 6a) _normalize_task：对象 / 纯字符串两种形态
    nt = _normalize_task({"step": "实现 init", "deliverables": ["game.py", "test_game.py"]})
    check("#82 _normalize_task(dict) → desc+deliverables",
          nt["desc"] == "实现 init" and nt["deliverables"] == ["game.py", "test_game.py"] and nt["status"] == "pending")
    nt2 = _normalize_task("只做调试")
    check("#82 _normalize_task(str) → 空 deliverables",
          nt2["desc"] == "只做调试" and nt2["deliverables"] == [])

    # 6b) _normalize_plan：tasks 为对象列表时保留交付物
    np_ = _normalize_plan({
        "summary": "s",
        "modules": [{"path": "g.py", "purpose": "p", "public": ["f"]}],
        "tasks": [{"step": "a", "deliverables": ["g.py"]}, "裸字符串步"],
    })
    check("#82 _normalize_plan 保留结构化交付物",
          np_["tasks"][0]["deliverables"] == ["g.py"] and np_["tasks"][1]["deliverables"] == [])

    # 6c) _apply_plan：deliverables 透传到 GLOBAL_STATE['tasks']
    S.reset_state()
    _apply_plan({
        "summary": "s", "modules": [{"path": "g.py", "purpose": "p", "public": []}],
        "interface": [], "forbidden": [], "tasks": [{"desc": "a", "deliverables": ["g.py"], "status": "pending"}],
    })
    check("#82 _apply_plan 透传 deliverables", S.GLOBAL_STATE["tasks"][0]["deliverables"] == ["g.py"])

    # 注：旧的「交付物落盘校验子系统」（state._verify_task_deliverables）已于
    # 完成标准重构（单杠校验）时整体移除，相关断言在此测试中被有意删除。

    print("\n" + M.describe_config())

    if fails:
        print(f"\n❌ 失败 {len(fails)} 项：{fails}")
        assert False
    print("\n✅ 全部通过")


if __name__ == "__main__":
    main()


def test_main():
    main()
