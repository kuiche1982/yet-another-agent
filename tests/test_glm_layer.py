import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from pathlib import Path
import tempfile, shutil

# 旧 demo.* 已迁到独立模块（均带/不带下划线前缀，按当前代码为准）：
#   llm_glm._extract_json_object         -> swe_agent.llm_glm
#   roles._normalize_plan                -> swe_agent.roles
#   roles.drift_issues  (原 _drift_issues)
#   roles.plan_contract_section (原 _plan_contract_section)
#   config.WORKSPACE / GLOBAL_STATE 来自 state / SYSTEM_PROMPT / GLM_MODEL
#   supervisor.execute_action           -> swe_agent.supervisor
from swe_agent import roles, config as C, state
from swe_agent.llm_glm import _extract_json_object
from swe_agent.supervisor import execute_action


def main():
    # -*- coding: utf-8 -*-
    """隔离单测：GLM 顶层新增逻辑（JSON 提取 / 契约规范化 / 漂移检测 / 契约段渲染 / plan 拦截）。"""
    FAILED = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            FAILED.append(name)

    # ---- 1. _extract_json_object ----
    E = _extract_json_object
    check("json: 纯 JSON", E('{"a": 1}') == {"a": 1})
    check("json: ```json 围栏", E('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]})
    check("json: 前后杂文", E('好的，以下是契约：\n{"a": {"b": "含}花括号"}}\n以上。') == {"a": {"b": "含}花括号"}})
    check("json: 嵌套数组", E('{"tasks": ["t1", "t2"], "m": [{"path": "a.py"}]}')["m"][0]["path"] == "a.py")
    check("json: 空串→None", E("") is None)
    check("json: 无对象→None", E("没有 JSON") is None)
    check("json: 截断→None", E('{"a": 1') is None)
    check("json: 字符串内引号转义", E('{"s": "he said \\"hi {ok}\\""}')["s"] == 'he said "hi {ok}"')

    # ---- 2. _normalize_plan ----
    N = roles._normalize_plan
    ok_plan = {"summary": "s", "modules": [{"path": "a.py", "purpose": "p", "public": ["f()"]}],
               "interface": ["step()"], "forbidden": ["no GUI"], "tasks": ["t1", "t2"]}
    np = N(ok_plan)
    check("plan: 正常契约", np is not None and np["modules"][0]["path"] == "a.py" and len(np["tasks"]) == 2)
    check("plan: 缺 tasks→None", N({"modules": []}) is None)
    check("plan: tasks 空列表→None", N({"tasks": []}) is None)
    check("plan: modules 缺省可容错", N({"tasks": ["t"]})["modules"] == [])
    check("plan: 杂质字段被转 str", N({"tasks": ["t"], "modules": [{"path": 123}]})["modules"][0]["path"] == "123")

    # ---- 3. 漂移检测 + 契约段渲染（临时工作区） ----
    tmp = tempfile.mkdtemp()
    saved_ws = C.WORKSPACE
    saved_plan = state.GLOBAL_STATE.get("plan")
    saved_dr = roles._drift_reported.copy()
    try:
        C.WORKSPACE = Path(tmp)
        (Path(tmp) / "game.py").write_text("x=1", encoding="utf-8")
        (Path(tmp) / "tests").mkdir()
        (Path(tmp) / "tests" / "test_game.py").write_text("def t(): pass", encoding="utf-8")
        (Path(tmp) / "extra.py").write_text("y=2", encoding="utf-8")          # 漂移文件
        (Path(tmp) / "node_modules").mkdir()
        (Path(tmp) / "node_modules" / "pkg.js").write_text("z", encoding="utf-8")  # 重目录，忽略
        (Path(tmp) / "README.md").write_text("r", encoding="utf-8")           # 非代码后缀，忽略

        state.GLOBAL_STATE["plan"] = {"summary": "方案", "modules": [
            {"path": "game.py", "purpose": "核心", "public": ["step()"]}], "interface": ["step()"],
            "forbidden": ["no GUI"], "tasks": ["t1"]}
        roles._drift_reported = set()
        d = roles.drift_issues()
        check("drift: 检出契约外 extra.py", d == ["extra.py"])
        roles._drift_reported.update(d)
        check("drift: 已报告项不再重复", roles.drift_issues() == [])
        check("drift: 测试文件不算漂移", "tests/test_game.py" not in d)

        # 契约段渲染
        sec = roles.plan_contract_section()
        check("contract: 含标题", "顶层 Planner 契约" in sec)
        check("contract: 含文件白名单", "game.py" in sec and "文件白名单" in sec)
        check("contract: 含禁止事项", "no GUI" in sec)
        check("contract: 含填空式指引", "填空式实现" in sec)
        state.GLOBAL_STATE["plan"] = None
        check("contract: 无 plan→空串", roles.plan_contract_section() == "")

        # plan 动作拦截（注意先重新设置 plan）
        state.GLOBAL_STATE["plan"] = {"summary": "方案", "modules": [{"path": "game.py"}],
                                     "interface": [], "forbidden": [], "tasks": ["t1"]}
        res = execute_action({"action": "plan", "tasks": ["自作主张"]})
        check("plan 拦截: 契约模式下被拒", "禁止重新规划" in str(res))
        state.GLOBAL_STATE["plan"] = None
        res2 = execute_action({"action": "plan", "tasks": ["本地任务"]})
        check("plan 拦截: 无契约时放行", "已规划" in str(res2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        C.WORKSPACE = saved_ws
        state.GLOBAL_STATE["plan"] = saved_plan
        roles._drift_reported = saved_dr

    # ---- 4. 会话序列化兼容（plan 可 JSON 化） ----
    state.GLOBAL_STATE["plan"] = np
    try:
        import json
        json.dumps(state.GLOBAL_STATE, ensure_ascii=False, default=str)
        check("session: GLOBAL_STATE 含 plan 可序列化", True)
    except Exception as e:
        check(f"session: 序列化失败 {e}", False)
    finally:
        state.GLOBAL_STATE["plan"] = saved_plan

    # ---- 5. 配置面：GLM 远程 provider 模型名 ----
    check("config: GLM_MODEL 已设置", isinstance(C.GLM_MODEL, str) and bool(C.GLM_MODEL))

    print()
    if FAILED:
        print(f"❌ {len(FAILED)} 项失败：{FAILED}")
        assert False
    print("✅ 全部通过")


if __name__ == "__main__":
    main()


def test_main():
    main()
