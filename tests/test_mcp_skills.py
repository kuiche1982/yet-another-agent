#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""swe_agent 的 MCP + 技能（skills）后端集成测试（不依赖远程 LLM，可离线跑）。

验证：
- MCPManager 按 REPO_ROOT/mcp.json 连接 demo 服务器，并能真实调用其工具（add / now）；
- 技能系统从 skills/ 目录加载磁盘技能 + 内置 bundled 技能，run_skill 注入上下文 / 报错；
- supervisor 把 mcp_tool / list_mcp_resources / read_mcp_resource / skill 路由到真实后端
  （不再返回「插件层未启用」桩提示）。

运行：.venv/bin/python test_mcp_skills.py
"""

import sys
from swe_agent.registry import ToolRegistry, ActionContext
import swe_agent.supervisor as _sup  # 触发所有 @tool 注册（含真实 mcp_tool/skill 动作）
import swe_agent.mcp as mcp
import swe_agent.skills as skills


_fail = 0


def check(label, cond, detail=""):
    global _fail
    if cond:
        print(f"✅ {label}")
    else:
        print(f"❌ {label}  {detail}")
        _fail += 1


def main():
    # ---- skills：磁盘 + bundled 加载 ----
    names = {s.name for s in skills.ALL_SKILLS}
    check("技能加载：磁盘技能 commit-helper 存在", "commit-helper" in names, str(names))
    check("技能加载：内置 bundled review 存在", "review" in names)
    check("技能加载：内置 bundled gen-tests 存在", "gen-tests" in names)

    # run_skill 不存在 -> 报错（含可用清单）
    err = skills.run_skill("nope", "", [])
    check("run_skill 未知技能 -> skill_error", err.startswith("skill_error: 未找到技能"), repr(err))

    # run_skill inline 把指令注入 messages
    msgs = []
    ok = skills.run_skill("commit-helper", "", msgs)
    check("run_skill(commit-helper) 注入上下文", len(msgs) == 1 and "commit-helper" in msgs[0]["content"],
          f"msgs={len(msgs)}")
    check("run_skill 返回 skill_loaded", ok.startswith("skill_loaded:"), repr(ok))

    # ---- MCP：连接 demo 服务器并真实调用 ----
    mgr = mcp.get_mcp()
    check("MCP 已连接 demo 服务器", "demo" in mgr.clients, str(list(mgr.clients)))
    check("MCP 工具索引含 demo:add", ("demo", "add") in mgr.tools_index)
    check("MCP 工具索引含 demo:now", ("demo", "now") in mgr.tools_index)

    r_add = mgr.call("demo", "add", {"a": 2, "b": 3})
    check("MCP demo.add(2,3) == '5'", r_add == "5", repr(r_add))

    r_now = mgr.call("demo", "now", {})
    # now() 返回 ISO 时间字符串，长度 > 10 且不含 mcp_error
    check("MCP demo.now() 返回非错误字符串", isinstance(r_now, str) and not r_now.startswith("mcp_error") and len(r_now) > 8,
          repr(r_now))

    # 未连接服务器 / 不存在工具 -> 受控报错
    err_srv = mgr.call("ghost", "add", {})
    check("MCP 未连接服务器 -> mcp_error", err_srv.startswith("mcp_error: 未连接的服务器"), repr(err_srv))
    err_tool = mgr.call("demo", "ghosttool", {})
    check("MCP 不存在工具 -> mcp_error", err_tool.startswith("mcp_error: 服务器 'demo' 无工具"), repr(err_tool))

    # list_mcp_resources 不应崩（demo 服务器可能不支持 resources，返回提示而非异常）
    lr = mcp.mcp_list_resources()
    check("list_mcp_resources 返回字符串", isinstance(lr, str), repr(lr))

    # ---- supervisor 路由：dispatch 走真实后端，而非桩 ----
    r_disp = ToolRegistry.dispatch(
        {"action": "mcp_tool", "server": "demo", "tool": "add", "arguments": {"a": 7, "b": 8}},
        ActionContext(messages=[]),
    )
    check("dispatch mcp_tool add(7,8) == '15'", r_disp == "15", repr(r_disp))

    msgs2 = []
    r_skill = ToolRegistry.dispatch({"action": "skill", "name": "commit-helper"}, ActionContext(messages=msgs2))
    check("dispatch skill 注入上下文", len(msgs2) == 1, f"msgs={len(msgs2)}")
    check("dispatch skill 非桩提示", "插件扩展层" not in r_skill, repr(r_skill))

    # 未知动作仍回退（与重构一致）
    unk = ToolRegistry.dispatch({"action": "mcp_tool", "server": "demo", "tool": "add"}, ActionContext(messages=[]))
    check("mcp_tool 缺 arguments 仍优雅返回(非崩)", isinstance(unk, str), repr(unk))

    # 清理：关闭 MCP 子进程
    mgr.close_all()

    print()
    if _fail == 0:
        print("✅ 全部通过（MCP + skills 真实后端集成）")
        sys.exit(0)
    else:
        print(f"❌ {_fail} 项失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
