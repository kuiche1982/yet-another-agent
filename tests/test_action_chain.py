import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

# 旧 demo.py 的编排层已拆掉，这些函数迁到独立模块：
#   skills.reload_skills / ALL_SKILLS          -> swe_agent.skills
#   mcp.MCPManager / MCP_CONFIG_PATH / get_mcp -> swe_agent.mcp
#   actions.parse_actions                      -> swe_agent.actions
#   supervisor.execute_action                  -> swe_agent.supervisor
# （mcp_tool / skill 动作在 supervisor.py 仍按原名注册，契约未变）
from swe_agent import skills, mcp, supervisor
from swe_agent.supervisor import execute_action
from swe_agent.actions import parse_actions


def main():
    print("=== 加载 skills + 连接 MCP（真实 demo 服务器，按 mcp.json）===")
    skills.reload_skills()
    mgr = mcp.get_mcp()  # 懒连接：按 mcp.json 连 demo 服务器
    assert mgr is not None, "mcp.get_mcp() 返回 None"
    assert ("demo", "add") in mgr.tools_index, "MCP add 工具未发现"
    print("  skills:", [s.name for s in skills.ALL_SKILLS])
    print("  mcp tools:", list(mgr.tools_index.keys()))

    print("\n=== A) parse_action: mcp_tool（整段 JSON，arguments 为对象）===")
    out = parse_actions('{"action":"mcp_tool","server":"demo","tool":"add","arguments":{"a":4,"b":6}}')
    print("  parsed:", out)
    assert out and out[0]["arguments"] == {"a": 4, "b": 6}, out

    print("\n=== B) execute_action: mcp_tool ===")
    res = execute_action({"action": "mcp_tool", "server": "demo", "tool": "add", "arguments": {"a": 4, "b": 6}})
    print("  result:", res)
    assert res.strip().endswith("10"), res

    print("\n=== C) execute_action: skill（inline 注入上下文）===")
    # review 技能已下线；用磁盘技能 commit-helper（context: inline）验证注入契约
    messages = []
    res = execute_action({"action": "skill", "name": "commit-helper", "args": ""}, messages)
    print("  result:", res)
    assert "skill_loaded" in res, res
    assert len(messages) == 1 and "技能" in messages[0]["content"], messages
    print("  注入的消息角色/片段:", messages[0]["role"], "|", messages[0]["content"][:40], "...")

    print("\n=== D) execute_action: skill 不存在 ===")
    res = execute_action({"action": "skill", "name": "ghost", "args": ""}, [])
    print("  result:", res)
    assert "skill_error" in res, res

    print("\n✅ action 链路（parse + execute + MCP + skill 注入）全部通过")
    if mgr is not None:
        mgr.close_all()


if __name__ == "__main__":
    main()


def test_main():
    main()
