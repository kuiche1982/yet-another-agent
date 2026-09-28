import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

# 旧 dev-agent demo.py 已精简为 stub，相关子系统已迁移到独立模块：
#   - 技能系统 -> swe_agent.skills（reload_skills / get_skill / ALL_SKILLS / skills_prompt_section）
#   - MCP 子系统 -> swe_agent.mcp（MCPManager / MCP_CONFIG_PATH / mcp_prompt_section）
# 本测试直接对这两个真实模块做冒烟验证。


def main():
    from swe_agent import skills, mcp as mcp_mod

    print("=== 1) 加载 skills ===")
    skills.reload_skills()
    for s in skills.ALL_SKILLS:
        print(f"  - {s.name} [{s.source}/{s.context}]: {s.description[:30]}...")
    assert skills.get_skill("commit-helper") is not None, "磁盘技能未加载"
    assert len(skills.ALL_SKILLS) > 0, "没有任何技能"
    print("  skills_prompt_section 长度:", len(skills.skills_prompt_section()))

    print("\n=== 2) 连接 MCP demo server ===")
    mgr = mcp_mod.MCPManager()
    mgr.connect_all(mcp_mod.MCP_CONFIG_PATH)
    assert ("demo", "add") in mgr.tools_index, "MCP add 工具未发现"
    assert ("demo", "now") in mgr.tools_index, "MCP now 工具未发现"
    print("  mcp_prompt_section 长度:", len(mcp_mod.mcp_prompt_section()))

    print("\n=== 3) 调用 MCP 工具 ===")
    r1 = mgr.call("demo", "add", {"a": 2, "b": 3})
    print("  add(2,3) =>", r1)
    assert r1.strip() == "5", r1
    r2 = mgr.call("demo", "now", {})
    print("  now() =>", r2[:30], "...")
    assert len(r2) > 0
    # 错误路径
    r3 = mgr.call("demo", "nope", {})
    print("  nope =>", r3[:40])
    assert "mcp_error" in r3
    mgr.close_all()
    print("\n✅ skills + MCP 冒烟测试通过")


if __name__ == "__main__":
    main()


def test_main():
    main()
