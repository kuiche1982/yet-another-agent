#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""swe_agent 的 Claude 插件子系统集成测试（不依赖远程 LLM，可离线跑）。

验证 plugins.load_plugins 把 ~/.claude/plugins 格式的插件正确汇入三层真实后端：
- skills/commands -> swe_agent.skills.PLUGIN_SKILLS（经 reload_skills 进入 ALL_SKILLS，带 插件名: 前缀）
- agents/*.md     -> supervisor.SUBAGENT_PROMPTS / SUBAGENT_ALLOWED（子智能体）
- .mcp.json       -> swe_agent.mcp.PLUGIN_MCP_SERVERS（仅 enable_mcp=True 时）
- plugins_prompt_section() 把插件扩展写进系统提示词

运行：.venv/bin/python test_plugins.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import swe_agent.supervisor as _sup  # 触发 @tool 注册 + 子智能体注册表
import swe_agent.plugins as plugins
import swe_agent.skills as skills
import swe_agent.mcp as mcp


_fail = 0


def check(label, cond, detail=""):
    global _fail
    if cond:
        print(f"✅ {label}")
    else:
        print(f"❌ {label}  {detail}")
        _fail += 1


def _write(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _make_plugin(install: Path):
    """构造一个最小但完整的 claude 插件目录。"""
    # skills/my-skill/SKILL.md
    _write(install / "skills" / "my-skill" / "SKILL.md", """---
name: my-skill
description: 插件自带技能（用于验证插件技能加载）
when_to_use: 当想调用插件自带技能时
allowed-tools: read_file, shell
---
# 插件技能 my-skill 指令
请 read_file agent_sandbox/README.md 并打印其内容。
""")
    # commands/my-command.md（slash command 语义，模型可自主调用）
    _write(install / "commands" / "my-command.md", """---
name: my-command
description: 插件命令，验证 commands/*.md 被注册为技能
allowed-tools: write_file
---
# 插件命令 my-command 指令
请 write_file agent_sandbox/cmd_out.txt 写入 "ok"。
""")
    # agents/my-agent.md（子智能体）
    _write(install / "agents" / "my-agent.md", """---
name: my-agent
description: 插件子智能体，验证 agents/*.md 注册
tools: read, write
---
你是一个只读/写的插件子智能体。
""")
    # .mcp.json（裸格式，含一个不存在的 command 以便离线测试不真正连接）
    _write(install / ".mcp.json", '{"demo2": {"command": "/nonexistent/mcp-stub", "args": []}}')


def main():
    tmp = Path(tempfile.mkdtemp(prefix="plugins_test_"))
    root = tmp / "plugins"          # 模拟 ~/.claude/plugins
    install = root / "myplugin@1.0.0"
    _make_plugin(install)

    # installed_plugins.json 指向真实安装路径
    manifest = {
        "plugins": {
            "myplugin@1.0.0": [{"installPath": str(install)}],
        }
    }
    _write(root / "installed_plugins.json", __import__("json").dumps(manifest))

    # 记录测试前状态，便于清理
    prev_sub_prompts = set(_sup.SUBAGENT_PROMPTS.keys())
    prev_sub_allowed = set(_sup.SUBAGENT_ALLOWED.keys())

    try:
        # ---- 主路径：enable_mcp=False（默认，避免未授权外连）----
        state = plugins.load_plugins(root, enable_mcp=False)
        check("load_plugins 返回统计", isinstance(state, dict) and state.get("plugins") == ["myplugin"], str(state))
        check("统计：1 个插件技能", state.get("skills") == 1, str(state))
        check("统计：1 个插件命令", state.get("commands") == 1, str(state))
        check("统计：1 个子智能体", len(state.get("agents", {})) == 1, str(state))
        check("统计：MCP 未启用（空）", state.get("mcp") == [], str(state))

        names = {s.name for s in skills.ALL_SKILLS}
        check("插件技能 myplugin:my-skill 进入 ALL_SKILLS",
              "myplugin:my-skill" in names, str(names))
        check("插件命令 myplugin:my-command 进入 ALL_SKILLS",
              "myplugin:my-command" in names, str(names))
        check("磁盘技能 commit-helper 仍在", "commit-helper" in names)

        # run_skill 执行插件技能（inline 注入上下文）
        msgs = []
        r = skills.run_skill("myplugin:my-skill", "", msgs)
        check("run_skill(插件技能) 注入上下文", len(msgs) == 1 and "my-skill" in msgs[0]["content"],
              f"msgs={len(msgs)}")
        check("run_skill 返回 skill_loaded", r.startswith("skill_loaded:"), repr(r))

        # 子智能体注册
        check("插件子智能体 my-agent 注册到 SUBAGENT_PROMPTS",
              "my-agent" in _sup.SUBAGENT_PROMPTS, str(list(_sup.SUBAGENT_PROMPTS)))
        check("插件子智能体 my-agent 注册到 SUBAGENT_ALLOWED",
              _sup.SUBAGENT_ALLOWED.get("my-agent") == ["read_file", "write_file", "edit_file",
                                                         "shell", "grep", "glob", "web_fetch",
                                                         "task_output", "task_stop", "sleep", "report"],
              str(_sup.SUBAGENT_ALLOWED.get("my-agent")))

        # 插件提示词段（技能/命令/工具）；3-loop 重构后子智能体目录已移出系统提示，
        # 改由 run_subagent 程序派发（见 supervisor.run_subagent），注册仍进 SUBAGENT_PROMPTS（见上）。
        sec = plugins.plugins_prompt_section()
        check("plugins_prompt_section 非空（技能/命令/工具）", "插件扩展" in sec, repr(sec))
        # 工具清单段（原 agents_prompt_section 已重命名 + 改语义为工具列表）
        tool_sec = _sup._tools_prompt_section()
        check("_tools_prompt_section 返回工具清单段",
              isinstance(tool_sec, str) and "可用工具" in tool_sec, repr(tool_sec[:120]))

        # enable_mcp=True 时写入 MCP 后端（不触发连接）
        mcp.PLUGIN_MCP_SERVERS = {}  # 先清空
        state2 = plugins.load_plugins(root, enable_mcp=True)
        check("enable_mcp=True 写入 PLUGIN_MCP_SERVERS",
              "demo2" in mcp.PLUGIN_MCP_SERVERS, str(list(mcp.PLUGIN_MCP_SERVERS)))
        check("enable_mcp=True 统计含 demo2", "demo2" in state2.get("mcp", []), str(state2.get("mcp")))

        # ---- 边界：清单不存在 -> 跳过且返回空 ----
        empty = tmp / "no_plugins"
        empty.mkdir()
        st_empty = plugins.load_plugins(empty, enable_mcp=False)
        check("清单缺失时返回空插件列表", st_empty.get("plugins") == [], str(st_empty))

        # ---- 边界：清单指向不存在的安装路径 -> 跳过该插件 ----
        bad = tmp / "bad_root"
        bad_install = bad / "ghost@1.0.0"
        bad_install.mkdir(parents=True)
        _write(bad / "installed_plugins.json",
               __import__("json").dumps({"plugins": {"ghost@1.0.0": [{"installPath": "/no/such/path"}]}}))
        st_bad = plugins.load_plugins(bad, enable_mcp=False)
        check("安装路径不存在时跳过该插件", st_bad.get("plugins") == [], str(st_bad))
    finally:
        # 清理：恢复全局状态，避免污染其它测试
        skills.PLUGIN_SKILLS = []
        skills.reload_skills()
        mcp.PLUGIN_MCP_SERVERS = {}
        for k in set(_sup.SUBAGENT_PROMPTS) - prev_sub_prompts:
            _sup.SUBAGENT_PROMPTS.pop(k, None)
        for k in set(_sup.SUBAGENT_ALLOWED) - prev_sub_allowed:
            _sup.SUBAGENT_ALLOWED.pop(k, None)
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if _fail == 0:
        print("✅ 全部通过（Claude 插件子系统集成）")
        sys.exit(0)
    else:
        print(f"❌ {_fail} 项失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
