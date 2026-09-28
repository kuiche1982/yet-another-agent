#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""swe_agent/plugins.py —— Claude 插件支持（对齐 claude-code plugin 机制）。

忠实移植自最初 demo.py 的插件子系统（~/.claude/plugins/installed_plugins.json）。
插件结构：.claude-plugin/plugin.json（元数据）+ skills/<n>/SKILL.md + commands/*.md
          + agents/*.md + .mcp.json（裸 / mcpServers 两种格式均支持）+ hooks（暂不支持）。

加载结果分四路汇入「插件扩展层」真实后端：
- 插件 skills/commands  -> swe_agent.skills.PLUGIN_SKILLS（经 reload_skills 进入 ALL_SKILLS）
- 插件 .mcp.json        -> swe_agent.mcp.PLUGIN_MCP_SERVERS（仅 enable_mcp=True 时连接）
- 插件 .lsp.json        -> swe_agent.lsp.PLUGIN_LSP_SERVERS（按 languageId 索引，写文件时回灌诊断）
- 插件 agents/*.md      -> supervisor.SUBAGENT_PROMPTS / SUBAGENT_ALLOWED（子智能体）

注意：为避免循环依赖，supervisor / mcp / lsp 仅在 load_plugins() 内部懒导入。
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .skills import Skill, load_skills_from_dir, _parse_frontmatter, reload_skills
from .mcp import normalize_mcp_config
from .registry import ToolRegistry
from . import config as C
from swe_agent.log import logger

PLUGINS_ROOT = Path.home() / ".claude" / "plugins"
PLUGIN_SKILLS: List[Skill] = []
PLUGIN_MCP_SERVERS: Dict[str, Any] = {}
PLUGIN_TOOL_NAMES: set = set()   # 由插件注册的工具名集合（用于重载/移除时清理）
PLUGIN_STATE: Dict[str, Any] = {
    "plugins": [], "skills": 0, "commands": 0,
    "agents": {}, "tools": [], "mcp": [], "lsp": [],
}

# claude 工具名 -> demo 动作 的映射（用于插件 agents 的 tools frontmatter）
_CLAUDE_TOOL_MAP = {
    "read": "read_file", "write": "write_file", "edit": "edit_file",
    "bash": "shell", "grep": "grep", "glob": "glob",
    "webfetch": "web_fetch", "websearch": "web_search",
    "taskoutput": "task_output", "taskstop": "task_stop",
    "sleep": "sleep", "todowrite": "todo_write",
}
_SUB_FULL_ACTIONS = ["read_file", "write_file", "edit_file", "shell", "grep", "glob",
                     "web_fetch", "task_output", "task_stop", "sleep", "report"]
_SUB_READONLY_ACTIONS = ["read_file", "grep", "glob", "web_fetch", "report"]


def _map_claude_tools(tools_str: str) -> List[str]:
    """把 frontmatter 里的 claude 工具名列表（逗号分隔或 YAML 列表残留）映射成 demo 动作。"""
    raw = (tools_str or "").strip().strip("[]")
    names = [t.strip().strip("-").strip() for t in raw.split(",")]
    out = []
    for n in names:
        if not n:
            continue
        mapped = _CLAUDE_TOOL_MAP.get(n.lower())
        if mapped:
            out.append(mapped)
    return out


def _register_plugin_agents(agents_dir: Path) -> Dict[str, str]:
    """把插件 agents/*.md 注册为可用子智能体（对齐 claude-code 的 plugin agent 机制）。

    写入 supervisor.SUBAGENT_PROMPTS / SUBAGENT_ALLOWED。
    """
    from . import supervisor  # 懒导入，避免循环依赖
    descs: Dict[str, str] = {}
    if not agents_dir.exists():
        return descs
    for md in sorted(agents_dir.glob("*.md")):
        try:
            raw = md.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, body = _parse_frontmatter(raw)
        if not body.strip():
            continue
        name = fm.get("name") or md.stem
        desc = fm.get("description", f"插件子智能体 {name}")
        mapped = _map_claude_tools(fm.get("tools", ""))
        if mapped:
            # 只读工具集合则限制为只读，否则给完整动作
            if set(mapped) <= {"read_file", "grep", "glob", "web_fetch", "web_search"}:
                allowed = _SUB_READONLY_ACTIONS
            else:
                allowed = [a for a in _SUB_FULL_ACTIONS]
        else:
            allowed = list(_SUB_FULL_ACTIONS)
        proto = (
            "\n\n=== 执行协议 ===\n"
            "你必须**每次只输出一个 JSON 动作对象**，不要输出任何多余文字、说明或 markdown 代码块。\n"
            "所有路径都使用相对路径（相对 {WORKSPACE}），严禁绝对路径。\n"
            f"可用动作：{' / '.join(allowed)}。\n"
            '完成后必须用 {"action":"report","content":"结论"} 返回报告（不能为空）。'
        )
        supervisor.SUBAGENT_PROMPTS[name] = body.strip() + proto.replace("{WORKSPACE}", "当前工作目录（即你的 cwd，直接写相对路径）")
        supervisor.SUBAGENT_ALLOWED[name] = allowed
        descs[name] = desc
    return descs


def _load_plugin_commands(commands_dir: Path, plugin_name: str) -> List[Skill]:
    """把插件 commands/*.md 注册为技能（名称带 插件名: 前缀，对齐 slash command 机制）。"""
    out: List[Skill] = []
    if not commands_dir.exists():
        return out
    for md in sorted(commands_dir.glob("*.md")):
        try:
            raw = md.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, body = _parse_frontmatter(raw)
        if not body.strip():
            continue
        # disable-model-invocation: true 表示仅用户可触发，模型不可自主调用
        if str(fm.get("disable-model-invocation", "")).strip().lower() in ("true", "yes", "1"):
            continue
        name = fm.get("name") or md.stem
        allowed = _map_claude_tools(fm.get("allowed-tools", ""))
        out.append(Skill(
            name=f"{plugin_name}:{name}",
            description=f"[插件命令] {fm.get('description', name)}",
            content=body.strip(),
            allowed_tools=allowed,
            context="inline",
            source="plugin",
        ))
    return out


def _collect_plugin_lsp(install_path: Path) -> Dict[str, Any]:
    """收集插件声明的 LSP 服务器，按 languageId 索引。

    来源（对齐 claude-code 插件 LSP 集成）：
    - .lsp.json：列表 [{languageId, command, args, ...}] 或 {"lspServers": [...]}；
    - .claude-plugin/plugin.json 的 lspServers 字段。
    返回值形如 {"python": {"command": "...", "args": [...]}, ...}。
    """
    out: Dict[str, Any] = {}

    def _ingest(entries):
        if isinstance(entries, dict) and isinstance(entries.get("lspServers"), list):
            entries = entries["lspServers"]
        if not isinstance(entries, list):
            return
        for e in entries:
            if not isinstance(e, dict):
                continue
            lang = e.get("languageId") or e.get("language")
            cmd = e.get("command")
            if lang and cmd:
                out[str(lang)] = {
                    "command": cmd,
                    "args": e.get("args", []) or [],
                }

    # 1) .lsp.json（裸列表或 {lspServers:[...]}）
    lsp_file = install_path / ".lsp.json"
    if lsp_file.exists():
        try:
            _ingest(json.loads(lsp_file.read_text(encoding="utf-8")))
        except Exception:
            pass
    # 2) .claude-plugin/plugin.json 的 lspServers
    pj = install_path / ".claude-plugin" / "plugin.json"
    if pj.exists():
        try:
            _ingest(json.loads(pj.read_text(encoding="utf-8")).get("lspServers", []))
        except Exception:
            pass
    return out


def _load_plugin_tools(tools_dir: Path, plugin_name: str) -> List[str]:
    """扫描插件 tools/ 目录，导入每个 .py 模块。

    模块内的 @tool 装饰器会在 import 时自动把工具注册进 ToolRegistry，
    因此「加载插件 = 注册其自带工具」。返回本次新注册的工具名列表。
    """
    out: List[str] = []
    if not tools_dir.exists():
        return out
    for py in sorted(tools_dir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        before = set(ToolRegistry.names())
        try:
            mod_name = f"_plugin_tool_{plugin_name}_{py.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, str(py))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod   # 缓存，便于后续引用/卸载
            spec.loader.exec_module(mod)   # @tool 装饰器在此触发 ToolRegistry.register
        except Exception as e:
            sys.modules.pop(mod_name, None)
            logger.info('%s', f'[plugins] 工具模块 {py.name} 加载失败：{e}')
            continue
        added = [n for n in ToolRegistry.names() if n not in before]
        out.extend(added)
        if added:
            logger.info('%s', f"[plugins] 插件 {plugin_name} 注册工具：{', '.join(added)}")
    return out


def load_plugins(plugins_root: Any = None, enable_mcp: bool = False,
                 register_agents: bool = True) -> Dict[str, Any]:
    """扫描 installed_plugins.json，加载所有已装插件的 skills / commands / agents / mcp / lsp 配置。

    plugins_root: 插件目录（默认 ~/.claude/plugins）。
    enable_mcp:   是否把插件 .mcp.json 中的服务器纳入 MCPManager（对齐 --enable-plugin-mcp）。
                  默认 False —— 插件 MCP 不自动连接，避免未授权外连。
                  （LSP 服务器为本地子进程、按需惰性启动，不受此开关影响。）
    register_agents: 是否把插件 agents/*.md 注册为子智能体（会懒导入 supervisor）。
                  forge v2（单 driver loop，无子智能体）传 False 以避免拉起多角色编排层。
    返回统计信息。
    """
    global PLUGIN_SKILLS, PLUGIN_MCP_SERVERS, PLUGIN_STATE
    import swe_agent.mcp as mcp  # 懒导入，便于在 enable_mcp 时赋值
    import swe_agent.lsp as lsp  # 懒导入，用于写入 PLUGIN_LSP_SERVERS
    from . import skills         # 用于写入 PLUGIN_SKILLS 并刷新 ALL_SKILLS

    # 每次调用前先清空插件后端，保证幂等（避免上次加载残留污染本次结果）
    skills.PLUGIN_SKILLS = []
    mcp.PLUGIN_MCP_SERVERS = {}
    lsp.PLUGIN_LSP_SERVERS = {}
    for _n in PLUGIN_TOOL_NAMES:
        ToolRegistry.unregister(_n)
    PLUGIN_TOOL_NAMES.clear()
    empty_state = {"plugins": [], "skills": 0, "commands": 0, "agents": {}, "mcp": [], "lsp": []}

    root = Path(plugins_root) if plugins_root else PLUGINS_ROOT
    manifest = root / "installed_plugins.json"
    if not manifest.exists():
        logger.info('%s', f'[plugins] 未找到 {manifest}，跳过插件加载。')
        skills.reload_skills()
        return empty_state
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as e:
        logger.info('%s', f'[plugins] 清单解析失败：{e}')
        skills.reload_skills()
        return empty_state

    skill_list: List[Skill] = []
    mcp_servers: Dict[str, Any] = {}
    lsp_servers: Dict[str, Any] = {}
    agent_descs: Dict[str, str] = {}
    n_commands = 0
    plugin_tool_names: List[str] = []
    loaded: List[str] = []
    seen: set = set()

    for full_name, entries in (data.get("plugins") or {}).items():
        plugin_name = full_name.split("@")[0]
        if plugin_name in seen:      # 同名插件只保留第一个（跨 marketplace 去重）
            continue
        install_path = None
        if isinstance(entries, list) and entries:
            p = entries[0].get("installPath")
            if p:
                install_path = Path(p)
        if install_path is None or not install_path.exists():
            continue
        seen.add(plugin_name)
        loaded.append(plugin_name)

        # 1) skills/<name>/SKILL.md（格式与本项目 skills 完全兼容）
        for s in load_skills_from_dir(install_path / "skills"):
            s.source = "plugin"
            s.name = f"{plugin_name}:{s.name}"
            skill_list.append(s)

        # 2) commands/*.md -> 技能（slash command 语义）
        cmds = _load_plugin_commands(install_path / "commands", plugin_name)
        skill_list.extend(cmds)
        n_commands += len(cmds)

        # 3) agents/*.md -> 子智能体（forge v2 可关闭，避免懒导入 supervisor）
        if register_agents:
            agent_descs.update(_register_plugin_agents(install_path / "agents"))

        # 4) .mcp.json（裸格式 / mcpServers 包装格式都支持）
        mcp_file = install_path / ".mcp.json"
        if mcp_file.exists():
            try:
                cfg = json.loads(mcp_file.read_text(encoding="utf-8"))
                mcp_servers.update(normalize_mcp_config(cfg))
            except Exception:
                pass

        # 5) .lsp.json / plugin.json lspServers（按 languageId 索引）
        lsp_servers.update(_collect_plugin_lsp(install_path))

        # 6) tools/*.py -> 工具（@tool 装饰器在导入时注册到 ToolRegistry）
        t_names = _load_plugin_tools(install_path / "tools", plugin_name)
        plugin_tool_names.extend(t_names)
        PLUGIN_TOOL_NAMES.update(t_names)

    # 写入技能后端并刷新全局清单（含磁盘 + bundled + 插件）
    skills.PLUGIN_SKILLS = skill_list
    skills.reload_skills()

    # 写入 MCP 后端（仅当启用）
    if enable_mcp:
        mcp.PLUGIN_MCP_SERVERS = mcp_servers
    else:
        mcp.PLUGIN_MCP_SERVERS = {}

    # 写入 LSP 后端（按 languageId 索引，写文件时惰性启动对应服务器）
    lsp.PLUGIN_LSP_SERVERS = lsp_servers

    PLUGIN_SKILLS = skill_list
    PLUGIN_MCP_SERVERS = mcp.PLUGIN_MCP_SERVERS
    PLUGIN_STATE = {
        "plugins": loaded,
        "skills": len(skill_list) - n_commands,
        "commands": n_commands,
        "agents": agent_descs,
        "tools": plugin_tool_names,
        "mcp": list(mcp_servers.keys()) if enable_mcp else [],
        "lsp": list(lsp_servers.keys()),
    }
    logger.info('%s', f"[plugins] 已加载 {len(loaded)} 个插件：{', '.join(loaded)}")
    logger.info('%s', f"[plugins] 技能 {PLUGIN_STATE['skills']} 个、命令 {n_commands} 个、工具 {len(plugin_tool_names)} 个（{', '.join(plugin_tool_names) or '无'}）、子智能体 {len(agent_descs)} 个、插件层 MCP {len(mcp_servers)} 个（{('已启用' if enable_mcp else '未启用（enable_mcp=False）；REPO_ROOT/mcp.json 见 [mcp] 行')}）、LSP 服务器 {len(lsp_servers)} 个（{', '.join(lsp_servers.keys()) or '无'}）")
    return PLUGIN_STATE


def plugins_prompt_section() -> str:
    """把插件带来的子智能体 / 命令写进系统提示词。"""
    if not PLUGIN_STATE["plugins"]:
        return ""
    lines = ["", "29) 插件扩展（来自 claude 插件目录的 claude-code 插件）："]
    if PLUGIN_STATE["skills"] or PLUGIN_STATE["commands"]:
        lines.append(f"新增技能/命令 {PLUGIN_STATE['skills'] + PLUGIN_STATE['commands']} 个"
                     "（名称带 插件名: 前缀，用 skill 动作调用，见上文技能清单）。")
    if PLUGIN_STATE["tools"]:
        lines.append("新增工具（已注册到动作清单，可直接用对应 action 调用）："
                     + "、".join(f"`{t}`" for t in PLUGIN_STATE["tools"]))
    return "\n".join(lines)
