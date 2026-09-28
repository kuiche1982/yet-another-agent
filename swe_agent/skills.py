#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""swe_agent/skills.py —— 技能系统（Skills）。

忠实移植自最初 demo.py 的技能子系统（对齐 claude-code 的 skills 目录加载 +
SkillTool 执行）。本模块是「插件扩展层」中技能部分的真实后端：supervisor 把
`skill` 动作路由到这里。

- 技能来源：磁盘目录（REPO_ROOT/skills/<name>/SKILL.md，claude-code 兼容格式）
  + 内置（bundled）技能 + 插件技能。
- 执行：inline 技能把指令注入对话上下文（由模型接着执行）；fork 技能在子智能体中独立运行。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import swe_agent.config as config
from swe_agent.log import logger


@dataclass
class Skill:
    name: str
    description: str
    content: str
    when_to_use: str = ""
    allowed_tools: List[str] = None
    model: str = ""
    context: str = "inline"  # inline | fork
    source: str = "disk"     # disk | bundled

    def __post_init__(self):
        if self.allowed_tools is None:
            self.allowed_tools = []


def _parse_frontmatter(text: str):
    """极简 YAML frontmatter 解析：--- key: value --- 后接正文。"""
    if not text.lstrip().startswith("---"):
        return {}, text
    body = text.lstrip()
    end = body.find("\n---", 3)
    if end == -1:
        return {}, text
    fm = body[3:end].strip("\n")
    content = body[end + 4:].lstrip("\n")
    data = {}
    for line in fm.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data, content


def load_skills_from_dir(skills_dir: Path) -> List[Skill]:
    """加载 <skills_dir>/<name>/SKILL.md，对齐 claude-code 的目录格式。"""
    out = []
    if not skills_dir.exists():
        return out
    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        sk = entry / "SKILL.md"
        if not sk.exists():
            continue
        try:
            raw = sk.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, body = _parse_frontmatter(raw)
        out.append(Skill(
            name=fm.get("name") or entry.name,
            description=fm.get("description", entry.name),
            content=body,
            when_to_use=fm.get("when_to_use", ""),
            allowed_tools=[t.strip() for t in fm.get("allowed-tools", "").split(",") if t.strip()],
            model=fm.get("model", ""),
            context="fork" if fm.get("context", "") == "fork" else "inline",
            source="disk",
        ))
    return out


# 内置（bundled）skills —— 对应 claude-code 的 registerBundledSkill（无需磁盘文件）。
BUNDLED_SKILLS: List[Skill] = []


def register_bundled_skill(name, description, content, when_to_use="", context="inline"):
    BUNDLED_SKILLS.append(Skill(
        name=name, description=description, content=content,
        when_to_use=when_to_use, context=context, source="bundled",
    ))


# 注册两个示例 bundled skills，证明该机制可用。
register_bundled_skill(
    "review", "对刚写的代码做自我审查（命名 / 边界 / 错误处理 / 测试覆盖）",
    "你现在是代码审查专家。请 read_file 查看本次改动的文件，重点检查：\n"
    "1) 命名与可读性；2) 边界条件与空输入；3) 错误处理是否完备；4) 是否有对应 pytest 测试。\n"
    "用中文给出审查结论（3-5 条要点），并指出是否需要补充测试。不要修改代码，只评审。",
    when_to_use="当代码初版写完、准备提交或运行测试前，想先做一次自检时",
)
register_bundled_skill(
    "gen-tests", "为模块生成 pytest 测试用例（含从零开始的新项目）",
    "你现在是测试专家。分两种情况：\n"
    "A. 目标模块已存在：先 read_file 读取它，理解公开函数/方法与前置条件，再写测试。\n"
    "B. 全新项目（模块尚不存在）：跳过 read_file，直接根据任务需求设计公开接口，\n"
    "   用 write_file 写 test_<module>.py（pytest 的 test_ 函数，含正常、边界、异常用例），\n"
    "   测试先行钉住接口，再用 write_file 写实现使其通过。\n"
    "最后用 shell 运行 python -m pytest -q 验证。不要重复调用本技能。",
    when_to_use="当某个模块已写好、但还没有充分测试，需要补单元测试时；或全新项目要先写测试时",
)

# 插件技能（带 插件名: 前缀），由插件加载器填充；此处保留空容器以便 reload_skills 引用。
PLUGIN_SKILLS: List[Skill] = []

ALL_SKILLS: List[Skill] = []
SKILLS_DIR = config.REPO_ROOT / "skills"


def reload_skills():
    """重新扫描 skills 目录 + bundled + 插件 skills，构建可用技能清单。"""
    global ALL_SKILLS
    disk = load_skills_from_dir(SKILLS_DIR)
    by = {s.name: s for s in BUNDLED_SKILLS}
    for s in disk:                      # 磁盘技能可覆盖同名 bundled
        by[s.name] = s
    for s in PLUGIN_SKILLS:             # 插件技能（带 插件名: 前缀）
        by[s.name] = s
    ALL_SKILLS = list(by.values())


def get_skill(name: str) -> Optional[Skill]:
    for s in ALL_SKILLS:
        if s.name == name:
            return s
    return None


def skills_prompt_section() -> str:
    if not ALL_SKILLS:
        return ""
    lines = [
        "",
        "27) 调用技能（skill，移植自 claude-code 的 SkillTool）：",
        "以下技能会把专属指令注入上下文，由你接着执行：",
    ]
    for s in ALL_SKILLS:
        wt = f"（适用：{s.when_to_use}）" if s.when_to_use else ""
        lines.append(f"- {s.name}：{s.description}{wt}")
    lines.append('调用格式：{"action":"skill","name":"技能名","args":"可选参数"}')
    lines.append("- 默认把技能指令注入上下文由你继续执行；若技能声明 fork，则在子智能体中独立运行。")
    return "\n".join(lines)


def run_skill(name: str, args: str, messages=None) -> str:
    """执行技能：inline 把指令注入 messages 上下文；fork 在子智能体独立运行。

    messages 应为可变的对话列表（ActionContext.messages）；传入 None 时退化为仅返回提示。
    """
    skill = get_skill(name)
    if not skill:
        avail = ", ".join(s.name for s in ALL_SKILLS) or "（无）"
        return f"skill_error: 未找到技能 '{name}'，可用技能：{avail}"
    content = skill.content
    # 参数替换（对齐 claude-code 的 substituteArguments / $ARGUMENTS）
    content = content.replace("${ARGUMENTS}", args or "").replace("$ARGUMENTS", args or "")
    if skill.context == "fork":
        # 子智能体机制在 supervisor 中；懒导入避免循环依赖
        try:
            from .supervisor import run_subagent
        except Exception as e:
            return f"skill_error: 无法加载子智能体执行器：{e}"
        logger.info('%s', f'\n[skill] 以 fork 子智能体运行：{name}')
        rep = run_subagent("general-purpose", f"# 技能指令：{name}\n{content}", "medium")
        return f"[{name} 技能（fork）返回]\n{rep}"
    # inline：把技能指令作为 user 消息注入上下文，模型下一轮即遵循
    if messages is not None:
        header = f"# 已加载技能：{name}\n以下为技能提供的专属指令，请严格遵循："
        messages.append({"role": "user", "content": f"{header}\n\n{content}"})
        return f"skill_loaded: 已把技能 '{name}' 的指令注入上下文，请按指令继续。"
    return "skill_error: 无法获取对话上下文以注入技能"


# 模块导入即加载技能清单（纯函数、无副作用）
reload_skills()
