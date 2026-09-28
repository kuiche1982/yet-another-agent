#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/supervisor.py —— 调度层（Supervisor）状态机

职责（对齐 SWE-Agent 架构文档的「调度层」）：
- 维护任务阶段（未规划/编码中/校验中/评审中/完成）；
- 驱动 Worker 工具（planner/executor）与客观校验层（harness 传感器流水线）；
- 基于结构化 Fact 做故障路由决策（重试编码 / 回退重规划 / 进入评审）；
- 只消费结构化 Fact，不读原始日志（红线）。

本模块是 demo.py 主循环的忠实移植 + 自描述改造：
- 工具清单由 ToolRegistry 动态生成（不再手写进 SYSTEM_PROMPT）；
- 外围集成（skills / MCP / plugins / LSP diagnostics / worktree）在分层架构里属于「插件扩展层」，
  此处以明确提示返回，便于后续作为插件接入，不阻塞核心 PDCA 闭环。
"""

import os
import sys
import json
import time
import argparse
import subprocess
import threading
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from .agent import Agent
from . import log as _logmod

import swe_agent.config as config
from . import models as M
from .state import (
    GLOBAL_STATE, reset_state, get_current_task, mark_current_task_done,
    _turn_fingerprint, _safe_rel, _telemetry, _reset_run_telemetry,
    stats_summary, RUN_TELEMETRY,
)
from .registry import ToolRegistry, SensorFact, tool, ActionContext, ROLE_TOOLS
from .tools import (
    read_file, exec_shell,
    grep_files, glob_files,
)
from .actions import parse_actions
from .compact import (
    MANUAL_COMPACT_REQUESTED, start_stdin_monitor,
)
from . import lfm_sidecar as _sidecar
from .roles import (
    run_planner, run_analyzer,
    plan_contract_section, drift_issues,
)
from .harness import _run_test_bar, _new_subagent_fence, run_harness_pipeline
from . import harness as _harness
from . import verify as _verify
from . import layers
from .agent import RunState
from .hooks import HookPoint, HOOK_HUB, GateAction
from .roles_config import make_chat_agent


def _snapshot_harness_facts() -> List[SensorFact]:
    """跑全量自描述传感器流水线，把结构化事实写入 GLOBAL_STATE（状态只存结构化对象）。

    设计文档红线：调度层只消费结构化 SensorFact；原始日志不向上透传。
    此快照与单杠校验（_run_test_bar）并存——单杠校验负责“测试 count>0 且全绿”这一唯一完成标准
    编排逻辑，这里提供「纯代码感知层」的统一结构化视图，供调度与插件消费。
    """
    try:
        facts = run_harness_pipeline(plan=GLOBAL_STATE.get("plan"))
    except Exception as e:  # 感知层异常绝不阻断主闭环
        facts = [SensorFact(sensor_name="harness", ok=False,
                            message=f"harness pipeline error: {e}", severity="error")]
    GLOBAL_STATE["harness_facts"] = [f.to_dict() for f in facts]
    return facts


# ======================================================================
# 工具清单：自描述生成（替代原 SYSTEM_PROMPT 内嵌的硬编码工具列表）
# ======================================================================
def _tools_prompt_section() -> str:
    """依据 ToolRegistry 中每个工具的自描述元数据，动态生成『可用工具』提示词片段。

    顶层 Executor 不展示 agent 动作：弱模型用它甩锅给零上下文的子智能体，既无价值又
    浪费轮次（agent 屏蔽由 _hook_block_executor_agent 在 BUILD 层兜底，此处只是不在
    提示里「勾引」它）。plan 子智能体经 run_subagent 派发，不经过本段提示，不受影响。
    """
    frag = ToolRegistry.prompt_fragment()
    if not config.EXECUTOR_ALLOW_AGENT:
        frag = "\n".join(ln for ln in frag.split("\n")
                         if not ln.strip().lstrip("- ").startswith("agent"))
    return "\n\n# 可用工具（动作名 → 说明）\n" + frag + (
        "\n\n输出格式：严格只输出一个 JSON 动作对象 {\"action\": <动作名>, ...}。"
        "动作名必须与上面一致。"
    )


# ======================================================================
# 工作环境 / 上下文
# ======================================================================
_HEAVY_DIRS = ("node_modules", ".git", "__pycache__", ".venv",
               "litert-lm-cache", ".mypy_cache", "agent_sandbox")


def _workspace_context_section() -> str:
    """把『当前工作目录 + 初始文件清单』注入系统提示。"""
    try:
        abs_ws = config.WORKSPACE.resolve()
    except Exception:
        abs_ws = config.WORKSPACE
    entries: List[str] = []
    try:
        for p in sorted(abs_ws.rglob("*")):
            if any(part in _HEAVY_DIRS for part in p.parts):
                continue
            try:
                entries.append(str(p.relative_to(abs_ws)))
            except Exception:
                continue
    except Exception:
        entries = []
    if entries:
        tree = "\n".join(f"  - {e}" for e in entries[:200])
        body = ("当前工作目录（即你的 cwd）已为你定位好，直接写相对路径即可，如 `src/fib.py`；"
                "**严禁**写绝对路径、严禁带工作目录前缀、无需 cd：\n"
                f"工作区初始文件清单：\n{tree}")
    else:
        body = ("当前工作目录（即你的 cwd）已为你定位好，直接写相对路径即可，如 `src/fib.py`；"
                "**严禁**写绝对路径、严禁带工作目录前缀、无需 cd：\n"
                f"工作区初始状态：**空目录**（从零开始，请自行创建所需文件与目录）。")
    return "\n\n# 工作环境\n" + body


def build_context(subtask: str) -> str:
    ctx = f"【目标】{GLOBAL_STATE['goal']}\n"
    if GLOBAL_STATE["tasks"]:
        ctx += "【任务列表】\n"
        for t in GLOBAL_STATE["tasks"]:
            dl = t.get("deliverables") or []
            dl_s = (" → 交付：" + ", ".join(dl)) if dl else ""
            ctx += f"  - [{t['status']}] {t['desc']}{dl_s}\n"
    ctx += f"【已完成】{', '.join(GLOBAL_STATE['done_list']) or '无'}\n"
    ctx += f"【当前待办】{subtask}\n"
    return ctx


# ======================================================================
# Executor 历史重置（任务完成时降本 + 防漂）
# ======================================================================
def _capture_last_turns(messages: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
    """从对话历史里抽取最近 k 个「(assistant 动作 → user 工具结果)」轮，作连续性上下文。

    只收 assistant→user(且 user 不是每轮下发的 build_context 提示) 的配对，
    避免把每轮的【目标】提示也误当成连续性。"""
    turns: List[tuple] = []
    buf = None
    for m in messages[1:]:  # 跳过 system
        if m.get("role") == "assistant":
            buf = m
        elif m.get("role") == "user" and buf is not None:
            if not str(m.get("content", "")).startswith("【目标】"):
                turns.append((buf, m))
            buf = None
    if k > 0:
        turns = turns[-k:]
    out: List[Dict[str, str]] = []
    for a, u in turns:
        out.append(a)
        out.append(u)
    return out


def _rebuild_executor_context(messages: List[Dict[str, str]],
                              user_msg: str, keep_turns: int) -> List[Dict[str, str]]:
    """任务完成后的历史重建：

        [ 静态 system(命中缓存) + 最近 keep_turns 轮(连续性) + 更新后的任务列表(指令置底) ]

    关键：任务列表(=当前 user_msg)必须置底，否则 chat 循环会让模型对陈旧工具结果作答。
    """
    system = messages[0] if (messages and messages[0].get("role") == "system") else {
        "role": "system", "content": ""}
    tail = _capture_last_turns(messages, keep_turns)
    return [system, *tail, {"role": "user", "content": user_msg}]


def build_system_prompt(userInputStr:str = None, lang:str = "") -> str:
    """在基础 SYSTEM_PROMPT 上：追加工作环境 + 顶层 Planner 契约（如有）。

    弱模型 executor 走精简 WEAK_SYSTEM_PROMPT（BUILD 层已强制约束不进 prompt）；
    强模型 executor 走 SYSTEM_PROMPT。两者都只追加「工作环境 + 契约」，跳过对编码无帮助的目录式 section
    （这些工具经 FC / 工具注册表已对模型可见，无需在 system 里再列一遍）。
    """
    if M.is_weak_executor():
        body = config.WEAK_SYSTEM_PROMPT
    else:
        body = config.SYSTEM_PROMPT
    # 根因修复：不要把完整绝对路径字面灌进 prompt——弱模型会把它当基前缀回写，
    # 造成双层路径（agent_sandbox/<run_id>/fib/<run_id>/fib/...）。模型靠工具 PWD=WORKSPACE
    # 与 pwd/ls 就能定位，prompt 只给稳定、不可照抄的相对句柄。
    body = body.replace("{WORKSPACE}", "当前工作目录（即你的 cwd，直接写相对路径）")
    if lang != "":
        body += f"\n当前项目编程语言：{lang}\n"
    if  userInputStr is not None:
        body += f"\n用户原始要求：{userInputStr}\n"
    sections = [
        _workspace_context_section(),
        plan_contract_section(),
    ]
    body = body + "".join(s for s in sections if s)
    # 原生 function calling：工具已通过 API tools 参数下发，禁止在系统提示里再贴
    # 「输出 JSON 动作对象」式的扁平格式说明——那会诱使弱模型退化成手写 JSON 而非调用工具
    # （forge 教训：工具清单进 BUILD 层 / API，不进 prompt）。
    if M.role_fc("executor") != M.FCCapability.NATIVE_TOOLS:
        body += _tools_prompt_section()
    if M.role_fc("executor") == M.FCCapability.NATIVE_TOOLS:
        lang = GLOBAL_STATE.get("lang", "python")
        if lang == "go":
            code_rule = ("你写出的代码必须是【可解析的合法 Go（gofmt 风格）】：只用 ASCII 运算符与标点"
                         "（比较用 <= >= !=，禁止 ≤ ≥ ≠ 等数学符号；引号/括号一律半角），"
                         "写完实现后必须用 shell 跑 `go test ./...` 确认真实通过，不要只 go build 就宣称完成。")
        else:
            code_rule = ("你写出的代码必须是【可解析的合法 Python】：只用 ASCII 运算符与标点"
                         "（比较必须用 <= >= !=，禁止 ≤ ≥ ≠ 等数学符号；引号/括号一律半角），"
                         "写完实现后必须用 shell 跑 pytest 确认真实通过，不要只跑 import 就宣称完成。")
        body += (
            "\n\n# ⚠️ Executor 输出约束（原生 function calling 模型）\n"
            "你必须通过【调用工具】来输出每一个动作（每个动作是一个独立 function，如 write_file / "
            "edit_file / shell / read_file / grep / glob / complete / verify），【不要】在回复里手写 JSON 文本、"
            "也不要用 ```json 代码块。每轮只调用一个工具，多行内容直接填入参数，无需转义或三引号包裹。\n"
            + code_rule
        )
    if GLOBAL_STATE.get("plan"):
        return config.CONTRACT_OVERRIDE_BANNER + body
    return body


# 语言决策已下放给 Planner（见 roles.py _PLANNER_SYSTEM 的 language 字段）。
# 此处不再用字符串正则猜测；Planner 不确定时 run_agent 默认 python。


# ======================================================================
# 接手模式 / 任务交付物辅助
# ======================================================================
def _is_test_file(relpath: str) -> bool:
    return bool(config._TEST_FILE_RE.search(str(relpath).replace("\\", "/")))


def _snapshot_existing_tests() -> List[str]:
    found: List[str] = []
    try:
        for p in config.WORKSPACE.rglob("*"):
            if not p.is_file():
                continue
            rel = _safe_rel(str(p))
            if rel and _is_test_file(rel):
                found.append(rel)
    except Exception:
        pass
    return sorted(set(found))


def _detect_takeover_mode(task_text: str, preexisting: List[str]) -> bool:
    if not preexisting:
        return False
    t = (task_text or "").lower()
    return any(k.lower() in t for k in config._TAKEOVER_KEYWORDS)


def _check_takeover_protection(path: str) -> str:
    """接手模式下阻止覆盖既有测试夹具。返回空串表示放行，非空为拦截说明。"""
    if not GLOBAL_STATE.get("takeover_mode"):
        return ""
    rel = _safe_rel(path)
    protected = GLOBAL_STATE.get("protected_tests") or set()
    if rel not in protected:
        return ""
    return (f"🛡️【接手保护】{rel} 是本次任务开始前就已存在的测试夹具（验收标准），"
            "禁止覆盖或重写。你的职责是修改**实现代码**让这些既有测试由红变绿，而不是改测试。\n"
            f"当前受保护测试：{', '.join(sorted(protected))}。\n"
            "请 read_file 现有实现 → 定位 bug → 用 edit_file 精确修复实现；"
            "确需补充测试时请使用**新的文件名**，不要覆盖上述文件。")


def _check_impl_overwrite_protection(path: str) -> str:
    """接手模式下，禁止用 write_file 整体覆盖/重写「继承来的实现文件」。

    只允许 edit_file 在其上做最小修复（改红为绿），避免弱 executor 把既有实现
    整文件重写成一堆垃圾、导致自测直接 import 报错而绕开 #84 隐藏验收闸门。
    """
    if not GLOBAL_STATE.get("takeover_mode"):
        return ""
    rel = _safe_rel(path)
    protected = GLOBAL_STATE.get("protected_impl") or set()
    if rel not in protected:
        return ""
    return (f"🛡️【接手保护】{rel} 是任务开始前就存在的实现文件（继承代码），"
            "禁止用 write_file 整体覆盖/重写。如需改动请用 edit_file 在其上做最小修复。")


def _auto_lsp_hint(path: str) -> str:
    """LSP 诊断提示（harness 传感器）：写入/编辑文件后回灌非阻塞诊断。

    委托真实 LSP 客户端（swe_agent.lsp）；未配置 LSP 服务器或任何异常时返回空串，
    绝不拖垮 write/edit 钩子（对齐 claude-code：写文件后自动展示 LSP 诊断）。
    """
    try:
        return _lsp.get_lsp().hint(path)
    except Exception:
        return ""


# ======================================================================
# 会话持久化
# ======================================================================
def new_session_id() -> str:
    import datetime
    return "s-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def save_session(session_id: str, messages: List[Dict[str, str]],
                truth: "Optional[List[Dict[str, Any]]]" = None) -> None:
    try:
        config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        state_dump = {}
        for k, v in GLOBAL_STATE.items():
            state_dump[k] = sorted(v) if isinstance(v, (set, frozenset)) else v
        payload = {
            "session_id": session_id,
            "goal": GLOBAL_STATE.get("goal", ""),
            "state": state_dump,
            "messages": messages,
            # 完整会话史（原文，未压缩）—— 与 messages（压缩后工作集）并列落盘：
            # 重启恢复时 messages 直接复用（不再重新压缩），truth 供 recall / 会话历史不丢失。
            "truth": truth if truth is not None else messages,
            "stats": config.STATS,
        }
        (config.SESSIONS_DIR / f"{session_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        logger.info('%s', f'[session] 保存失败：{e}')


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    f = config.SESSIONS_DIR / f"{session_id}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        logger.info('%s', f'[session] 读取失败：{e}')
        return None


def list_sessions() -> List[str]:
    if not config.SESSIONS_DIR.exists():
        return []
    return sorted(p.stem for p in config.SESSIONS_DIR.glob("*.json"))


# ======================================================================
# 后台 shell（轻量实现；复杂场景可移入 plugins 层）
# ======================================================================
_bg_tasks: Dict[str, subprocess.Popen] = {}
_bg_counter = 0


def run_shell_bg(cmd: str) -> str:
    global _bg_counter
    _bg_counter += 1
    tid = f"bg-{_bg_counter}"
    try:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(config.WORKSPACE),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        _bg_tasks[tid] = proc
        return f"background_started: {tid}（用 task_output 读取结果）"
    except Exception as e:
        return f"shell_error: {e}"


def task_output(task_id: str, block: bool = False, timeout: int = 30) -> str:
    proc = _bg_tasks.get(task_id)
    if proc is None:
        return f"task_error: 找不到后台任务 {task_id}"
    try:
        if block:
            out, _ = proc.communicate(timeout=timeout)
        else:
            try:
                out = proc.stdout.read() if proc.stdout else ""
            except Exception:
                out = ""
    except Exception:
        out = ""
    if proc.poll() is None and not block:
        return f"[{task_id}] 仍在运行……（block=true 可阻塞等待）"
    return out or ""


def task_stop(task_id: str) -> str:
    proc = _bg_tasks.get(task_id)
    if proc is None:
        return f"task_error: 找不到后台任务 {task_id}"
    proc.terminate()
    return f"task_stopped: {task_id}"


def sleep_action(seconds: int = 1) -> str:
    time.sleep(max(0, min(int(seconds or 1), 60)))
    return f"slept {seconds}s"


# ======================================================================
# 内置子智能体（只读 explore / plan；通用 general-purpose）
# ======================================================================
SUBAGENT_MAX_ITER = 12
_SUB_READONLY = ["read_file", "grep", "glob", "web_fetch", "report"]
SUBAGENT_ALLOWED = {
    "explore": _SUB_READONLY,
    "plan": _SUB_READONLY + ["agent"],   # plan 可嵌套派发 explore 等子智能体（对齐 claude plan mode）
    "general-purpose": ["read_file", "write_file", "edit_file", "shell",
                        "grep", "glob", "web_fetch", "report",
                        "task_output", "task_stop", "sleep"],
}

# 子智能体「职责 + 何时用」单一真源：既用于生成父系统提示的目录（agents_prompt_section），
# 也作为各子智能体自身系统提示的职责首句，避免与 SUBAGENT_PROMPTS 文本漂移。
SUBAGENT_ROLES = {
    "explore": "只读探索专家：搜索/定位代码与实现，不写文件、不跑命令。先摸清现状、定位相关位置时优先使用。",
    "plan": "架构与规划专家：只读探索后产出分步实现计划；可再派发 explore 做深入调研。动手实现前先设计模块/接口时使用。",
    "general-purpose": "通用执行体：可读写文件、跑命令、跑测试，完成具体实现与验证。需要真正改代码/落盘时使用。",
}
SUBAGENT_PROMPTS = {
    "explore": """你是负责探索代码库的搜索专家。每次只输出一个 JSON 动作对象。只读模式，禁止写文件/跑命令。可用动作：read_file / grep / glob / web_fetch / report。最后用 report 返回发现（content 不能为空）。""",
    "plan": """你是软件架构与规划专家。只读探索，设计实现方案。可用动作：read_file / grep / glob / web_fetch / report / agent（可再派发 explore 做深入调研）。最后用 report 返回分步计划。""",
    "general-purpose": """你是通用智能体。用 write_file/edit_file/shell/grep/glob/web_fetch 完成任务，最后用 report 返回结论（content 不能为空）。""",
}
_EXPLORE_THOROUGHNESS = {
    "quick": "请做基础搜索即可，尽快给出结果。",
    "medium": "请做适度探索，覆盖主要相关位置。",
    "very thorough": "请做全面分析，跨多个位置与命名约定深入搜索。",
}


# 旧的 _sub_execute（手写 if/elif 直调后端、绕过 ToolRegistry 与全局钩子）已删除：
# 子智能体动作现统一在 run_subagent 里经 ToolRegistry.dispatch(a, subagent_ctx) 执行，
# 自动复用与父路径相同的钩子集（接手保护 / 实现覆盖保护 / plan_mode 守卫 / 计数器 / *_after 反馈），
# 集中可控、抓得住问题。需要新增子智能体动作时，只需注册 @tool，无需再改分发主干。


def run_subagent(kind: str, prompt: str, thoroughness: str = "medium") -> str:
    kind = (kind or "general-purpose").lower()
    if kind not in SUBAGENT_PROMPTS:
        kind = "general-purpose"
    sys_p = SUBAGENT_PROMPTS[kind]
    if kind == "explore":
        sys_p += "\n\n" + _EXPLORE_THOROUGHNESS.get(thoroughness, _EXPLORE_THOROUGHNESS["medium"])
    allowed = SUBAGENT_ALLOWED[kind]
    # 子智能体执行上下文：携带白名单，使全局 before 钩子（含本层 _hook_subagent_whitelist）
    # 能在 BUILD 层兜底拦截越权动作。子智能体动作统一走 ToolRegistry.dispatch，
    # 复用与父路径完全一致的钩子集（接手保护 / 实现覆盖保护 / plan_mode 守卫 / 计数器 / *_after 反馈），
    # 杜绝原来 _sub_execute 直接调后端函数导致「子智能体乱干、抓不住问题」的安全缺口。
    subagent_ctx = ActionContext(extra={"subagent_allowed": allowed,
                                       "subagent_kind": kind,
                                       "_fence": _new_subagent_fence()})
    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": prompt},
    ]
    report = None
    for _ in range(SUBAGENT_MAX_ITER):
        out = M.chat_messages("executor", messages, tools=ToolRegistry.glm_tools(role=kind))
        if out.startswith("llm_error"):
            return f"subagent_error: {out}"
        actions = parse_actions(out)
        if not actions:
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": "你的输出无法解析为合法动作 JSON，请只输出一个动作对象。"})
            continue
        actions = [a for a in actions if a.get("action") in allowed]
        if not actions:
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"该子智能体仅允许动作：{allowed}。请使用允许的动作。"})
            continue
        results = []
        stop = False
        for a in actions:
            if a.get("action") == "report":
                content = (a.get("content", "") or "").strip()
                if not content:
                    messages.append({"role": "assistant", "content": out})
                    messages.append({"role": "user", "content": "你发出了 report 但 content 为空。请重新用 report 动作，在 content 中给出真实发现/结论。"})
                    stop = False
                    break
                report = content
                stop = True
                break
            act = a.get("action")
            # 嵌套派发守卫：仅 plan 子智能体可再派发子智能体；其余一律禁止（防递归 + 越权）。
            # 白名单已把 agent 排除在 explore/general-purpose/插件智能体外，这里再兜底拦截。
            if act == "agent":
                if kind != "plan":
                    messages.append({"role": "assistant", "content": out})
                    messages.append({"role": "user", "content": "子智能体不允许再派发子智能体（仅 plan 子智能体可嵌套调用）。请直接用你被允许的动作完成。"})
                    break
                rep2 = run_subagent(a.get("subagent", "general-purpose"),
                                    a.get("prompt", ""), a.get("thoroughness", "medium"))
                results.append(f"[{a.get('subagent', 'general-purpose')} 子智能体返回]\n{rep2}")
                continue
            # 其余动作统一经 ToolRegistry.dispatch 执行：自动复用全套全局/工具级钩子，
            # 子智能体再也不会绕过接手保护、实现覆盖保护、plan_mode 守卫等约束。
            results.append(ToolRegistry.dispatch(a, subagent_ctx))
        if stop:
            break
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": f"【工具执行结果】\n" + "\n".join(results)})
    if report is None:
        report = "(子智能体未在限定轮数内给出 report，可能未完成)"
    return report


# ======================================================================
# 动作分发（Execute）
# ======================================================================
def execute_action(action_obj: Dict[str, Any], messages: Optional[List[Dict[str, Any]]] = None) -> str:
    """动作分发：统一从 ToolRegistry 匹配 action -> run，并按注册钩子组合执行。

    - action 与实现函数的映射只在 ToolRegistry 一处定义（新增动作 = 注册一个 @tool）；
    - 各类校验/副作用（接手保护、自动完成任务、测试反馈、计数器、plan_mode 守卫）
      全部以「钩子」注册，有就执行、没有就不执行，不再硬编码在分发主干里。
    """
    if not isinstance(action_obj, dict):
        return "无法解析的指令"
    return ToolRegistry.dispatch(action_obj, ActionContext(messages=messages))


# ======================================================================
# 元动作（Meta Actions）：原 execute_action 的硬编码 if/elif 分支，现注册进 ToolRegistry
# ======================================================================
@tool(
    name="plan", category="meta",
    description="制定有序开发任务清单（遵循 TDD：先写会失败的测试钉住接口，再实现使其变绿）。",
    schema={"type": "object",
            "properties": {"tasks": {"type": "array", "items": {"type": "string"}}},
            "required": ["tasks"]},
)
def _m_plan(ctx: ActionContext, tasks: Optional[List[str]] = None) -> str:
    if GLOBAL_STATE.get("planning_done") or GLOBAL_STATE.get("plan"):
        return ("⚠️ 当前任务清单已由规划阶段给定，禁止重新规划。"
                "请直接按系统提示中的任务清单逐项执行（填空式实现），用 complete 推进，不要重写任务列表。")
    tasks = tasks or []
    if tasks:
        # 契约：tasks 应为「一串文字描述」。但部分模型会把任务写成对象
        # {step, description, actions:[...]}（把真实动作也塞进 plan）。这里做归一化：
        # 无论模型给出字符串还是对象，统一抽出「纯描述文本」作为 desc，
        # 丢弃嵌套的 actions（执行由 harness 分步驱动，不在此预展开）。
        norm = []
        for d in tasks:
            if isinstance(d, str):
                desc = d.strip()
            elif isinstance(d, dict):
                desc = (d.get("description") or d.get("desc") or d.get("title") or "").strip()
                if not desc and d.get("step") is not None:
                    desc = f"步骤 {d.get('step')}"
            else:
                desc = str(d).strip()
            if desc:
                norm.append(desc)
        if not norm:
            return "计划为空（未解析到任何任务描述）"
        GLOBAL_STATE["tasks"] = [{"id": i, "desc": desc, "status": "pending"}
                                 for i, desc in enumerate(norm, 1)]
        GLOBAL_STATE["planning_done"] = True
        return f"已规划 {len(norm)} 个任务：{norm}"
    return "计划为空"



@tool(name="complete", category="meta",
      description="声明当前任务已完成（仅推进任务列表；真正的完成标准是工作区测试全部通过）。",
      schema={"type": "object", "properties": {}})
def _m_complete(ctx: ActionContext) -> str:
    task = get_current_task()
    if not task:
        return "已手动完成任务：无待办"
    task["status"] = "done"
    GLOBAL_STATE["done_list"].append(task["desc"])
    return (f"已手动完成任务：{task['desc']}"
            f"（注意：最终完成以工作区测试全部通过为准，harness 会持续校验直到测试全绿）。")


@tool(name="agent", category="meta",
      description="派发子智能体执行子任务（可用子智能体及其职责见系统提示「子智能体清单」）。"
                  "仅顶层智能体（planner/executor）与 plan 子智能体可调用，其余子智能体不可嵌套。",
      schema={"type": "object",
              "properties": {
                  "subagent": {"type": "string"},
                  "prompt": {"type": "string"},
                  "thoroughness": {"type": "string"},
              }})
def _m_agent(ctx: ActionContext, subagent: str = "general-purpose",
             prompt: str = "", thoroughness: str = "medium") -> str:
    rep = run_subagent(subagent, prompt, thoroughness)
    return f"[{subagent} 子智能体返回]\n{rep}"


@tool(name="compact", category="meta",
      description="手动压缩对话历史以释放上下文（需要 messages 上下文）。",
      schema={"type": "object",
              "properties": {"instructions": {"type": "string"}}})
def _m_compact(ctx: ActionContext, instructions: str = "") -> str:
    if ctx is None or ctx.messages is None:
        return "compact_error: 无法获取对话历史"
    instr = instructions or ""
    logger.info('%s', f"\n[压缩] 手动触发（指令：{instr or '无'}）……")
    try:
        new_msgs, stats = _sidecar.manual_compact(ctx.messages, instructions=instr)
        ctx.messages[:] = new_msgs
        return (f"compact_success: 对话已压缩，tokens {stats['pre']} → {stats['post']}"
                f"（节省 {stats['pre'] - stats['post']}）")
    except Exception as e:
        return f"compact_error: {e}"


@tool(name="todo_write", category="meta",
      description="写入/覆盖待办清单（tasks 为有序步骤，可带 status）。",
      schema={"type": "object",
              "properties": {"todos": {"type": "array",
                                       "items": {"type": "object"}}}})
def _m_todo_write(ctx: ActionContext, todos: Optional[List[Any]] = None) -> str:
    if GLOBAL_STATE.get("planning_done"):
        return ("⚠️ 任务清单已由规划阶段给定，禁止覆盖/重新规划待办。"
                "请直接按现有任务清单逐项执行，用 complete 推进，不要重写任务列表。")
    todos = todos or []
    norm = []
    for i, t in enumerate(todos, 1):
        if isinstance(t, str):
            t = {"content": t}
        content = (t.get("content") or t.get("desc") or f"任务{i}").strip()
        status = t.get("status", "pending")
        if status in ("completed", "done"):
            status = "done"
        elif status == "in_progress":
            status = "in_progress"
        else:
            status = "pending"
        norm.append({"id": i, "desc": content, "status": status})
    GLOBAL_STATE["tasks"] = norm
    pending = sum(1 for x in norm if x["status"] == "pending")
    return f"todo_write_success: 已写入 {len(norm)} 项待办（{pending} 项待办 / {len(norm) - pending} 项完成）"


@tool(name="todo_read", category="meta",
      description="查看当前待办清单。",
      schema={"type": "object", "properties": {}})
def _m_todo_read(ctx: ActionContext) -> str:
    if not GLOBAL_STATE["tasks"]:
        return "todo_read: 当前无待办清单（可用 todo_write 或 plan 创建）"
    lines = ["当前待办清单："]
    for t in GLOBAL_STATE["tasks"]:
        mark = "✅" if t["status"] == "done" else ("🔄" if t["status"] == "in_progress" else "⏳")
        lines.append(f"  {mark} [{t['id']}] {t['desc']}")
    return "\n".join(lines)


@tool(name="sleep", category="meta",
      description="暂停若干秒（用于等待后台任务）。",
      schema={"type": "object",
              "properties": {"seconds": {"type": "integer"}}})
def _m_sleep(ctx: ActionContext, seconds: int = 1) -> str:
    return sleep_action(seconds)


@tool(name="task_output", category="meta",
      description="读取后台 shell 任务的输出。",
      schema={"type": "object",
              "properties": {
                  "task_id": {"type": "string"},
                  "block": {"type": "boolean"},
                  "timeout": {"type": "integer"},
              }})
def _m_task_output(ctx: ActionContext, task_id: str = "", block: bool = False,
                   timeout: int = 30) -> str:
    return task_output(task_id, bool(block), int(timeout or 30))


@tool(name="task_stop", category="meta",
      description="终止后台 shell 任务。",
      schema={"type": "object",
              "properties": {"task_id": {"type": "string"}}})
def _m_task_stop(ctx: ActionContext, task_id: str = "") -> str:
    return task_stop(task_id)


@tool(name="report", category="meta",
      description="子智能体返回结论（content 不能为空）。",
      schema={"type": "object",
              "properties": {"content": {"type": "string"}}})
def _m_report(ctx: ActionContext, content: str = "") -> str:
    return "<<REPORT>>" + (content or "")


@tool(name="verify", category="meta",
      description="随时调用独立只读 tester 验证当前实现是否满足 planner 生成的语义验收点"
                  "（verify_points）。返回每条验收点的 pass/fail 与证据。用于编码中途自查，"
                  "不等到 done。tester 看不到你的实现思路，独立判定，防「自己验自己」放水。",
      schema={"type": "object", "properties": {}})
def _m_verify(ctx: ActionContext) -> str:
    from . import verify as _verify
    verdict, detail = _verify.verify_gate()
    return f"[verify] {verdict}: {detail}"


# ---- 插件扩展层桩动作（enter_worktree / exit_worktree / tool_search / enter_plan_mode /
#      exit_plan_mode）按用户决策 #3 暂时移除：不注册即不出现在工具集，避免给本地 7B 模型
#      注入无后端支撑的噪声动作。后续若接入真实 plugins/LSP/worktree 后端再按需注册。 ----

# ======================================================================
# 插件扩展层：已接入真实后端的动作（MCP 工具 + 技能 + LSP）
# ======================================================================
from . import mcp as _mcp
from . import skills as _skills
from . import plugins as _plugins
from . import lsp as _lsp
from swe_agent.log import logger


@tool(
    name="mcp_tool", category="plugin",
    description="调用已连接 MCP 服务器暴露的工具（见 mcp.json，如 demo 服务器的 add/now）。",
    schema={"type": "object", "properties": {
        "server": {"type": "string", "description": "服务器名（mcp.json 中的键，如 demo）"},
        "tool": {"type": "string", "description": "工具名"},
        "arguments": {"type": "object", "description": "工具参数 JSON 对象"},
    }},
)
def _m_mcp_tool(ctx: ActionContext, server: str = "", tool: str = "",
                arguments: Optional[Dict[str, Any]] = None) -> str:
    m = _mcp.get_mcp()
    if m is None:
        return "mcp_error: MCP 未启用或未连接任何服务器"
    return m.call(server, tool, arguments or {})


@tool(
    name="list_mcp_resources", category="plugin",
    description="列出已连接 MCP 服务器暴露的资源（resources/list）。",
    schema={"type": "object", "properties": {}},
)
def _m_list_mcp_resources(ctx: ActionContext) -> str:
    m = _mcp.get_mcp()
    if m is None:
        return "mcp_error: MCP 未启用"
    return _mcp.mcp_list_resources()


@tool(
    name="read_mcp_resource", category="plugin",
    description="读取某个 MCP 资源的内容（resources/read）。",
    schema={"type": "object", "properties": {
        "uri": {"type": "string", "description": "资源 URI"},
    }},
)
def _m_read_mcp_resource(ctx: ActionContext, uri: str = "") -> str:
    m = _mcp.get_mcp()
    if m is None:
        return "mcp_error: MCP 未启用"
    return _mcp.mcp_read_resource(uri)


@tool(
    name="skill", category="plugin",
    description="调用已注册技能（skills/ 目录或内置）；把技能指令注入上下文由模型执行。",
    schema={"type": "object", "properties": {
        "name": {"type": "string", "description": "技能名"},
        "args": {"type": "string", "description": "传给技能的参数（替换 $ARGUMENTS）"},
    }},
)
def _m_skill(ctx: ActionContext, name: str = "", args: str = "") -> str:
    return _skills.run_skill(name, args, ctx.messages)


@tool(
    name="lsp", category="plugin",
    description="对已写/已编辑的文件运行 LSP 诊断（服务器由插件 .lsp.json 提供，或用 --lsp-server 指定）。返回错误/警告列表。",
    schema={"type": "object", "properties": {
        "path": {"type": "string", "description": "要诊断的文件路径（相对 agent_sandbox 或绝对路径）"},
    }},
)
def _m_lsp(ctx: ActionContext, path: str = "") -> str:
    if not path:
        return "lsp_error: 缺少 path"
    return _lsp.lsp_diagnostics(path)


# ======================================================================
# 钩子（Hooks）：可拼装的校验/副作用，注册到具体工具或全局
# ======================================================================
def _hook_count_action(ctx: ActionContext, params: Dict[str, Any], result: Optional[str]) -> Optional[str]:
    """全局 before 钩子：动作计数器（原硬编码在 execute_action 主干）。

    子智能体动作计入其独立的 fence 桶（ctx.extra["_fence"]["actions"]），不污染父 agent
    的全局 STATS["actions"]——保证「谁干的、干了多少」在统计上互相独立、抓得住问题。
    """
    if ctx.action:
        fence = ctx.extra.get("_fence")
        if fence is not None:
            fence["actions"][ctx.action] = fence["actions"].get(ctx.action, 0) + 1
            return None
        config.STATS["actions"][ctx.action] = config.STATS["actions"].get(ctx.action, 0) + 1
    return None


def _hook_plan_mode_guard(ctx: ActionContext, params: Dict[str, Any], result: Optional[str]) -> Optional[str]:
    """全局 before 钩子：规划模式下禁止写文件/执行命令。"""
    if GLOBAL_STATE.get("plan_mode") and ctx.action in ("write_file", "edit_file", "shell"):
        return ("plan_mode_blocked: 当前处于规划模式，禁止写文件/执行命令。"
                "请先退出规划模式再执行。")
    return None


def _hook_subagent_whitelist(ctx: ActionContext, params: Dict[str, Any], result: Optional[str]) -> Optional[str]:
    """全局 before 钩子：子智能体动作白名单（BUILD 层兜底）。

    仅当 ctx.extra 携带 subagent_allowed 时生效 —— 即子智能体经 dispatch 派发的动作。
    父路径 / 顶层 agent 动作的 ctx 不带该字段，此处直接跳过，零影响。
    与 run_subagent 循环内的「allowed 过滤」互为双层防护：即便模型绕过提示层过滤，
    这里的钩子仍会在 BUILD 层拦截越权动作，集中可控、抓得住问题。
    """
    allowed = ctx.extra.get("subagent_allowed")
    if allowed and ctx.action and ctx.action not in allowed:
        return (f"subagent_blocked: 该子智能体仅允许动作：{allowed}。"
                f"请使用允许的动作完成当前子任务。")
    return None


def _hook_block_executor_agent(ctx: ActionContext, params: Dict[str, Any], result: Optional[str]) -> Optional[str]:
    """全局 before 钩子：顶层执行体（Executor）禁止派发子智能体（agent 动作）。

    Executor 的职责是亲自实现代码并跑测试。若放行 agent，弱模型会把活甩给一个
    「从头开始、零任务上下文」的子智能体，且常给出空 prompt（如「请生成 pytest
    测试用例并补全 src/life.py」），既无价值又浪费轮次。

    判定：子智能体经 run_subagent 派发时，ctx.extra 必带 subagent_allowed；
    顶层执行体的 ActionContext 不带该字段。因此「无 subagent_allowed 的 agent 调用」
    一律视为顶层甩锅，拦截。plan 子智能体的合法嵌套派发不受影响。
    """
    if ctx.action == "agent" and "subagent_allowed" not in ctx.extra and not config.EXECUTOR_ALLOW_AGENT:
        return ("agent_blocked: 顶层执行体（Executor）禁止派发子智能体——子智能体没有你的任务上下文，"
                "无法独立完成工作。请直接使用 write_file / edit_file / shell 自己完成实现与测试，"
                "不要转交给他人。")
    return None


def _hook_shell_before(ctx: ActionContext, params: Dict[str, Any], result: Optional[str]) -> Optional[str]:
    """shell 的 before 钩子：支持 run_in_background 走后台任务。"""
    if params.get("run_in_background"):
        return run_shell_bg(params.get("cmd", ""))
    return None


def _hook_write_protection(ctx: ActionContext, params: Dict[str, Any], result: Optional[str]) -> Optional[str]:
    """write_file/edit_file 的 before 钩子：接手模式保护（禁止覆盖受保护测试/继承实现）。"""
    path = params.get("path", "")
    prot = _check_takeover_protection(path) or _check_impl_overwrite_protection(path)
    if prot:
        logger.info('%s', f'🛡️ 已拦截：模型试图覆盖受保护文件 {_safe_rel(path)}')
        return prot
    return None


def _is_executor(ctx) -> bool:
    """自动完成任务等 BUILD 层副作用只在【顶层 Executor】派发的动作上生效；
    只读角色（tester / analyzer）绝不允许因 read/shell 而推进编码任务状态。"""
    return bool(ctx and getattr(ctx, "extra", None) and ctx.extra.get("role") == "executor")


def _append_task_done(res: str) -> str:
    """write/edit 成功后自动把当前 pending 任务标记完成（不做交付物检查，完成标准只看测试）。"""
    task = mark_current_task_done()
    if task and task.get("status") == "done":
        res += f" | 自动完成任务：{task['desc']}"
    return res


def _hook_write_after(ctx: ActionContext, params: Dict[str, Any], result: str) -> Optional[str]:
    """write_file 的 after 钩子：LSP 提示（仅成功时）+ Go 模块兜底 + 自动完成任务（仅 Executor）。"""
    if result.startswith("write_success"):
        _lsp = _auto_lsp_hint(params.get("path", ""))
        if _lsp:
            result += _lsp
        _harness._ensure_go_module()
    if _is_executor(ctx):
        result = _append_task_done(result)
    return result


def _hook_edit_after(ctx: ActionContext, params: Dict[str, Any], result: str) -> Optional[str]:
    """edit_file 的 after 钩子：LSP 提示（仅成功时）+ Go 模块兜底 + 自动完成任务（仅 Executor）。"""
    if result.startswith("edit_success"):
        _lsp = _auto_lsp_hint(params.get("path", ""))
        if _lsp:
            result += _lsp
        _harness._ensure_go_module()
    if _is_executor(ctx):
        result = _append_task_done(result)
    return result


def _hook_read_after(ctx: ActionContext, params: Dict[str, Any], result: str) -> Optional[str]:
    """read_file 的 after 钩子：自动完成任务【仅限 Executor】。
    读是只读动作，tester/analyzer 的 read 绝不能推进编码任务状态（否则会污染任务进度）。"""
    if _is_executor(ctx):
        return _append_task_done(result)
    return result


def _hook_shell_after(ctx: ActionContext, params: Dict[str, Any], result: str) -> Optional[str]:
    """shell 的 after 钩子：非错误时自动完成任务（仅 Executor；已关闭测试反馈回灌）。"""
    if not result or result.startswith("shell_error"):
        return result
    if _is_executor(ctx):
        result = _append_task_done(result)
    return result


# ---- 把钩子挂到具体工具（有就执行，没有就不执行）----
ToolRegistry.get("write_file").before.append(_hook_write_protection)
ToolRegistry.get("write_file").after.append(_hook_write_after)
ToolRegistry.get("edit_file").before.append(_hook_write_protection)
ToolRegistry.get("edit_file").after.append(_hook_edit_after)
ToolRegistry.get("read_file").after.append(_hook_read_after)
ToolRegistry.get("shell").before.append(_hook_shell_before)
ToolRegistry.get("shell").after.append(_hook_shell_after)

# ---- 全局钩子（对所有动作生效）----
ToolRegistry.register_global_hook("before", _hook_count_action)
ToolRegistry.register_global_hook("before", _hook_plan_mode_guard)
ToolRegistry.register_global_hook("before", _hook_subagent_whitelist)
ToolRegistry.register_global_hook("before", _hook_block_executor_agent)


def execute_actions(action_objs: List[Dict[str, Any]],
                    messages: Optional[List[Dict[str, Any]]] = None) -> str:
    results = []
    for a in action_objs:
        results.append(execute_action(a, messages))
    return "\n".join(results)


# ======================================================================
# 主循环（PDCA）—— 忠实移植 demo.py run_agent
# ======================================================================
def run_agent(user_task: str, messages: Optional[List[Dict[str, str]]] = None,
              resume_state: Optional[Dict[str, Any]] = None,
              truth: "Optional[List[Dict[str, Any]]]" = None) -> List[Dict[str, str]]:
    """跑一轮完整的 PDCA 循环（三层架构编排）：

    L3 派发子任务 → L1 模型产出结构化动作 → L2 fences 执行并观察 → L3 目标检测。
    层间契约见 swe_agent/contracts.py；跨切面逻辑走 registry 的循环级 hook
    （pre_loop/post_loop/on_error）；每层吐结构化遥测事件（layers.emit_event）。
    """
    # —— 初始化：系统状态（与模型上下文严格分离，见 RunState / ContextManager）——
    if messages is None:
        reset_state()
        GLOBAL_STATE["lang"] = "python"  # 默认；Planner 决策后会被覆盖（Planner 不确定 → 默认 python）
        # 只读分析阶段（Analyzer，tools-only）：探查代码库现状，产出 findings 喂给 Planner。
        # 失败（模型不可用 / lmstudio 未起）返回空串，Planner 直接进入规划，不影响主流程。
        research_findings = run_analyzer(user_task)
        if research_findings:
            logger.info('%s', f'[analyzer] 调研发现（{len(research_findings)} 字符）已注入 Planner。')
        else:
            logger.info('%s', '[analyzer] 未产出可用调研（模型不可用/未配置），Planner 直接进入规划。')
        run_planner(user_task, research_findings=research_findings)
        messages = [{"role": "system", "content": build_system_prompt(user_task, lang=GLOBAL_STATE["lang"])}]
    else:
        if resume_state:
            GLOBAL_STATE.update({k: v for k, v in resume_state.items() if k != "goal"})
        pending = [t for t in GLOBAL_STATE.get("tasks", []) if t.get("status") == "pending"]
        if not pending:
            GLOBAL_STATE["tasks"] = []
            GLOBAL_STATE["planning_done"] = False
            GLOBAL_STATE["done_list"] = []
        messages[0] = {"role": "system", "content": build_system_prompt(lang=GLOBAL_STATE["lang"])}
        messages.append({"role": "user", "content": f"# 新任务（继续既有会话）\n{user_task}"})
    GLOBAL_STATE["goal"] = user_task if user_task else GLOBAL_STATE.get("goal", "")
    GLOBAL_STATE.setdefault("current_focus", "理解需求并规划")
    _reset_run_telemetry()
    layers.reset_layer_events()

    # 「接手既有代码」模式
    _pre_tests = _snapshot_existing_tests()
    GLOBAL_STATE["protected_tests"] = set(_pre_tests)
    GLOBAL_STATE["takeover_mode"] = _detect_takeover_mode(user_task, _pre_tests)
    system_extras = ""
    if GLOBAL_STATE["takeover_mode"]:
        # 继承来的「实现文件」也一并保护：禁止整体覆盖，只允许在其上 edit_file 最小修复。
        _pre_impl: List[str] = []
        try:
            for p in config.WORKSPACE.rglob("*"):
                if p.is_file():
                    r = _safe_rel(str(p))
                    if r and not _is_test_file(r):
                        _pre_impl.append(r)
        except Exception:
            pass
        GLOBAL_STATE["protected_impl"] = set(_pre_impl)
        logger.info('%s', f"🛡️ 接手模式已启用：保护 {len(_pre_tests)} 个既有测试夹具（禁止覆盖）—— {', '.join(_pre_tests)}")
        system_extras += (
            "\n\n# ⚠️ 本次是「中途接手既有代码」任务（不是从零写新项目）\n"
            "工作区里已经存在【既有测试】（tests/ 目录），它们是验收标准 / 标准答案，受系统保护，\n"
            "【严禁用 write_file 或 edit_file 修改/覆盖它们】。你的职责是读懂现有实现 → 定位 bug → "
            "用 edit_file 最小改动修复实现代码，让既有测试由红变绿。"
        )

    _memory_note = build_memory_note()
    if _memory_note:
        system_extras += "\n\n" + _memory_note
    if system_extras:
        messages[0] = {"role": "system", "content": messages[0]["content"] + system_extras}

    # 模块级去重集合与循环共享状态（系统状态，非模型上下文）
    global _drift_reported
    _drift_reported = set()
    ctx = RunState(max_iter=config.MAX_ITER)
    # 恢复会话「真相日志」（原文，未压缩）：messages 是压缩后的工作集（直接复用，不重新压缩），
    # truth 是完整会话史，供后续 recall / 「会话历史不丢失」。ctx.cm 随后由 Agent 接管，
    # 其 reset() 只改 _msgs 不动 _truth，故此处写入会在整个 run 期间存活。
    if ctx.cm is None:
        from .management import ContextManager
        ctx.cm = ContextManager()
    if truth:
        ctx.cm._truth = list(truth)

    # 重置 harness 内的「必然失败」校验状态（活变量）。
    # 注意：原先清空的是 config.* 的死副本，harness 实际读取的是自身模块级变量，
    # 导致循环闸门历史跨任务不清零。现改为重置 harness.* 本体。
    _harness._VAL_SIG_HIST = []
    _harness._VAL_DOOMED_STREAK = 0
    _harness._SELFHEAL_DONE = set()

    # 循环级 hook：pre_loop（GLOBAL 机制，补充③）
    ToolRegistry.run_loop_hooks("pre_loop", ctx)

    # =====================================================================
    # 三层嵌套循环（吸收 forge 的 OpenAI 原生 toolcall 协议 + 独立只读 tester）：
    #   loop_1(attempt): 整体重试；耗尽仍未通过则结束（limit_reached）
    #     loop_2(round): 编码重试；lint 不过则重编码
    #       loop_3(step): executor 原生 toolcall 循环（assistant/tool 交替，不拍平）
    #     -> lint -> pytest(单杠校验) -> tester(独立验收；全过则 break loop_1)
    # 详见 _run_nested_loops → build_executor_agent（executor 循环逻辑收敛进 agent._apply_toolcall）。
    # =====================================================================
    _run_nested_loops(messages, ctx)

    logger.info('%s', '\nAgent 运行结束。')
    logger.info('%s', f'[stats] {stats_summary()}')
    if config.SESSION_ID:
        save_session(config.SESSION_ID, messages,
                     truth=ctx.cm.truth_list() if ctx.cm else None)
        logger.info('%s', f'[session] 已保存：{config.SESSION_ID}（--session {config.SESSION_ID} 可恢复）')

    try:
        _fv = GLOBAL_STATE.get("final_validation") or {}
        _passed = bool(_fv.get("passed"))
        _verdict = ("success" if _passed
                    else "early_stopped" if ctx.early_stop
                    else "unsolvable" if ctx.metadata.get("unsolvable") else "failed")
        _lesson = _derive_lesson(GLOBAL_STATE.get("goal", ""), _passed, ctx.early_stop)
        GENERIC_FAIL = "任务未在预算内通过校验，建议复盘测试接口与实现是否对齐。"
        # 不落盘「无洞察力的通用失败结论」：它仅会在后续每次 run 的 system prompt 里
        # 注入一条形如 [ts] failed：… 的日志式噪声，既误导模型又无复用价值。
        if _lesson and _lesson != GENERIC_FAIL:
            append_agent_memory({
                "ts": time.strftime("%Y-%m-%d %H:%M"),
                "task": (GLOBAL_STATE.get("goal") or "")[:160],
                "verdict": _verdict,
                "telemetry": list(RUN_TELEMETRY),
                "lesson": _lesson,
            })
    except Exception as e:
        logger.info('%s', f'[memory] 收尾沉淀失败：{e}')

    # 循环级 hook：post_loop
    ToolRegistry.run_loop_hooks("post_loop", ctx)
    return messages

# 供 run_agent 使用的模块级漂移去重集合
_drift_reported: set = set()


# ======================================================================
# 三层嵌套循环（重构：委托给统一 Agent + 嵌套 LoopConfig）
# 三层 loop 实为同一 executor 角色在三种 LoopConfig 粒度上的嵌套（外层包外层，
# 只差 max_iter + 收尾闸门 + 反馈回灌），不是三个不同角色。
# ======================================================================
def _l2_start(ctx, it):
    """loop_2 每 round 起始：追加当前任务上下文；重置循环防护；长度感知压缩；最后 load 工作模型。

    时序铁律（防显存爆炸）：先压缩 context（用 LFM 副驾，压缩完即自卸），
    再 load 本 loop 的工作模型。绝不两个模型同时驻留。
    """
    cur = get_current_task()
    subtask = (cur["desc"] if cur else
               "所有规划任务均已标记完成；请确认工作区测试全部通过（pytest 全绿），"
               "必要时补写/修复测试与实现。")
    ctx.cm.append("user", build_context(subtask))
    # 每层 round 重置循环防护状态由 Agent._run_loop 按 hook_point 自动调 guard.reset_at 完成
    # （L2_LOOP_START 清零 stall 三件套），不再在此手工敲 reset_guard。
    # 长度感知上下文压缩（每 round 一次，与原 _sidecar.auto_compact 位置一致）
    ctx.cm.compress_if_needed(ctx)
    # 压缩已用 LFM 完成且 LFM 已自卸；此刻才 load 本 loop 工作模型（GLM 等远程模型 load_unload=False 跳过）
    if ctx.metadata.get("exec_load_unload"):
        from .management import ModelManager
        ModelManager.load(ctx.metadata["exec_model"])


def _l2_gate(ctx, reason):
    """loop_2 收尾闸门：lint 不过→续轮；过→出 round 进 pytest。model_error→提前终止。

    时序铁律（防显存爆炸）：本 round 结束即卸载本 loop 工作模型（load 在 _l2_start 完成，
    且 _l2_start 已先压缩 context（LFM 自卸）再 load，保证压缩副驾与工作模型绝不同时驻留）。
    """
    if reason == "model_error":
        ctx.early_stop = True
        logger.info('%s', '[early-stop] executor 模型不可达，提前终止。')
        _snapshot_harness_facts()
        if ctx.metadata.get("exec_load_unload"):
            from .management import ModelManager
            ModelManager.unload(ctx.metadata["exec_model"])
        return "break"
    lang = GLOBAL_STATE.get("lang", "python")
    lint_v, lint_d, _ = _harness.run_lint(lang)
    logger.info('%s', f'[lint] {lint_v}: {lint_d[:300]}')
    if lint_v == "fail":
        # L2 lint 连续失败熔断（2026-09-02）：弱模型写不出可编译代码时，loop_2 会陷入
        # 「lint失败→重编码→lint失败」空转（fizzbuzz e2e 实测 lfm2.5-2.6b 把 print("\n".join)
        # 写成真实换行，语法永不可过，烧满 MAX_ROUNDS×MAX_ATTEMPTS 才判 failed）。
        # 复用 BUILD 常量 VAL_DOOMED_THRESHOLD(=2)：连续 lint 失败达阈值即判 unsolvable，
        # 提前终止，避免把预算烧在「当前模型无法逾越的墙」上。lint 通过即清空（保持「连续」语义）。
        ctx.guard.tick("lint_consec_fail")
        _streak = ctx.guard.value("lint_consec_fail")
        if _streak >= config.VAL_DOOMED_THRESHOLD:
            # 连续 lint 失败达阈值 → emit VALIDATION_FAIL，由 UnsolvableGuard 统一判 unsolvable 并 BREAK_LOOP
            logger.info('%s', f'⚠️ 连续 {_streak} 次 lint 校验失败（达 VAL_DOOMED_THRESHOLD={config.VAL_DOOMED_THRESHOLD}）→ 判定 unsolvable，提前终止本任务。')
            dec = HOOK_HUB.first_block_decision(
                HOOK_HUB.emit(HookPoint.VALIDATION_FAIL, ctx=ctx, detail=lint_d[:300]))
            if dec is not None and dec.action == GateAction.BREAK_LOOP:
                return "break"
        ctx.cm.append("user",
            f"⚠️ 静态校验（lint）未通过，请修复以下问题后继续编码：\n{lint_d}")
        return "continue"
    # lint 通过 → 连续 lint 失败计数清空（保持「连续」语义）；出 round 进 pytest 前卸载模型
    ctx.guard.reset("lint_consec_fail")
    if ctx.metadata.get("exec_load_unload"):
        from .management import ModelManager
        ModelManager.unload(ctx.metadata["exec_model"])
    return "break"


def _l1_start(ctx, it):
    """loop_1 每 attempt 起始：第 2 次起回灌上一轮验收反馈；轻量漂移检测。"""
    if it > 1:
        fb = ["⚠️ 上一轮 attempt 未通过验收。请基于下方反馈修复实现与测试，"
              "不要重写任务清单、不要重规划："]
        if ctx.metadata.get("last_bar"):
            fb.append(f"[单杠校验] {ctx.metadata['last_bar']}")
        if ctx.metadata.get("last_verify"):
            fb.append(f"[独立验收] {ctx.metadata['last_verify']}")
        ctx.cm.append("user", "\n".join(fb))
    # 漂移检测（每 attempt 一次，轻量，仅记录不阻断）；计数收口进 guard（drift_injections）
    if ctx.guard.value("drift_injections") < config.DRIFT_MAX_INJECTIONS:
        _new = drift_issues()
        if _new:
            ctx.guard.tick("drift_injections")
            _drift_reported.update(_new)
            _telemetry("plan_drift")
            logger.info('%s', f"⚠️ 漂移检测：发现 {len(_new)} 个契约外文件（仅记录）：{', '.join(_new)}")


def _l1_gate(ctx, reason):
    """loop_1 收尾闸门：pytest 单杠 → tester 独立验收 → done/continue。

    若内层闸门已置 early_stop（如 _l2_gate 的 model_error），直接收尾，不再跑 pytest/tester
    （模型已不可达，验收无意义，且避免回灌误导信息）。
    """
    if ctx.early_stop:
        logger.info('%s', '\n[early-stop] 内层已触发提前终止，跳过 pytest/tester 收尾。')
        return "break"
    logger.info('%s', f'\n===== loop_1 单杠校验（pytest）=====')
    bar_status, bar_detail, bar_out = _harness._run_test_bar(ctx.cm.to_list())
    ctx.metadata["last_bar"] = f"{bar_status}: {bar_detail}"
    if bar_status == "no_tests":
        # P1 扩展（2026-09-02 e2e 验证）：「无测试文件」也是「未过单杠」的一种，须纳入 doomed
        # 连续计数——弱模型会在「写了但 1 失败」(fail) 与「根本没写测试」(no_tests) 间摆动，若只把
        # fail 计入、no_tests 清零，连续链会被反复重置、永远凑不到阈值 → 烧满 MAX_ATTEMPTS 才
        # 判 failed。统一口径：只有 bar 真正 pass 才清零（见下方），fail 与 no_tests 都累加。
        ctx.guard.tick("bar_consec_fail")
        _streak = ctx.guard.value("bar_consec_fail")
        if _streak >= config.VAL_DOOMED_THRESHOLD:
            # 连续单杠未过（含无测试文件态）达阈值 → emit VALIDATION_FAIL，由 UnsolvableGuard 统一判 unsolvable 并 BREAK_LOOP
            logger.info('%s', f'⚠️ 连续 {_streak} 次单杠未过（含无测试文件态，达 VAL_DOOMED_THRESHOLD={config.VAL_DOOMED_THRESHOLD}）→ 判定 unsolvable，提前终止本任务。')
            dec = HOOK_HUB.first_block_decision(
                HOOK_HUB.emit(HookPoint.VALIDATION_FAIL, ctx=ctx, detail="工作区无测试文件"))
            if dec is not None and dec.action == GateAction.BREAK_LOOP:
                return "break"
        logger.info('%s', 'ℹ️ 工作区尚无测试文件，提示 executor 补写测试后重试。')
        ctx.cm.append("user",
            "ℹ️ 当前工作区还没有任何测试文件。请编写 test_*.py（Python）并用 pytest 跑通，"
            "测试全部通过后才算完成。不要只写实现不写测试。")
        return "continue"
    if bar_status == "fail":
        # P1（2026-09-02）：连续 pytest 失败熔断 —— 弱模型反复写错测试期望/实现时，
        # loop1 重跑只会无限空耗（fizzbuzz e2e 实测 executor 卡空转直到 1800s 被 SIGKILL）。
        # 复用 BUILD 常量 VAL_DOOMED_THRESHOLD(=2)：连续失败达阈值即判 unsolvable，提前终止本
        # 任务，避免把预算烧在「当前模型无法逾越的墙」上。signature-agnostic（任意连续失败都计），
        # 与旧的「同根因签名」意图一致，但更稳健、不依赖未接线的 _VAL_DOOMED_STREAK 死代码。
        ctx.guard.tick("bar_consec_fail")
        _streak = ctx.guard.value("bar_consec_fail")
        if _streak >= config.VAL_DOOMED_THRESHOLD:
            # 达阈值时也必须落 detail：否则「为什么判失败」在日志里完全不可见（2026-09-08 教训）。
            # 连续 pytest 失败达阈值 → emit VALIDATION_FAIL，由 UnsolvableGuard 统一判 unsolvable 并 BREAK_LOOP
            logger.info('%s', f'⚠️ 连续 {_streak} 次 pytest 单杠失败（达 VAL_DOOMED_THRESHOLD={config.VAL_DOOMED_THRESHOLD}）→ 判定 unsolvable，提前终止本任务。')
            logger.info('%s', f'   末次单杠 detail：{bar_detail}')
            if bar_out:
                logger.info('%s', '   末次单杠输出尾部：\n' + "\n".join(
                    (bar_out or "").splitlines()[-40:]))
            dec = HOOK_HUB.first_block_decision(
                HOOK_HUB.emit(HookPoint.VALIDATION_FAIL, ctx=ctx, detail=f"pytest 单杠失败：{bar_detail}"))
            if dec is not None and dec.action == GateAction.BREAK_LOOP:
                return "break"
        tail = bar_out or ""
        logger.info('%s', f'⚠️ 单杠校验失败（{bar_detail}）。回灌失败输出，进入下一轮 attempt。')
        ctx.cm.append("user",
            f"⚠️ 单杠校验失败（{bar_detail}）。请 read_file 定位并修复，重跑直到全绿：\n{tail}")
        return "continue"
    # pytest 通过 → 连续失败计数清零（仅在真正跑通时，保持「连续」语义）
    ctx.guard.reset("bar_consec_fail")
    # 完成判定 = 单杠三件套齐全：
    #   ① lint（每 round 出 round 前的 lint 闸门，见 _l2_gate）—— 静态校验
    #   ② 模型自生成 unittest（pytest 单杠，即上面的 _run_test_bar）—— 确定性执行校验
    #   ③ tester 自然语言验收（verify_gate）—— 独立只读 agent 逐条核对验收点并引用证据
    # 三者任一不过，都不算完成；tester 是单杠的一部分，绝不能摘掉当非阻断信号。
    # （早期曾误把 tester 降级成非阻断以绕开弱模型空转，那等于把三件套拆成两件套，是错误的。）
    verdict, detail = _verify.verify_gate()
    ctx.metadata["last_verify"] = f"{verdict}: {detail}"
    if verdict == "pass":
        logger.info('%s', '✅ 单杠三件套齐全（lint + unittest + tester 验收）→ Agent 完成。')
        _snapshot_harness_facts()
        return "done"
    # no_points：planner 未产出验收点，无自然语言验收标准可验 → 不阻断（确定性 pytest 已通过）。
    if verdict == "no_points":
        logger.info('%s', 'ℹ️ 单杠 pytest 通过，且未生成验收点（no_points）→ 视为完成（tester 无标准可验）。')
        _snapshot_harness_facts()
        return "done"
    # skipped：tester 未产出验收结果（模型 finish_reason=stop 未提交 finish_verify / 或
    # FORGE_TESTER_MODEL=off 关闭验收）。仅确定性 pytest 闸门生效，skipped 不触发 loop1 重跑，
    # 直接视为单杠完成——避免弱模型空转到 max_iter 才返 fail 的死循环（见 2026-09-02 修复）。
    if verdict == "skipped":
        logger.info('%s', '⚠️ 单杠 pytest 通过，但 tester 独立验收被跳过（skipped）：tester 未产出验收结果。按约定 skipped 不重跑 attempt，直接视为单杠完成（仅确定性 pytest 闸门生效）。')
        _snapshot_harness_facts()
        return "done"
    # 其余（fail / error）= 单杠未齐，继续 attempt 重试。
    #   - fail：tester 发现失败用例（failed_case>0）→ 回灌反馈，由 loop_1 重试预算兜底修复；
    #   - error：tester 模型不可达（model_error）→ 回灌并进入下一轮 attempt（重试预算兜底）。
    # 注意：tester 是单杠的一部分，绝不能摘掉当非阻断信号；但只有明确的失败用例才令 loop1 重跑，
    # skipped / 漏提交点不计入失败（见 verify._normalize_results）。
    if verdict == "error":
        logger.info('%s', f'⚠️ 单杠 pytest 通过，但 tester 模型不可达（error）。回灌并进入下一轮 attempt（重试预算兜底）。')
    else:
        logger.info('%s', f'⚠️ 单杠 pytest 通过，但 tester 发现失败用例（failed_case>0）：{detail}')
    ctx.cm.append("user",
        f"⚠️ 独立验收（tester）未通过（{verdict}）：{detail}。"
        "存在失败用例，请检查实现是否真正满足验收点（verify_points），修复后重新跑测试与验收。")
    return "continue"


def build_executor_agent(ctx: Any = None):
    """构造 executor 的统一 Agent：executor 角色 + 三层嵌套 LoopConfig（L1(L2(L3))）。

    - L3：单步 toolcall（Agent._step，循环防护在 agent._apply_toolcall）
    - L2：round，收尾 lint 闸门；on_iter_start 追加 subtask 上下文 + 长度感知压缩
    - L1：attempt，收尾 pytest+tester 闸门，第 2 次起回灌反馈

    模型生命周期（load/unload 时序）走**手动**控制，而非 Agent 的 pre_loop/post_loop 钩子：
    因为时序要求「先压缩 context（LFM 副驾，压缩完即自卸）→ 再 load 工作模型」，不能在
    pre_loop 一进去就 load。由 _l2_start 在压缩后 load、_l2_gate 在出 round 时 unload，
    通过 ctx.metadata["exec_load_unload"]/["exec_model"] 驱动
    （GLM/远程 = False → 跳过；本地 lmstudio = True → 每 round 装卸一次，显存不长驻）。

    ctx 可选：传入则复用调用方的 RunState（run_agent / dbg 的单一运行态），保证 early_stop
    等状态直接写在共享对象上，无需回写；不传则自建 RunState。RunState.cm（ContextManager）
    拥有对话 buffer + 循环防护状态，由 Agent.run(messages) 注入初始对话。
    """
    from .agent import Agent, LoopConfig, RunState
    from .roles_config import make_role_config

    rc = make_role_config("executor")
    rc.result_compress = _sidecar.compress_content  # 工具结果压缩（fail-open）
    # L3：单步（hook_point 决定 stall 限制的重置锚点；L3 层每步清零 tool_call_count）
    l3 = LoopConfig(max_iter=config.MAX_STEPS, hook_point=HookPoint.L3_LOOP_START)
    # L2：round（每 round 起始重置 stall 三件套 consec_repeat/no_tool_streak/empty_streak）
    l2 = LoopConfig(max_iter=config.MAX_ROUNDS, child=l3, hook_point=HookPoint.L2_LOOP_START,
                    on_iter_start=_l2_start, on_iter_end=_l2_gate)
    # L1：attempt（熔断计数 bar/lint/drift 的 reset_at=RUN_START，跨 attempt 累计、不逐轮清零）
    l1 = LoopConfig(max_iter=config.MAX_ATTEMPTS, child=l2, hook_point=HookPoint.L1_LOOP_START,
                    on_iter_start=_l1_start, on_iter_end=_l1_gate)
    # 模型生命周期元数据：exec_load_unload 决定 _l2_start/_l2_gate 是否装卸工作模型
    if ctx is None:
        ctx = RunState(role="executor")
    ctx.role = ctx.role or "executor"
    ctx.metadata.setdefault("last_bar", "")
    ctx.metadata.setdefault("last_verify", "")
    ctx.metadata.setdefault("exec_load_unload", M.role_load_unload("executor"))
    ctx.metadata.setdefault("exec_model", rc.model_id())
    return Agent(rc, l1, ctx=ctx)


def _run_nested_loops(messages: List[Dict[str, Any]], ctx: Any) -> None:
    """三层嵌套循环主驱（重构：委托给统一 Agent + 嵌套 LoopConfig）。

    行为与原实现一致：loop_1 attempt → loop_2 round → loop_3 executor toolcall →
    lint → pytest → tester；tester 全过则 Agent 完成，否则换 attempt 重来。

    所有控制流（含模型 load/unload 时序、上下文压缩、循环防护、各层闸门）均已收敛进
    Agent（swe_agent/agent.py）与 _l2_start / _l2_gate / _l1_start / _l1_gate 回调，
    此处只做装配 + 入口调用。

    调用方与 Agent 共享同一个 RunState 对象，early_stop / metadata 直接写在其上，无需回写。
    """
    agent = build_executor_agent(ctx)
    agent.run(messages)


def _derive_lesson(goal: str, passed: bool, early_stop: bool) -> str:
    """从遥测信号生成一句可复用经验（避免重复 from .state 的循环依赖，本地实现）。"""
    tags = RUN_TELEMETRY
    if early_stop:
        return "运行被停滞/循环防护提前终止，模型未能在预算内收敛（可能卡在读/空输出循环）。"
    if "doom_guard" in tags:
        return "测试以相同根因反复失败，触发必然失败防护——需换根本方案而非小修小补。"
    if passed:
        return "任务通过技术栈校验（pytest 等），交付物可用。"
    return "任务未在预算内通过校验，建议复盘测试接口与实现是否对齐。"


# ======================================================================
# 入口
# ======================================================================
DEFAULT_TASK = """# 技术栈：python, pytest, OOP
# 目标任务：
开发命令行版的「康威生命游戏（Conway's Game of Life）」，默认棋盘 20x20。

规则：
- 棋盘：二维网格，每个格子为存活(1)或死亡(0)。
- 邻居：每个细胞周围 8 个格子。
- 迭代规则（每 tick 同时更新所有细胞）：
  1. 活细胞邻居数 == 2 或 3 → 下一代存活；
  2. 活细胞邻居数 < 2（孤独）或 > 3（拥挤）→ 下一代死亡；
  3. 死细胞邻居数 == 3 → 下一代复活。
- 实现要求：
  - 一个 GameOfLife 类（__init__ 可指定尺寸，默认 20x20；可随机初始化或给定初始矩阵）。
  - step() 方法推进一代；display() 方法以文本打印当前棋盘。
  - 提供一个命令行入口：python game_of_life.py 可直接运行并演示若干代演化。
- 测试要求：编写 test_game_of_life.py，覆盖邻居计数、单步演化、边界处理等，并用 pytest 运行通过。
"""


def _resolve_executor_arg(val: str) -> str:
    """把 --executor 参数解析成 models.MODELS 里的 catalog id。

    兼容旧的三选一别名（local/glm/lmstudio），也接受直接传 catalog 模型 id。
    """
    val = (val or "local").lower()
    # local/plaintext provider 已废弃，统一走 lmstudio 上的 qwen 执行器；
    # glm/lmstudio 别名保留向后兼容。
    _LEGACY = {"local": "qwen2.5.1-coder-7b-instruct",
               "glm": "glm-4.7",
               "lmstudio": "liquid/lfm2.5-1.2b"}
    if val in _LEGACY:
        return _LEGACY[val]
    if val in M.MODELS:
        return val
    logger.info('%s', f'[warn] --executor {val!r} 不在 Model 目录，回退 qwen2.5.1-coder-7b-instruct。')
    return "qwen2.5.1-coder-7b-instruct"


def main():
    import swe_agent.config as C
    parser = argparse.ArgumentParser(description="SWE-Agent 分层调度 + 插件化 Harness（自描述工具）")
    parser.add_argument("task", nargs="*", help="开发任务描述；省略则用默认任务")
    parser.add_argument("--no-autocompact", action="store_true", help="关闭自动对话压缩")
    parser.add_argument("--context-window", type=int, default=None, help="覆盖有效上下文窗口 token 数")
    parser.add_argument("--session", default=None, help="恢复指定会话继续")
    parser.add_argument("--list-sessions", action="store_true", help="列出已保存的会话后退出")
    parser.add_argument("--yolo", action="store_true", help="放行危险命令（危险！）")
    parser.add_argument("--fresh", action="store_true",
                        help="启动前清空工作区（默认保留，避免误删 agent_sandbox 等目录）")
    parser.add_argument("--model", default=None, help="覆盖使用的模型名")
    parser.add_argument("--lsp-server", default="pylsp", help="指定 LSP 服务器命令")
    parser.add_argument("--no-glm", action="store_true", help="禁用远程 GLM 顶层（Executor 纯本地自规划）")
    parser.add_argument("--executor", default=None, choices=("local", "glm", "lmstudio"),
                        help="覆盖 Executor 后端：local（rapid-mlx）/ glm（远程）/ lmstudio（LM Studio 本地服务）")
    parser.add_argument("--enable-plugin-mcp", action="store_true",
                        help="连接插件声明的 MCP 服务器（默认不连，避免未授权外连）")
    # L1/L2/L3 三层 loop 上限（unattend 模式可由命令行覆盖；当前 config 值作默认）
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="覆盖 L1(attempt) 循环上限（默认 config.MAX_ATTEMPTS）")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="覆盖 L2(round) 循环上限（默认 config.MAX_ROUNDS）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="覆盖 L3(step) 循环上限（默认 config.MAX_STEPS）")
    args = parser.parse_args()

    global GLOBAL_STATE  # noqa
    if args.executor:
        C.EXECUTOR_MODEL = _resolve_executor_arg(args.executor)
    if args.no_glm:
        # 关闭远程 GLM 顶层：planner 不挂模型 → 优雅降级（planner→本地自规划）。executor 不受影响。
        C.PLANNER_MODEL = ""
    if args.context_window and args.context_window > 0:
        C.CONTEXT_WINDOW = args.context_window
    if args.no_autocompact:
        C.AUTO_COMPACT_ENABLED = False
    if args.yolo:
        C.YOLO_MODE = True
    if args.model:
        C.MODEL_OVERRIDE = args.model
    if args.lsp_server:
        os.environ["LSP_SERVER_CMD"] = args.lsp_server
    if args.enable_plugin_mcp:
        C.ENABLE_PLUGIN_MCP = True
    # L1/L2/L3 loop 上限覆盖（unattend 模式命令行传入，当前值作默认）
    if args.max_attempts and args.max_attempts > 0:
        C.MAX_ATTEMPTS = args.max_attempts
    if args.max_rounds and args.max_rounds > 0:
        C.MAX_ROUNDS = args.max_rounds
    if args.max_steps and args.max_steps > 0:
        C.MAX_STEPS = args.max_steps

    if args.list_sessions:
        ss = list_sessions()
        logger.info('%s', '\n'.join((f'  {s}' for s in ss)) if ss else '（暂无会话）')
        logger.info('%s', '恢复方式：python -m swe_agent <task> --session <id>')
        return

    resumed = None
    if args.session:
        resumed = load_session(args.session)
        if resumed is None:
            logger.info('%s', f'[session] 找不到会话 {args.session}（--list-sessions 查看）。')
            return
        C.SESSION_ID = args.session
        if resumed.get("stats"):
            C.STATS.update(resumed["stats"])
        logger.info('%s', f"[session] 恢复会话 {args.session}：{len(resumed.get('messages', []))} 条消息")

    # 工作区清空策略：默认【保留】（非破坏性），避免手动单跑时误删 agent_sandbox 等目录。
    # 需从零开始（消除跨任务/跨运行文件泄漏）时显式开启：命令行 --fresh 或 SWE_FRESH_WORKSPACE=1。
    # e2e 电池已显式设 SWE_FRESH_WORKSPACE=1 且自管隔离目录，翻转默认不影响电池行为。
    # --session 恢复（resumed is not None）时一律不清空。
    _fresh = bool(getattr(args, "fresh", False)) or os.environ.get("SWE_FRESH_WORKSPACE", "0") == "1"
    if resumed is None and _fresh:
        try:
            if C.WORKSPACE.exists():
                shutil.rmtree(C.WORKSPACE)
        except Exception as e:
            logger.info('%s', f'[workspace] 清理失败（忽略，继续）：{e}')
    C.WORKSPACE.mkdir(parents=True, exist_ok=True)

    # ---- 插件加载（启动时扫描配置的插件根目录）----
    # 让 harness 在启动即加载已配置的插件（技能/命令/子智能体/MCP），
    # 满足「启动时加载并使用配置好的 plugins」。设 ENABLE_PLUGINS=0 可关闭。
    # enable_mcp 默认 False（避免未授权外连），用 --enable-plugin-mcp 或 ENABLE_PLUGIN_MCP=1 开启。
    if C.ENABLE_PLUGINS:
        try:
            _plugins.load_plugins(C.PLUGINS_ROOT, enable_mcp=C.ENABLE_PLUGIN_MCP)
        except Exception as e:
            logger.info('%s', f'[plugins] 启动时加载失败（忽略，不影响核心闭环）：{e}')

    # ---- repo 根 mcp.json 启动期 eager-connect ----
    # 与插件层 .mcp.json（受 enable_mcp 门控）不同，REPO_ROOT/mcp.json 始终启用。
    # 此处启动期即连接：连成功/连失败当场报，让用户一眼确认 MCP 是否装好，
    # 不再只看到插件层那行误导性的「MCP 服务器 0 个（未启用）」。
    try:
        _mm = _mcp.get_mcp()
        if _mm is None or not _mm.clients:
            logger.info('%s', '[mcp] REPO_ROOT/mcp.json：无可连接服务器（配置缺失或解析失败）')
        else:
            _parts = [f"{n}({sum(1 for (s, _) in _mm.tools_index if s == n)}工具)" for n in _mm.clients]
            logger.info('%s', f"[mcp] 已连接 {len(_mm.clients)} 个服务器（REPO_ROOT/mcp.json，始终启用）：{', '.join(_parts)}")
    except Exception as e:
        logger.info('%s', f'[mcp] 启动期连接异常（忽略，运行时再试）：{e}')

    # 无人值守模式不启动 stdin 监听线程（避免后台进程下 readline 阻塞）
    # if not config.UNATTENDED_MODE:
    #     start_stdin_monitor()
    task = " ".join(args.task).strip()
    # 非交互（管道 / 后台 / 无参数）或无人值守时跑默认任务，避免静默进入 REPL 卡死
    if not task and (config.UNATTENDED_MODE or not sys.stdin.isatty()):
        task = DEFAULT_TASK
    if task:
        C.SESSION_ID = C.SESSION_ID or new_session_id()
        run_agent(task,
                  messages=resumed.get("messages") if resumed else None,
                  resume_state=resumed.get("state") if resumed else None,
                  truth=resumed.get("truth") if resumed else None)
    else:
        # REPL 模式：静音 console，避免 harness 启动/执行期日志污染交互界面；文件日志不受影响。
        # UNATTEND（task 分支）不进此分支，console 原样。interactive_repl 内部会再静音一次（幂等还原）。
        _saved_console = _repl_silence_console()
        try:
            interactive_repl(resumed)
        finally:
            _repl_restore_console(_saved_console)


def build_memory_note(max_entries: int = 8) -> str:
    """把最近的经验压缩成一段系统提示（空记忆返回 ''）。"""
    try:
        if config.AGENT_MEMORY_PATH.exists():
            data = json.loads(config.AGENT_MEMORY_PATH.read_text(encoding="utf-8"))
        else:
            data = []
    except Exception:
        data = []
    if not isinstance(data, list) or not data:
        return ""
    recent = data[-max_entries:]
    GENERIC_FAIL = "任务未在预算内通过校验，建议复盘测试接口与实现是否对齐。"
    lines = ["# 🧠 历史经验（跨任务记忆，仅供参考，勿盲从）"]
    for e in recent:
        # 只注入 lesson 正文：ts/verdict 仅用于落盘归档，绝不能渲染进 system prompt，
        # 否则会变成「[2026-08-30 12:10] failed：…」这种与运行时日志无法区分的噪声，
        # 既误导模型又让人工排查误以为是日志泄漏（实测踩过）。
        # 同时过滤掉「无洞察力的通用失败结论」——它只是把"失败"复述一遍，零复用价值。
        lesson = (e.get("lesson") or "").strip()
        if not lesson or lesson == GENERIC_FAIL:
            continue
        lines.append(f"- {lesson}")
    return "\n".join(lines) if len(lines) > 1 else ""


def append_agent_memory(entry: Dict[str, Any]) -> None:
    try:
        mem = []
        if config.AGENT_MEMORY_PATH.exists():
            mem = json.loads(config.AGENT_MEMORY_PATH.read_text(encoding="utf-8")) or []
        mem.append(entry)
        # 去重：若与上一条 lesson 完全一致，跳过，避免同一失败结论反复堆叠污染记忆。
        if len(mem) >= 2 and mem[-1].get("lesson") == mem[-2].get("lesson"):
            mem.pop()
        if len(mem) > config.AGENT_MEMORY_MAX:
            mem = mem[-config.AGENT_MEMORY_MAX:]
        config.AGENT_MEMORY_PATH.write_text(
            json.dumps(mem, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        logger.info('%s', f'[memory] 写入失败：{e}')


# ======================================================================
# 交互 REPL（无任务参数时进入）
# ======================================================================
_SLASH_HELP = """可用斜杠命令：
  /help      本帮助
  /task ...  给 agent 一个新开发任务
  /clear     清空会话重新开始
  /compact   立即压缩当前对话历史
  /stats     查看运行统计
  /sessions  列出已保存会话
  /exit      保存并退出
直接输入文本 = 新任务（在当前对话上下文中继续）。"""


@dataclass
class AgentDisplay:
    """REPL 富显示配置 + 可注入输出 sink。

    - sink: 文本输出通道（REPL 默认 print；未来可注入 logger.info / 富 UI 回调）。
    - show_thinking / show_tool_args / show_status: 是否渲染思维链 / 工具参数 / L3 状态行。
    UNATTEND 路径（run_agent）完全不构造本对象、也不调用 _print_repl_msg，
    这些开关对无人值守模式零影响；reasoning_content 在出向边界(prepare_messages)也被剥除。
    """
    show_thinking: bool = True
    show_tool_args: bool = False
    show_status: bool = True
    sink: Callable[[str], None] = print


def _print_repl_msg(m: Dict[str, Any], display: AgentDisplay) -> None:
    """把一条对话消息渲染到控制台（REPL 自有打印，不依赖 logger、不污染 UNATTEND 的 run_agent 路径）。

    思维链(reasoning_content)仅在此渲染；出向(prepare_messages)已剥除，UNATTEND 回发模型不带该字段。
    """
    out = display.sink
    role = m.get("role")
    if role == "assistant":
        if display.show_thinking and m.get("reasoning_content"):
            out(f"\n[thinking]\n{m['reasoning_content']}\n")
        if m.get("content"):
            out(f"\n[assistant]\n{m['content']}\n")
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {}).get("name", "?")
            args = tc.get("function", {}).get("arguments", "")
            if display.show_tool_args and args:
                out(f"  ↳ 调用工具：{fn} 参数：{args}")
            else:
                out(f"  ↳ 调用工具：{fn}")
    elif role == "tool":
        c = m.get("content") or ""
        if len(c) > 2000:
            c = c[:2000] + " …[已截断]"
        out(f"  [tool] {c}\n")


def _attach_repl_streaming(chat: Agent, display: AgentDisplay) -> None:
    """把 REPL 渲染器挂到 ChatAgent 的 buffer 上：每条消息一入 buffer 就立即打印，
    实现「边跑边出」——thinking / 工具调用 / 工具结果按发生顺序实时上屏，
    而不是等整轮 run() 跑完（可能几十次工具调用）再一次性吐一大块。

    UNATTEND 路径（run_agent）从不调用本函数 → ctx.cm.on_message 恒为 None → 零影响。
    注意：必须在「恢复历史灌完 buffer 之后」再挂，否则会把整段历史重打印一遍。
    """
    chat.ctx.cm.on_message = lambda m: _print_repl_msg(m, display)


def _drive_chat(chat: Agent, user_text: str, display: AgentDisplay) -> None:
    """驱动一个 ChatAgent 实例跑完一轮对话：喂入用户输入 → 调 Agent.run()（复用 L3 工具交换
    循环，chat_loop_gate 在 all_done/model_error/stuck 时收尾）。

    打印走 ctx.cm.on_message 增量钩子（由 _attach_repl_streaming 挂载），不再 run() 后批量打印。
    工具调用路径与普通 Agent 完全一致；打印由 REPL 负责，不进 Agent.run（UNATTEND 路径不受影响）。
    """
    _attach_repl_streaming(chat, display)
    chat.ctx.cm.append("user", user_text)
    chat.run()  # 不传 messages → 不清空 buffer，保留跨轮上下文


def _repl_silence_console() -> list:
    """REPL 会话期间把全局 logger 的 console handler 抬到 WARNING，避免 agent 执行期 DEBUG/INFO
    （如 agent.py:363 的 tool-call trace、插件启动 INFO）污染交互界面。文件日志（FileHandler）不受影响。
    UNATTEND 路径（run_agent）从不调用本函数，console 保持原级别。
    返回 [(handler, 原level), ...] 供还原。"""
    saved = []
    for h in _logmod.logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            saved.append((h, h.level))
            h.setLevel(logging.WARNING)
    return saved


def _repl_restore_console(saved: list) -> None:
    """还原 console handler 级别（REPL 退出时调用）。"""
    for h, lvl in saved:
        h.setLevel(lvl)


def interactive_repl(resumed: Optional[Dict[str, Any]] = None) -> None:
    reset_state()
    # 真实 TTY 的 stdin 是 TextIOWrapper，带 reconfigure；但 pytest 的 DontReadFromInput
    # 等假 stdin（以及部分管道/重定向环境）没有该方法，直接调用会抛 AttributeError。
    # 仅当方法存在时才重配置编码，避免非交互/被测环境崩溃（GLOBAL 健壮性修复）。
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding='utf-8')

    import swe_agent.config as C
    messages: Optional[List[Dict[str, str]]] = None
    # REPL 专用 ChatAgent：普通 Agent 实例，全量工具，自带打印由 _drive_chat 负责；
    # 与 UNATTEND 的 run_agent 路径完全隔离，互不影响。
    chat = make_chat_agent()
    # REPL 富显示配置：开 thinking / 工具参数 / L3 status；sink=print（不污染 UNATTEND 的 logger 路径）。
    # 若要更干净，可把 sink 换成 noop 或富 UI 回调。
    # sink 显式 flush：管道/重定向下 stdout 是块缓冲，不 flush 仍会表现为「攒到最后才输出」。
    display = AgentDisplay(show_thinking=True, show_tool_args=True, show_status=True,
                           sink=lambda s: print(s, flush=True))
    if resumed:
        messages = resumed.get("messages")
        if resumed.get("state"):
            GLOBAL_STATE.update({k: v for k, v in resumed["state"].items() if k != "goal"})
        # 把恢复的历史（非 system）灌进 chat buffer，保持多轮连续
        if messages:
            for m in messages:
                if m.get("role") in ("user", "assistant", "tool"):
                    chat.ctx.cm.append(m["role"], m.get("content"),
                                       **({"tool_call_id": m["tool_call_id"]} if m.get("role") == "tool" and m.get("tool_call_id") else {}))
    # 挂上增量渲染钩子：必须在「恢复历史灌完 buffer」之后挂，否则历史会被重打印一遍。
    _attach_repl_streaming(chat, display)
    C.SESSION_ID = C.SESSION_ID or new_session_id()
    # REPL 模式：logger 默认会写控制台，与对话混在一起；改为纯 print，保持 UNATTEND 路径（run_agent）不受影响
    # logger.info('%s', f'\n=== SWE-Agent · 交互模式（会话 {C.SESSION_ID}）===')
    # logger.info('%s', _SLASH_HELP + '\n')
    # logger.info('WORKSPACE:%s', config.WORKSPACE)
    # REPL 会话期间静音 console（tool-call DEBUG / 启动 INFO 不进终端）；对话/工具/思维链只经
    # _print_repl_msg 的 print 渲染。文件日志不受影响。UNATTEND（run_agent）不进本函数，console 原样。
    _saved_console = _repl_silence_console()
    while True:
        # line:str = None
        try:
            print("user>", end='')
            # line = input("agent> ").strip()
            line = sys.stdin.readline()
            # 去除换行符
            line = line.strip('\n\r')
            if not line or line == "":
                print('需要我做什么？\n')
                continue
        except KeyboardInterrupt:
            # REPL 模式：日志会污染控制台
            # logger.info('%s', '\n（Ctrl+C 中断输入；/exit 退出）')
            _repl_restore_console(_saved_console)
            return
        except EOFError:
            # REPL 模式：日志会污染控制台
            # logger.info('error')
            break
        
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/exit", "/quit", "/q"):
                print('正在退出。。。')
                break
            if cmd == "/help":
                display.sink(_SLASH_HELP)
            elif cmd == "/task":
                if not arg:
                    display.sink('用法：/task 任务描述')
                    continue
                messages = run_agent(arg, messages)
            elif cmd == "/clear":
                messages = None
                reset_state()
                chat = make_chat_agent()
                _attach_repl_streaming(chat, display)  # 新实例要重新挂钩子
                display.sink('已清空对话与任务状态（磁盘会话文件未删）。')
            elif cmd == "/compact":
                if not messages:
                    display.sink('当前没有对话可压缩。')
                    continue
                try:
                    new_msgs, st = _sidecar.manual_compact(messages)
                    messages[:] = new_msgs
                    display.sink(f"[压缩] {st['pre']} → {st['post']} tokens（节省 {st['pre'] - st['post']}）")
                except Exception as e:
                    display.sink(f'[压缩] 失败：{e}')
            elif cmd == "/stats":
                display.sink(f'[stats] {stats_summary()}')
            elif cmd == "/sessions":
                ss = list_sessions()
                display.sink('\n'.join((f'  {s}' for s in ss)) if ss else '（暂无会话）')
            else:
                display.sink(f'未知命令 {cmd}，/help 查看可用命令。')
            continue
        # 纯文本：交给 ChatAgent（全量工具 + 自带打印），不再走 run_agent（批处理/UNATTEND 路径）
        # _drive_chat(chat, line)
        chat.ctx.cm.append("user", line)
        # 输出经 ctx.cm.on_message 增量实时上屏（thinking/工具调用/工具结果按发生顺序），
        # 不再等 run() 跑完全部轮次后一次性批量打印。
        chat.run()  # 不传 messages → 不清空 buffer，保留跨轮上下文
    # 保存：优先 chat 对话 buffer，其次 /task 产出的 messages
    to_save = chat.ctx.cm.to_list() if len(chat.ctx.cm.to_list()) > 1 else messages
    if to_save and C.SESSION_ID:
        save_session(C.SESSION_ID, to_save,
                     truth=chat.ctx.cm.truth_list() if chat.ctx.cm else None)
        display.sink(f'[session] 已保存 {C.SESSION_ID}')
    display.sink(f'\n[stats] {stats_summary()}\n再见。')
    _repl_restore_console(_saved_console)


if __name__ == "__main__":
    main()
