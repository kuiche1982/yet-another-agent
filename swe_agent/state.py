#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/state.py —— 全局运行态 + 运行时辅助（从 demo.py 抽取）

包含：
- GLOBAL_STATE：任务/计划/完成列表等全局状态（调度层唯一输入源）
- 任务清单辅助、交付物校验（harness 级兜底）
- 「停滞/循环」防护指纹/签名/提示
- 跨 run 记忆、遥测、统计
- 路径/命令归一化小工具（供 tools 与 supervisor 共用）

所有「相对仓库根」的路径改用 config.REPO_ROOT。
"""

import os
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .config import (
    WORKSPACE, STATS, REPO_ROOT, OUTPUT_BUDGET, AGENT_MEMORY_PATH,
)


# ======================================================================
# 全局状态
# ======================================================================
def _default_state() -> Dict[str, Any]:
    """GLOBAL_STATE 的**唯一**默认值来源（模块加载 + reset_state 共用，两者永不漂移）。

    🔴 为什么必须在模块加载时就以默认值初始化、而不是 `GLOBAL_STATE = {}`：
    REPL 路径（`interactive_repl`）**从不调用 `reset_state()`** —— 该函数只在
    `run_agent`（UNATTEND）与 REPL 的 `/clear` 里调。所以交互式会话全程
    `GLOBAL_STATE` 都停在模块加载时的内容。若初始为空字典，任何
    `GLOBAL_STATE["tasks"]` 直接取键都会 `KeyError`（实测：REPL 里 `todo_read`
    报 `KeyError: 'tasks'`；同类隐患还有 `build_context` 的 goal/tasks/done_list、
    接手模式的 takeover_mode）。根治办法是让这些键**恒定存在**，而不是在
    每个读取点补 `.get(..., 默认值)`。
    """
    return {
        "goal": "",
        "tasks": [],
        "done_list": [],
        "current_focus": "",
        "planning_done": False,
        "plan_mode": False,
        "worktree": None,
        # GLM 顶层 Planner 的执行契约（None = 本地自规划模式）
        "plan": None,
        # 本轮 agent 自己写入/生成的文件（相对路径）。用于最终校验只跑「自己的测试」。
        "generated_files": [],
        # 接手模式：受保护的既有测试夹具集合 / 是否接手模式
        "protected_tests": set(),
        "takeover_mode": False,
        # 终态校验结果缓存
        "final_validation": None,
    }


GLOBAL_STATE: Dict[str, Any] = _default_state()
work_memory: List[Dict[str, Any]] = []


def reset_state() -> None:
    # 注意：必须在【原地】修改 GLOBAL_STATE（clear + update），绝不能重新赋值。
    # supervisor / llm_glm 等模块以 `from .state import GLOBAL_STATE` 按引用持有同一个
    # 字典对象；若此处 `GLOBAL_STATE = {...}` 重新赋值，那些模块会持有旧字典，
    # 导致 Planner 写入的计划落在旧对象、而 get_current_task() 读取新对象 —— 表现就是
    # 迭代第 1 轮立刻「所有任务已完成」。原地修改可保证所有引用一致。
    global GLOBAL_STATE, work_memory
    GLOBAL_STATE.clear()
    GLOBAL_STATE.update(_default_state())
    del work_memory[:]


def get_current_task() -> Optional[Dict[str, Any]]:
    # 容错：dispatch 可能在 reset_state() 之前被调用（如离线冒烟/单测），
    # 此时 GLOBAL_STATE 尚未含 "tasks"。返回 None 而非抛 KeyError，
    # 以免任一 after 钩子崩溃整个分发器（钩子应为可拼装、永不阻断）。
    for t in GLOBAL_STATE.get("tasks", []):
        if t.get("status") == "pending":
            return t
    return None


def mark_current_task_done() -> Optional[Dict[str, Any]]:
    for t in GLOBAL_STATE.get("tasks", []):
        if t.get("status") == "pending":
            t["status"] = "done"
            GLOBAL_STATE["done_list"].append(t["desc"])
            return t
    return None




# ======================================================================
# 「停滞/循环」防护：签名 / 指纹 / 提示
# ======================================================================
_HEAVY_DIRS = ("node_modules", ".git", "__pycache__", ".venv",
               "litert-lm-cache", ".mypy_cache", "agent_sandbox")


def _exclude_heavy(files) -> List[str]:
    """过滤掉位于重型/依赖目录中的文件。"""
    out = []
    for f in files:
        rel = f.replace("\\", "/")
        if set(rel.split("/")) & set(_HEAVY_DIRS):
            continue
        out.append(f)
    return out


def _workspace_signature() -> frozenset:
    """工作区状态签名：所有文件(相对路径:md5) + 已完成任务集合。用于进展判定。"""
    sig = set()
    try:
        for p in WORKSPACE.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(WORKSPACE)
            if set(rel.parts) & set(_HEAVY_DIRS):
                continue
            try:
                h = hashlib.md5(p.read_bytes()).hexdigest()[:12]
            except Exception:
                h = "?"
            sig.add(f"{rel}:{h}")
    except Exception:
        pass
    sig.add("DONE:" + ",".join(sorted(GLOBAL_STATE.get("done_list", []))))
    return frozenset(sig)


def _safe_rel(path: str) -> str:
    """把任意路径规整为落在 WORKSPACE 内的相对路径，防止模型写出沙箱。

    根因修复（对应 e2e 双层路径 bug）：supervisor 把 {WORKSPACE} 字面（如
    agent_sandbox/<run_id>/fib，可能为绝对路径）注入系统提示，弱模型会把该完整
    路径当相对基回写（如 agent_sandbox/<run_id>/fib/src/fib.py）。旧实现只剥首层
    字面 "agent_sandbox/"，再用 WORKSPACE 重拼 → 出现
    agent_sandbox/<run_id>/fib/<run_id>/fib/... 双层路径。

    修复：优先剥 WORKSPACE 自身前缀（绝对形态，或相对仓库根形态 ws_rel），
    任意形态都能规整回 WORKSPACE/src/fib.py；仅当完全不匹配时才回退旧的字面兜底。
    """
    s = str(path).replace("\\", "/")
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)
    ws = str(WORKSPACE).replace("\\", "/").rstrip("/")
    # WORKSPACE 相对仓库根（或自身为相对路径）的形态：模型按 prompt 回写该完整前缀时匹配。
    if Path(WORKSPACE).is_absolute():
        try:
            ws_rel = str(Path(WORKSPACE).relative_to(REPO_ROOT)).replace("\\", "/").rstrip("/")
        except Exception:
            ws_rel = ""
    else:
        ws_rel = ws
    # 反复剥 WORKSPACE 前缀（含模型已写出双层的形态：agent_sandbox/<run_id>/fib/<run_id>/fib/...），
    # 直到不再有前缀。单次剥离不足以消掉已经多层嵌套的回写。
    while True:
        if ws and (s == ws or s.startswith(ws + "/")):
            s = "" if s == ws else s[len(ws) + 1:]
        elif ws_rel and (s == ws_rel or s.startswith(ws_rel + "/")):
            s = "" if s == ws_rel else s[len(ws_rel) + 1:]
        else:
            break
    # 完全不匹配 WORKSPACE 时的旧兜底：仅剥首层字面 "agent_sandbox/"，防历史用法退化。
    i = s.find("agent_sandbox/")
    if i != -1:
        s = s[i + len("agent_sandbox/"):]
    s = s.lstrip("/")
    parts = [p for p in s.split("/") if p not in ("", ".", "..")]
    return "/".join(parts)


def _norm_cmd(cmd: str) -> str:
    """归一化 shell 命令：去掉前导 `cd <dir> &&` / `cd <dir>;` 与多余空白，用于「重复命令」判定。"""
    c = (cmd or "").strip()
    c = re.sub(r'^cd\s+["\']?[^"\';\s]+["\']?\s*(?:&&|;)\s*', '', c)
    c = re.sub(r'\s+', ' ', c).strip()
    return c


def _content_hash(s: str) -> str:
    """对动作内容取短哈希，用于区分「同一文件的不同写入/编辑」。"""
    return hashlib.md5((s or "").encode("utf-8", "ignore")).hexdigest()[:10]


def _turn_fingerprint(actions: List[Dict[str, Any]]) -> str:
    """把一轮动作归一化为可比较的指纹，用于「重复动作」判定。

    关键：write_file / edit_file 必须把内容写进指纹，否则「同一文件追加不同代码」
    会被误判为「完全相同的重复动作」，触发虚假的循环防护与强制重规划。
    """
    if len(actions) == 1:
        a = actions[0]
        act = a.get("action")
        if act == "shell":
            return "shell:" + _norm_cmd(a.get("cmd", ""))
        if act == "write_file":
            return f"write:{_safe_rel(a.get('path', ''))}:{_content_hash(a.get('content', '') or '')}"
        if act == "edit_file":
            blob = (a.get("old_string", "") or "") + "\u0001" + (a.get("new_string", "") or "")
            return "edit:" + _safe_rel(a.get("path", "")) + ":" + _content_hash(blob)
        if act == "read_file":
            return f"{act}:{a.get('path', '')}:offset:{a.get('offset', '')}:limit:{a.get('limit', '')}"
        if act in ("grep", "glob"):
            # grep/glob 的区分度主要在 pattern，path 只是限定范围，两者都要进指纹。
            # 陷阱：不能写 a.get('path', a.get('pattern','')) —— dict.get 的默认值只在
            # 键【不存在】时才生效，而 path 几乎总存在（常被模型填成 "."），
            # 于是 pattern 永远取不到 → 「同目录搜不同关键词」被判成同一动作 → 虚假 stuck。
            return f"{act}:{a.get('pattern', '')}:{a.get('path', '')}"
        return act
    # 多动作轮次（parallel tool calls）：逐项算单动作指纹再拼接。
    # 不能只用动作名拼接——「read a.py + grep x」与「read a.py + grep y」动作名序列相同，
    # 会被误判成重复动作、触发虚假 stuck，白砍掉模型的一轮多步计划。
    return "multi:" + "|".join(_turn_fingerprint([a]) for a in actions)


def _loop_nudge(kind: str, val: int, fp: str) -> str:
    if kind == "repeat":
        return ("⚠️ 重复动作告警：你已连续 "
                f"{val} 次执行完全相同的动作（指纹 {fp!r}）却没有产生任何新进展。"
                "请【立即停止重复】，不要反复下发同一条命令。工作区是空目录就直接用 write_file 创建文件、"
                "用 edit_file 修改、用 shell 跑对应技术栈的测试验证（Python 用 pytest、Go 用 `go test ./...`）；你现在必须【动手写代码】，而不是继续只做同一个动作。"
                "继续重复将强制终止运行。")
    return ("⚠️ 停滞告警：已连续 "
            f"{val} 轮没有任何进展（没有新增/修改文件、任务也未推进）。"
            "请停止当前做法：**读取报错的测试/文件，修正语法或逻辑错误后重新运行验证"
            "（Python 用 `python -m pytest`，Go 用 `go test ./...`），不要重新规划任务清单**。"
            "任务清单已固定，直接按当前项动手改代码，不要重写任务列表。")


def _loop_replan(kind: str, val: int, fp: str) -> str:
    return ("⚠️ 已多次停滞/重复（"
            f"{kind}={val}，最近动作 {fp!r}），强制要求你【先用 plan 动作重新规划任务与实现方式】，"
            "再继续推进。不要重复刚才的做法，也不要继续在同一个文件上小幅修改却无实质进展。")


def _action_fp(a: Dict[str, Any]) -> str:
    """动作指纹：区分「同一动作被重复下发」与「不同的有产出动作」。"""
    act = a.get("action", "?")
    target = a.get("path", a.get("pattern", a.get("command", "")))
    extra = ""
    if act in ("write_file", "edit_file"):
        extra = (a.get("content") or a.get("new_string") or "")[:60]
    return f"{act}|{target}|{extra}"


# ======================================================================
# 遥测 / 跨 run 记忆 / 统计
# ======================================================================
RUN_TELEMETRY: List[str] = []


def _reset_run_telemetry() -> None:
    RUN_TELEMETRY.clear()


def _telemetry(tag: str) -> None:
    if tag not in RUN_TELEMETRY:
        RUN_TELEMETRY.append(tag)


# AGENT_MEMORY_PATH / AGENT_MEMORY_MAX 现已统一定义在 config.py（单一事实来源），此处直接复用。
from .config import AGENT_MEMORY_MAX  # noqa: F401  （保持模块级名字向后兼容）
from swe_agent.log import logger


def load_agent_memory() -> List[Dict[str, Any]]:
    try:
        if AGENT_MEMORY_PATH.exists():
            data = __import__("json").loads(AGENT_MEMORY_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def append_agent_memory(entry: Dict[str, Any]) -> None:
    try:
        mem = load_agent_memory()
        mem.append(entry)
        # 去重：与上一条 lesson 完全相同则跳过，避免重复堆叠。
        if len(mem) >= 2 and mem[-1].get("lesson") == mem[-2].get("lesson"):
            mem.pop()
        if len(mem) > AGENT_MEMORY_MAX:
            mem = mem[-AGENT_MEMORY_MAX:]
        AGENT_MEMORY_PATH.write_text(
            __import__("json").dumps(mem, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        logger.info('%s', f'[memory] 写入失败：{e}')


def build_memory_note(max_entries: int = 8) -> str:
    """把最近的经验压缩成一段系统提示（空记忆返回 ''）。"""
    mem = load_agent_memory()
    if not mem:
        return ""
    recent = mem[-max_entries:]
    GENERIC_FAIL = "任务未在预算内通过校验，建议复盘测试接口与实现是否对齐。"
    lines = ["# 🧠 历史经验（跨任务记忆，仅供参考，勿盲从）"]
    for e in recent:
        # 只注入 lesson 正文；ts/verdict 不渲染进 system prompt（避免变成日志式噪声）。
        # 过滤无洞察力的通用失败结论。
        lesson = (e.get("lesson") or "").strip()
        if not lesson or lesson == GENERIC_FAIL:
            continue
        lines.append(f"- {lesson}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _derive_lesson(goal: str, passed: bool, early_stop: bool) -> str:
    """根据本次 run 的遥测信号，生成一句可复用的经验。"""
    tags = RUN_TELEMETRY
    if early_stop:
        return "运行被停滞/循环防护提前终止，模型未能在预算内收敛（可能卡在读/空输出循环）。"
    if "doom_guard" in tags:
        return "测试以相同根因反复失败，触发必然失败防护——需换根本方案而非小修小补。"
    if "read_loop_sanitized" in tags and not passed:
        return "模型反复只读不写、被读循环净化器压制，未能产出有效编辑/修复；应尽早动手 edit。"
    if "xml_toolcall_parsed" in tags:
        return "模型以 <tool_call> XML 信封输出动作，已由兜底解析器兼容执行（无需改提示词）。"
    if passed:
        return "任务通过技术栈校验（对应测试框架，如 pytest / go test 等），交付物可用。"
    return "任务未在预算内通过校验，建议复盘测试接口与实现是否对齐."


def _cap_history(text: str, limit: int = 1500) -> str:
    """把存入对话历史的模型输出截断，防止一次性写出超大文件时上下文被灌爆。"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[历史已截断，原文 {len(text)} 字符，避免上下文膨胀；如需完整内容请用 read_file]"


def stats_summary() -> str:
    total = STATS["prompt_tokens"] + STATS["completion_tokens"]
    out = (f"调用 {STATS['calls']} 次（重试 {STATS['retries']}），"
           f"tokens {STATS['prompt_tokens']}+{STATS['completion_tokens']}={total}"
           f"（其中思考 {STATS.get('reasoning_tokens', 0)}），"
           f"动作 {sum(STATS['actions'].values())} 个："
           + ", ".join(f"{k}x{v}" for k, v in sorted(STATS['actions'].items())))
    if STATS.get("glm_calls"):
        out += (f"；GLM 顶层 {STATS['glm_calls']} 次 / "
                f"{STATS['glm_tokens']} tokens")
    return out
