#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/tools.py —— 自描述工具原语（从原 demo.py 抽取，保持逻辑一致）

所有路径均相对于工作区 WORKSPACE（agent_sandbox）。工具通过 @tool 装饰器注册进
ToolRegistry，模型输出 {"action": <name>, ...} 由 Supervisor 调用 ToolRegistry.dispatch。
"""

import re
import sys
import json
import time
import shutil
import shlex
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests  # web_fetch / web_search（best-effort，依赖见 demo.py）

from .config import (
    WORKSPACE, YOLO_MODE, VENV_BIN,
    _TEST_FILE_RE, UNATTENDED_MODE,
)
from .state import GLOBAL_STATE, _safe_rel
from .registry import tool, ToolRegistry, ToolDef
from swe_agent.log import logger

# 搜索/列目录时跳过的重型目录（与 demo.py 一致；config 未导出，这里本地保留）。
_HEAVY_DIRS = ("node_modules", ".git", "__pycache__", ".venv",
               "litert-lm-cache", ".mypy_cache", "agent_sandbox")


def _sanitize(msg: str) -> str:
    """把用户可见报错里的真实绝对 WORKSPACE 路径替换成相对锚点 `./`。

    与 prompt「所有路径使用相对路径（相对 cwd）」保持一致：Python 的 FileNotFoundError/
    OSError 异常消息会内嵌解析后的绝对路径（如 `'/abs/.../conway/src/life.py'`），若直接
    回灌工具结果，弱模型会照抄成绝对路径、触发 cwd thrash。

    早期实现把 WORKSPACE 前缀替换成占位符 `<WORKSPACE>`——但弱模型把 `<WORKSPACE>` 当成
    真实目录记号，反而重建出 `cd /workspace/...`（见 2026-09-03 analyzer/tester 实测：
    报错 `<WORKSPACE>/src/fib.py` 成了模型写 `cd /workspace` 的诱因之一）。故改为相对锚点
    `./`：报错展示为 `./src/fib.py` 这类相对路径，与 prompt 的「用相对路径」引导同构，
    模型无需、也无从拼出绝对 /workspace。
    """
    if not msg:
        return msg
    # 规范化 WORKSPACE：统一成「带尾斜杠」形态（原路径可能带 / 也可能不带）。
    # 这样 `/abs/ws/src` 里的 `/abs/ws/` 整体替换为 `./`，不会残留出 `.//` 双斜杠；
    # 末尾再补一次「无尾斜杠」替换，兼容消息直接引用工作区根本身（无尾斜杠）的写法。
    ws = str(WORKSPACE).rstrip("/") + "/"
    return msg.replace(ws, "./").replace(ws.rstrip("/"), "./")


# ======================================================================
# 命令规整
# ======================================================================
def _strip_redundant_workspace_cd(cmd: str) -> str:
    """exec_shell 已把子进程 cwd 设为 WORKSPACE，若模型又加了 `cd agent_sandbox &&`、
    `cd /workspace/agent_sandbox` 等「试图进入工作区」的前导 cd，会变成「在工作区里再找工作区」
    而失败，导致 pytest 永远跑不起来。

    剥掉任何「试图进入工作区」的前导 cd（判定：解析后==WORKSPACE，或路径末段是 WORKSPACE 叶子，
    或路径末段是历史/规范名「agent_sandbox」——任务 spec 把工作区称作 agent_sandbox，弱模型据此
    常幻觉出 `cd /workspace/agent_sandbox` 这类绝对路径）。不影响其它合法 cd（如 cd src）。

    注意：e2e 下 WORKSPACE 是嵌套隔离目录 `agent_sandbox/<run_id>/<name>`，其末段是任务名
    （如 `fizzbuzz`）而非 `agent_sandbox`；旧实现只比对 `WORKSPACE.name`，漏掉了模型写出的
    `agent_sandbox` 末段，导致 strip 静默失效。故冗余名集合必须同时含二者。"""
    m = re.match(r'^cd\s+("[^"]*"|\'[^\']*\'|[^\s;|&]+)\s*(?:&&|;)?\s*(.*)$',
                 (cmd or "").strip(), re.DOTALL)
    if not m:
        return cmd
    target = m.group(1).strip('"\'')
    ws_basename = Path(WORKSPACE).name  # e2e 下是任务名如 "fizzbuzz"，非 "agent_sandbox"
    redundant_names = {ws_basename, "agent_sandbox"}
    try:
        # 精确匹配：解析后就是工作区本身
        if Path(target).resolve() == Path(WORKSPACE).resolve():
            return m.group(2)
    except Exception:
        pass
    # 宽松匹配：路径末段是工作区叶子或规范名 agent_sandbox（覆盖幻觉绝对路径写法）
    if Path(target).name in redundant_names:
        return m.group(2)
    return cmd


def _fix_cmd(cmd: str) -> str:
    """把裸 python/python3/pytest 指向当前 venv，确保环境一致。

    只替换作为「命令词」（行首或 && ; || | 之后）出现的解释器/pytest，
    避免破坏 `python -m pytest ...` 这类模块调用（旧实现用 str.replace 会把
    `-m pytest` 误改成 `-m .venv/bin/pytest`，导致永远 ModuleNotFoundError）。
    """
    cmd = _strip_redundant_workspace_cd(cmd)
    vp = str(VENV_BIN / "python")
    vpt = str(VENV_BIN / "pytest") + " --timeout=30"
    cmd = re.sub(r"(^|[&|;]\s*)python3?(?=\s|$)",
                 lambda m: m.group(1) + vp, cmd)
    cmd = re.sub(r"(^|[&|;]\s*)pytest(?=\s|$)",
                 lambda m: m.group(1) + vpt, cmd)
    return cmd


# ======================================================================
# 危险命令闸门
# ======================================================================
# (正则, 说明)。命中则交互确认；批处理/无法交互时直接拒绝，除非 --yolo。
DANGEROUS_CMD_PATTERNS = [
    (r"\brm\b[^|;&-]*\-[a-zA-Z]*[rf][a-zA-Z]*\b", "rm 递归/强制删除"),
    (r"(^|[;&|]\s*)sudo\b", "sudo 提权执行"),
    (r"git\s+push\s+[^;|&]*(--force\b|(^|\s)-f\b)", "git push 强制推送"),
    (r"git\s+reset\s+--hard\b", "git reset --hard 丢弃改动"),
    (r"\bmkfs(\.\w+)?\b|\bdd\s+[^;|&]*\bof=", "磁盘级写入"),
    (r"chmod\s+(-R\s+)?777\b", "chmod 777 全开放权限"),
    (r">\s*/(etc|usr|bin|sbin|System|Library)\b", "重定向写系统目录"),
    (r"(curl|wget)[^|;&]*\|\s*(ba)?sh\b", "远程脚本直接执行"),
    (r"\b(shutdown|reboot|halt)\b|\bkillall\b|\bpkill\b", "系统级进程/电源操作"),
    (r"\bcrontab\b", "修改定时任务"),
    (r"(~|/)\.ssh\b", "访问 SSH 密钥"),
]

# write_file/edit_file 不允许覆盖的核心文件（防模型自毁运行环境）
CORE_PROTECTED_FILES = {"demo.py", "mcp.json", "mcp_demo_server.py",
                         "pyproject.toml", "uv.lock", "requirements.txt"}


def check_dangerous(cmd: str) -> Optional[str]:
    for pat, desc in DANGEROUS_CMD_PATTERNS:
        if re.search(pat, cmd):
            return desc
    return None


def _gate_confirm(desc: str, detail: str) -> Optional[str]:
    """返回 None 表示放行；返回字符串为拒绝原因。"""
    if YOLO_MODE:
        logger.info('%s', f'[gate] (--yolo) 放行危险操作：{desc}')
        return None
    # 无人值守模式（UNATTENDED_MODE=1）：BUILD 层不向用户提问，危险操作一律拒绝（不调 input()），
    # 避免后台/batch 下 isatty() 误判导致的 stdin 卡死。
    if UNATTENDED_MODE:
        return (f"denied: 无人值守模式已拦截危险操作[{desc}]。如确需执行请加 --yolo 启动，"
                f"或改用更安全的方式（例如只删除沙箱内相对路径文件）。")
    # 交互模式（TTY）下询问；批处理/管道下直接拒绝
    try:
        if sys.stdin.isatty():
            logger.info('%s', f'\n⚠️  危险操作待确认 [{desc}]：{detail[:200]}')
            ans = input("允许执行吗？(y/N) ").strip().lower()
            if ans in ("y", "yes"):
                return None
    except Exception:
        pass
    return (f"denied: 已拦截危险操作[{desc}]。如确需执行请加 --yolo 启动，"
            f"或改用更安全的方式（例如只删除沙箱内相对路径文件）。")


@tool(
    name="shell",
    description="在工作区内执行一条 shell 命令并返回 stdout/stderr/exit code（输出完整返回，不做长度截断）。",
    category="shell",
    schema={"type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"]},
    dangerous=True,
    examples=[
        '{"action":"shell","cmd":"pytest tests/test_add.py"}',
        '{"action":"shell","cmd":"python -c \\"import math; print(math.factorial(5))\\""}',
    ],
    when_to_use="需要运行测试、编译、安装依赖或任何命令行操作时使用。",
)
def exec_shell(cmd: str, timeout: int = 120) -> str:
    cmd = _fix_cmd(cmd)
    danger = check_dangerous(cmd)
    if danger:
        verdict = _gate_confirm(danger, cmd)
        if verdict:
            return verdict
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(WORKSPACE),
        )
        out = f"=== stdout ===\n{r.stdout}\n=== stderr ===\n{r.stderr}\n=== exit code: {r.returncode} ==="
        # 包结构自愈（TDD 安全网）：模型自己 `shell pytest` 失败时，若因本地模块未打包
        # 触发 ModuleNotFoundError（如 `from src.game import` 但 src 不是包），自动补
        # __init__.py / conftest.py，使下一轮 pytest 直接可达。
        # 之前只在旧式 _run_final_validation 里触发，导致 mid-run 反复跑测试卡在导入错误死循环；
        # 现为单杠校验（_run_test_bar）的 TDD 安全网，mid-run 失败即自愈。
        if r.returncode != 0 and "ModuleNotFoundError: No module named" in out:
            try:
                from .harness import _try_package_selfheal
                healed, heal_msg = _try_package_selfheal(out, {"lang": "python"})
                if healed and heal_msg:
                    out = out + f"\n[harness 自愈] {heal_msg}"
            except Exception:
                pass
        return out
    except Exception as e:
        return f"shell_error: {_sanitize(str(e))}"

# 隐藏/权威验收测试文件名（test_acceptance*.py）——由 harness 在最终校验时临时落盘到工作区运行。
# Executor 既不应读取也不应写入/修改它，否则可照抄断言或对拍篡改验收标准来作弊。
_HIDDEN_TEST_RE = re.compile(r"test_acceptance.*\.py$", re.IGNORECASE)


def _is_hidden_test_path(path: str) -> bool:
    return bool(_HIDDEN_TEST_RE.search(path or ""))


def _hidden_test_guard(path: str) -> Optional[str]:
    """若 path 命中隐藏验收测试文件名，返回拒绝说明；否则返回 None（放行）。"""
    if _is_hidden_test_path(path):
        return (f"denied: {path} 是隐藏/权威验收测试文件，由 harness 在最终校验时管理，"
                f"Executor 不可读取、写入或修改（防止照抄断言或篡改验收标准绕过闸门）。"
                f"请只修改你自己的实现代码。")
    return None



@tool(
    name="read_file",
    description="读取工作区内文件内容，支持按行区间读取并显示行号；完整返回不做截断（大文件建议用 offset/limit 按需读取）。",
    category="fs",
    schema={"type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"]},
    dangerous=False,
    examples=[
        '{"action":"read_file","path":"src/game.py"}',
        '{"action":"read_file","path":"main.py","offset":1,"limit":50}',
    ],
    when_to_use="需要查看文件内容、定位某一行或只读取文件尾部时使用。",
)
def read_file(path: str, offset: int = None, limit: int = None) -> str:
    """读取文件内容，优先按需「按行区间」读取，可以搭配LSP先确定要读取的行号、行数，避免一次把大文件整篇塞进上下文。

    - offset / limit 均为「1 基」行号：offset 为起始行（含），limit 为读取行数。
    - 都不给 -> 读取完整文件（向后兼容旧调用 / 小文件）。
    - 仅给 limit -> 读取末尾 limit 行（便于看文件尾部 / 最新追加内容）。
    - 每行带「行号\\t内容」前缀，方便后续用 edit_file 精准定位，也顺带劝阻「整文件重抄」。
    - 输出首行标注「[read_file <path> 显示第 a-b 行 / 共 N 行]」，让模型知道上下文与总量。
    """
    g = _hidden_test_guard(path)
    if g:
        return g
    p = WORKSPACE / _safe_rel(path)
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"read_error: {_sanitize(str(e))}"
    lines = text.split("\n")
    total = len(lines)
    if offset is None and limit is None:
        shown = lines
        start = 1
    elif limit is None:
        start = max(1, offset if offset and offset >= 1 else 1)
        shown = lines[start - 1:]
    elif offset is None:
        start = max(1, total - limit + 1)
        shown = lines[start - 1:]
    else:
        start = max(1, offset if offset >= 1 else 1)
        shown = lines[start - 1:start - 1 + limit]
    if not shown:
        return f"[read_file {path} 第 {start} 行超出文件范围（共 {total} 行）]"
    end = start + len(shown) - 1
    numbered = [f"{start + i:6d}\t{ln}" for i, ln in enumerate(shown)]
    header = f"[read_file {path} 显示第 {start}-{end} 行 / 共 {total} 行]"
    return header + "\n" + "\n".join(numbered)


def _protect_core_file(path: str) -> Optional[str]:
    rel = _safe_rel(path)
    if rel in CORE_PROTECTED_FILES:
        if YOLO_MODE:
            logger.info('%s', f'[gate] (--yolo) 放行覆盖核心文件：{rel}')
            return None
        return (f"denied: 禁止覆盖核心文件 {rel}（会破坏 Agent 运行环境）。"
                f"如确需修改请人工编辑，或加 --yolo 启动。")
    return None


def _fuzzy_line_replace(text: str, old_string: str, new_string: str):
    """old_string 精确匹配失败时，按「逐行空白归一化」做模糊定位：容忍缩进/空格差异，
    把原文中对应的连续行区间替换为 new_string（保留 new_string 自身空白）。
    弱模型重抄代码极易引入空白差异导致 old_string 对不上、edit 全失败而陷入空转——
    这里命中即替换，让 edit 真正落地（被视为「有进展」，避免被停滞保护误杀）。
    返回替换后的文本；无法唯一定位则返回 None。"""
    def _wsp(line):
        return " ".join(line.split())
    tl = text.split("\n")
    ol = old_string.split("\n")
    nl = new_string.split("\n")
    if not ol:
        return None
    pat = [_wsp(x) for x in ol]
    tnorm = [_wsp(x) for x in tl]
    L = len(pat)
    hits = [i for i in range(len(tnorm) - L + 1) if tnorm[i:i + L] == pat]
    # 单行 old_string 若多处命中则放弃（避免误替换）；多行 chunk 取首个匹配（通常已足够独特）
    if (L == 1 and len(hits) != 1) or (L > 1 and not hits):
        return None
    i = hits[0]
    return "\n".join(tl[:i] + nl + tl[i + L:])



def _write_verify(p, text):
    """写盘后真实回读校验内容一致性：写盘再读回，与预期字符串逐字比对。
    返回 None 表示通过；否则返回错误说明字符串（落盘被截断 / 编码异常 / IO 失败）。"""
    try:
        p.write_text(text, encoding="utf-8")
    except Exception as e:
        return f"写入失败：{_sanitize(str(e))}"
    try:
        got = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"回读失败：{_sanitize(str(e))}"
    if got != text:
        return (f"回读校验失败：落盘内容与预期不一致"
                f"（期望 {len(text)} 字节，实际 {len(got)} 字节，"
                f"可能被截断或编码异常）。请重试 edit/write。")
    return None


@tool(
    name="write_file",
    description="以整文件覆盖方式在工作区写入/创建文件，超长文件会被拒绝以强制拆分。写后整篇回读确保写入成功。",
    category="fs",
    schema={"type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"]},
    dangerous=False,
    examples=[
        '{"action":"write_file","path":"src/add.py","content":"def add(a,b):\\n    return a+b\\n"}',
        '{"action":"write_file","path":"test_add.py","content":"from src.add import add\\ndef test_add():\\n    assert add(1,2)==3\\n"}',
    ],
    when_to_use="需要新建或整体覆盖一个文件时使用（建议内容精简，≤80 行）。",
)
def write_file(path: str, content: str) -> str:
    if not path:
        return "write_error: 路径为空"
    g = _hidden_test_guard(path)
    if g:
        return g
    # 接手模式保护：禁止覆盖预置测试夹具（oracle）
    if (GLOBAL_STATE.get("takeover_mode")
            and _TEST_FILE_RE.search(path)
            and path in GLOBAL_STATE.get("protected_tests", set())):
        return (f"denied: 接手模式下禁止覆盖受保护测试文件 {path}（这是预置校验夹具，"
                f"请改用 edit_file 只改被测源码使其由红变绿）。")
    verdict = _protect_core_file(path)
    if verdict:
        return verdict
    p = WORKSPACE / _safe_rel(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # 真实回读校验：写盘后读回比对，确保落盘内容与写入完全一致（非仅语法校验）
        _wv = _write_verify(p, content)
        if _wv:
            return f"write_error: {_sanitize(_wv)}"
        # 记录为本轮 agent 生成的文件（用于最终校验仅跑自己的测试）
        gf = GLOBAL_STATE.setdefault("generated_files", [])
        if path not in gf:
            gf.append(path)
        # 注意：语法/lint 校验不在此内联返回——按架构 lint 只在 loop_2 收尾闸门
        # （_l2_gate，finish_reason 之后）跑，避免对每一次 write/edit（含增量草稿）做
        # 完成度审判、制造 premature+punitive 的反馈节奏。写工具返回值只描述写入本身。
        return f"write_success: {path} ({len(content)} 字节)"
    except Exception as e:
        return f"write_error: {_sanitize(str(e))}"


# 单次写入行数护栏 / 单次读取行数护栏 已迁为自描述 guard 类（swe_agent/guards.py）：
#   WriteSizeGuard（BEFORE_TOOL_CALL，仅管 write_file，超阈值 REJECT 回灌+继续）
#   ReadSizeGuard（BEFORE_TOOL_CALL，仅管 read_file，超阈值 REJECT 回灌+继续）
# 二者在 guards.py 模块 import 时经 HOOK_HUB.register 自注册（REPL / selfcheck / dispatch 直调全覆盖），
# 工具内不再有任何行数硬拒绝。新增类似护栏 = 写一个 Guard 子类 + 注册，本文件零改动。
from .guards import register_all_builtin  # 触发内置 guard 注册（guards 模块 import 即注册；显式再确保，幂等）
register_all_builtin()


@tool(
    name="edit_file",
    description="在文件内精确替换 old_string 为 new_string；old_string 为空时在文件末尾追加内容。写入前read_file读取要更改的行区间，写入后需要回读写入区间检查确保写入成功。",
    category="fs",
    schema={"type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"]},
    dangerous=False,
    examples=[
        '{"action":"edit_file","path":"src/add.py","old_string":"return a+b","new_string":"return a+b+0"}',
        '{"action":"edit_file","path":"src/add.py","old_string":"","new_string":"\\n# 追加的方法\\n"}',
    ],
    when_to_use="需要修改既有文件（最小改动）或往文件末尾追加代码时使用。",
)
def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    if not path:
        return "edit_error: 路径为空"
    g = _hidden_test_guard(path)
    if g:
        return g
    # 接手模式保护：禁止覆盖预置测试夹具（oracle）
    if (GLOBAL_STATE.get("takeover_mode")
            and _TEST_FILE_RE.search(path)
            and path in GLOBAL_STATE.get("protected_tests", set())):
        return (f"denied: 接手模式下禁止覆盖受保护测试文件 {path}（这是预置校验夹具，"
                f"请改用 edit_file 只改被测源码使其由红变绿）。")
    if old_string == "":
        # 模型常见意图：在文件末尾【追加】内容（如补一个方法 / 一段代码），而非「唯一定位替换」。
        # 直接把 new_string 接到文件末尾，避免反复下发空 old_string 触发 edit_error 后陷入
        # 「下发→报错→再下发」的空转死循环（qwen-7b 常以此方式尝试追加方法）。
        p = WORKSPACE / _safe_rel(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            cur = p.read_text(encoding="utf-8") if p.exists() else ""
            sep = "" if (not cur or cur.endswith("\n")) else "\n"
            new_text = cur + sep + new_string
            # 真实回读校验（模型最常走 edit_file old_string="" 追加路径，必须确认落盘）
            _wv = _write_verify(p, new_text)
            if _wv:
                return f"edit_error: 追加写入校验失败：{_sanitize(_wv)}"
            gf = GLOBAL_STATE.setdefault("generated_files", [])
            if path not in gf:
                gf.append(path)
            return f"edit_success: {path}（追加 {len(new_string)} 字节）"
        except Exception as e:
            return f"edit_error: 追加失败：{_sanitize(str(e))}"
    verdict = _protect_core_file(path)
    if verdict:
        return verdict
    p = WORKSPACE / _safe_rel(path)
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"edit_error: 无法读取 {path}：{_sanitize(str(e))}"
    # 空转 edit 防护：old_string 与 new_string 完全相同时，替换必然是 no-op。
    # 必须拒绝并回灌错误——否则会报 edit_success、白勾任务（GLM 实测以此刷假进度 4 轮）。
    if old_string == new_string:
        return (f"edit_error: old_string 与 new_string 完全相同，这是无效的空转编辑。"
                f"请让 new_string 与 old_string 有实质差异；若任务是补实现，"
                f"请把待替换的旧代码（如 pass / 占位符）放进 old_string，完整新实现放进 new_string。")
    count = text.count(old_string)
    if count == 0:
        # 容错：弱模型重抄代码常因缩进/空格差异导致 old_string 对不上。按行空白归一化再定位一次，
        # 命中即用 new_string（保留其自身空白）替换对应行区间——让 edit 真正落地，避免空转死循环。
        fuzzy = _fuzzy_line_replace(text, old_string, new_string)
        if fuzzy is not None:
            _wv = _write_verify(p, fuzzy)
            if _wv:
                return f"edit_error: 模糊替换校验失败：{_sanitize(_wv)}"
            gf = GLOBAL_STATE.setdefault("generated_files", [])
            if path not in gf:
                gf.append(path)
            return f"edit_success: {path}（模糊匹配替换，+{len(new_string) - len(old_string)} 字节）"
        return f"edit_error: 在 {path} 中未找到 old_string（请确认内容/缩进完全一致）"
    if not replace_all and count > 1:
        return (f"edit_error: old_string 在 {path} 中出现 {count} 次，未指定 replace_all。"
                f"请提供更长的上下文唯一定位，或设置 replace_all=true。")
    if replace_all:
        new_text = text.replace(old_string, new_string)
    else:
        new_text = text.replace(old_string, new_string, 1)
    _wv = _write_verify(p, new_text)
    if _wv:
        return f"edit_error: 替换写入校验失败：{_sanitize(_wv)}"
    return f"edit_success: {path}（替换 {count} 处，+{len(new_string)-len(old_string)} 字节）"


# 搜索时跳过的二进制/大体积后缀，避免把模型上下文灌爆
_SKIP_SUFFIX = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".bin", ".safetensors", ".gguf",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".zip",
    ".gz", ".tar", ".pdf", ".db", ".sqlite", ".wav", ".mp3", ".mp4",
}


def _resolve_path(path: str) -> Path:
    return WORKSPACE / _safe_rel(path)


@tool(
    name="grep",
    description="在工作区内按正则递归搜索文件内容，返回匹配行（含相对路径与行号），自动跳过二进制文件。",
    category="fs",
    schema={"type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"]},
    dangerous=False,
    examples=[
        '{"action":"grep","pattern":"def add","path":"src"}',
        '{"action":"grep","pattern":"import os"}',
    ],
    when_to_use="需要在代码库中查找某段文本、符号或模式时使用。",
)
def grep_files(pattern: str, path: str = ".") -> str:
    """在工作区内按正则递归搜索文件内容（移植自 GrepTool）。"""
    base = _resolve_path(path)
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"grep_error: 非法正则表达式 {e}"
    matches: List[str] = []
    try:
        it = base.rglob("*") if base.exists() else []
        for f in it:
            if not f.is_file() or f.suffix.lower() in _SKIP_SUFFIX:
                continue
            try:
                lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    rel = f.relative_to(WORKSPACE)
                    matches.append(f"{rel}:{i}: {line}")
                    if len(matches) >= 200:
                        break
            if len(matches) >= 200:
                break
    except Exception as e:
        return f"grep_error: {e}"
    if not matches:
        return "grep: 无匹配"
    return "\n".join(matches[:200])


@tool(
    name="glob",
    description="在工作区内按 glob 模式列出文件（如 **/*.py），自动排除重型依赖目录。",
    category="fs",
    schema={"type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"]},
    dangerous=False,
    examples=[
        '{"action":"glob","pattern":"**/*.py"}',
        '{"action":"glob","pattern":"src/**/*.ts","path":"."}',
    ],
    when_to_use="需要列出符合某文件名模式的文件（摸清工作区结构）时使用。",
)
def glob_files(pattern: str, path: str = ".") -> str:
    """在工作区内按 glob 模式列出文件（移植自 GlobTool）。"""
    base = _resolve_path(path)
    try:
        files = sorted(
            str(p.relative_to(WORKSPACE)) for p in base.glob(pattern)
            if p.is_file() and not (set(p.relative_to(WORKSPACE).parts) & set(_HEAVY_DIRS))
        )
    except Exception as e:
        return f"glob_error: {e}"
    if not files:
        return "glob: 无匹配"
    # 排除 node_modules/.git 等重型目录后若仍很多，截断避免上下文膨胀（曾因 **/* 列出
    # 整个 node_modules 树导致上下文暴涨、第 18 轮卡死）。
    if len(files) > 200:
        return "\n".join(files[:200]) + f"\n...[共 {len(files)} 个匹配，仅显示前 200]"
    return "\n".join(files)


@tool(
    name="ask",
    description="向人类提问并读取回答（非交互模式下返回占位提示）。",
    category="interact",
    schema={"type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question"]},
    dangerous=False,
    examples=[
        '{"action":"ask","question":"需要使用哪个数据库？","options":["sqlite","postgres"]}',
    ],
    when_to_use="遇到需要人类决策/澄清的歧义点时向用户提问。",
)
def ask_user(question: str, options: Optional[List[str]] = None) -> str:
    """向人类提问并读取回答（移植自 AskUserQuestionTool）。

    非交互模式（stdin 不是 TTY，例如后台 e2e）下不调用 input() 阻塞整条链路，
    而是直接返回提示，让上层把这次 ask 当作「无法求助」来处理（转 re-plan / 升级）。
    """
    # 无人值守模式（UNATTENDED_MODE=1）：BUILD 层已摘除 ask 工具，这里再兜底短路，
    # 即便工具被异常暴露也绝不调用 input() 阻塞 stdin。
    from . import config as C
    if C.UNATTENDED_MODE:
        return ("(无人值守模式，不向人类提问；请改用其它动作完成当前任务，"
                "或重新规划（plan）后再试。不要重复 ask。)")
    opt = f" 选项: {' / '.join(options)}" if options else ""
    logger.info('%s', f'\n[agent 提问] {question}{opt}')
    try:
        if not sys.stdin.isatty():
            return ("(非交互模式，无法向人类提问；请改用其它动作完成当前任务，"
                    "或重新规划（plan）后再试。不要重复 ask。)")
        ans = input("你的回答> ").strip()
    except (EOFError, OSError):
        ans = "(非交互模式，无输入)"
    return f"用户回答: {ans}"


@tool(
    name="reload_plugins",
    description="重新扫描插件目录并热加载已配置的插件（技能/命令/子智能体/MCP）。用于「按需加载」——当插件目录新增/修改后，无需重启 harness 即可生效。",
    category="plugin",
    schema={"type": "object",
            "properties": {
                "enable_mcp": {"type": "boolean",
                               "description": "是否同时连接插件声明的 MCP 服务器（默认 false，避免未授权外连）"}
            },
            "required": []},
    dangerous=False,
    examples=[
        '{"action":"reload_plugins"}',
        '{"action":"reload_plugins","enable_mcp":true}',
    ],
    when_to_use="需要刷新或热加载插件时使用（如新增/修改了插件目录里的 skill / agent / mcp 配置）。",
)
def reload_plugins(enable_mcp: bool = False) -> str:
    """按配置的插件根目录重新加载插件（按需热加载，无需重启 harness）。"""
    from . import plugins as _plugins      # 懒导入，避免循环依赖
    from . import config as C
    import os
    root = os.environ.get("SWE_PLUGINS_ROOT") or C.PLUGINS_ROOT
    try:
        state = _plugins.load_plugins(root, enable_mcp=bool(enable_mcp))
    except Exception as e:
        return f"reload_plugins_error: {e}"
    return (f"reload_plugins_ok：已加载 {len(state.get('plugins', []))} 个插件"
            f"（技能 {state.get('skills', 0)} / 命令 {state.get('commands', 0)} / "
            f"工具 {len(state.get('tools', []))} / "
            f"子智能体 {len(state.get('agents', {}))} / MCP {state.get('mcp', [])}）。")


def register_builtin_tools():
    """Import side-effect registration; call once at startup."""
    pass
