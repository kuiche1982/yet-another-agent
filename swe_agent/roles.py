#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/roles.py —— Agent 层（planner / 故障路由）

这一层「绑定自己要做的事情」，但【不绑定任何具体模型】：
- Planner 负责把任务变成结构化执行契约；
- 故障路由负责严重失败时退回 Planner 重规划。

它们需要什么模型，完全由 models.py 的 ROLE_ENV（PLANNER_MODEL/
EXECUTOR_MODEL）在运行时决定。换 GLM → 换本地模型，只是改 env，本文件一行不改。
模型调用统一走 models.chat_text / models.chat_messages，按 provider 路由到对应传输层。
"""

import json
from typing import Dict, Any, List, Optional, Tuple

from . import config as C
from . import models as M
from swe_agent.log import logger, _clip
from .state import (
    GLOBAL_STATE, _telemetry, _HEAVY_DIRS, RUN_TELEMETRY,
)
from .llm_glm import _extract_json_object   # 通用 JSON 提取器（传输无关）
from .registry import ToolRegistry
from . import roles_config as RC   # make_agent / single_loop：analyzer 走统一 Agent loop
# 顶层 import：确保本模块被加载时 tools.py 的 @tool 装饰器已触发注册。
# 否则脱离 supervisor 单独调用 run_analyzer（如 dbg / 单测 / 探针）时，
# read_file/grep/glob 不会被注册，glm_tools 静默丢掉这些工具导致 analyzer 丧失探查能力。
from .tools import read_file, grep_files, glob_files, exec_shell


# ---------------- Planner：任务 → 结构化执行契约 ----------------

_PLANNER_SYSTEM = """你是顶层软件架构 Planner。职责是把开发、测试、debug任务转化为一份【结构化执行契约】，交给下游本地编码 Executor「填空式」实现。Executor 能力有限，契约必须具体、无歧义、可分步执行。

当前目录就是工作目录， 禁止更换目录。 

只输出一个 JSON 对象，禁止任何解释文字、禁止 markdown 代码块。必须包含字段：
- language：目标实现语言（go / python / node）。含 golang/Go→go；含 python→python；含 node/js/ts→node；未指明→python。
- summary：Analyzer的完整需求分析/报告
- modules：文件级白名单契约，Executor 只允许创建/修改这些文件及测试文件，不得新增契约外模块。每项含 path（相对路径：python 用 src/<x>.py、go 用 src/<x>.go、node 用 src/<x>.js|ts）、purpose、public（类/函数签名列表）
- interface：本任务【全部】公开函数最终签名（参数名、类型、返回值），与各 modules.public 完全一致、与任务函数名一一对应；禁止改名或改参数个数。CLI 入口须显式含 def main(...) -> None
- forbidden：禁止事项（如不要引入第三方依赖；不要做 GUI）
- tasks：有序步骤（3~8 步）；每步 = step（描述）+ deliverables（本步落盘的相对路径文件清单，含测试文件；只读调试用 []）
- verify_points：至少 3 条自然语言验收点（覆盖核心功能/主要更改/边界异常/CLI 可运行/关键算法语义），含 id 与 check_hint；供独立只读 tester 逐条验证，比测试更偏用户视角

要求：
- 遵循 TDD，但测试必须覆盖完整验收规格，不能只写第一步测试就变绿：把验证全部功能的测试一次性写进测试文件（首步集中写全，或每步追加，最终测试文件必须覆盖完整规格）。
- 最后一步必须是「运行测试并确认全绿」。完成标准只有一个：工作区测试 count>0 且全部通过。
- 测试【只建一个】文件，放工作区根目录（如 test_life.py），禁止在 tests/ 子目录再建一份或留空文件（pytest 收集会报错）；各步 deliverables 都指向这同一个测试文件。
- modules 数量克制：能单文件解决就不要拆多文件（Executor 是小模型，文件越少越稳）。
- 语言后缀与技术栈一致：go 仅 .go（go test ./...、仅标准库）、python 仅 .py（pytest）、node 仅 .js/.ts（jest/vitest）。
- 任务含「命令行/CLI/可运行/程序」：modules 须规划可执行入口文件（自带 if __name__=='__main__': 调 main()），写一条测试调用 main() 断言不抛异常、退出 0；main() 内部用固定示例演示，不依赖交互式 input()。禁止只交付库不交付命令行入口。
- 接手场景（工作区已有代码/测试）：规划最小修复路径。
- 如果Analyzer认为测试用例正确， forbidden 必须含「无充分理由禁止修改既有测试文件」。

"""




# 语言/技术栈决策已下放给 Planner：见 _PLANNER_SYSTEM 的 language 字段与约束段。
# BUILD 层不再用字符串正则猜测语言——Planner 不确定时由 harness 默认 python。


def _workspace_listing() -> str:
    """给 Planner 的紧凑工作区文件清单（相对路径 + 大小，排除重目录）。"""
    try:
        ws = C.WORKSPACE.resolve()
        lines = []
        for p in sorted(ws.rglob("*")):
            if not p.is_file():
                continue
            if any(part in _HEAVY_DIRS for part in p.parts):
                continue
            try:
                rel = str(p.relative_to(ws))
                size = p.stat().st_size
                lines.append(f"  - {rel} ({size}B)")
            except Exception:
                continue
            if len(lines) >= 120:
                lines.append("  - …（其余略）")
                break
        return "\n".join(lines) if lines else "（空目录，从零开始）"
    except Exception:
        return "（无法读取工作区）"


def _norm_lang(v) -> str:
    """归一化 Planner 给出的 language；不在已知集合内 → python（默认）。"""
    lv = str(v or "python").strip().lower()
    return lv if lv in ("go", "python", "node") else "python"


def _normalize_plan(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """校验/规范化 Planner 产出。结构不合格返回 None。"""
    tasks = plan.get("tasks")
    modules = plan.get("modules")
    if not isinstance(tasks, list) or not tasks:
        return None
    if not isinstance(modules, list):
        modules = []
    norm_mods = []
    for m in modules:
        if isinstance(m, dict) and m.get("path"):
            norm_mods.append({
                "path": str(m["path"]),
                "purpose": str(m.get("purpose", "")),
                "public": [str(x) for x in (m.get("public") or []) if x],
            })
    norm = {
        "summary": str(plan.get("summary", "")),
        "language": _norm_lang(plan.get("language")),
        "modules": norm_mods,
        "interface": [str(x) for x in (plan.get("interface") or []) if x],
        "forbidden": [str(x) for x in (plan.get("forbidden") or []) if x],
        "tasks": [_normalize_task(t) for t in tasks],
        "verify_points": _normalize_verify_points(plan.get("verify_points")),
    }
    return norm


def _normalize_verify_points(vps) -> List[Dict[str, Any]]:
    """把 Planner 的 verify_points 规范为 [{'id','point','check_hint'}]。"""
    if not isinstance(vps, list):
        return []
    out = []
    for i, v in enumerate(vps, 1):
        if not isinstance(v, dict):
            continue
        point = str(v.get("point") or "").strip()
        if not point:
            continue
        out.append({
            "id": v.get("id") or i,
            "point": point,
            "check_hint": str(v.get("check_hint") or "").strip(),
        })
    return out


def _normalize_task(t) -> Dict[str, Any]:
    """把 Planner 的单个任务规范化为 {'desc', 'deliverables', 'status'}。

    兼容两种形态：
      - 对象：{"step": "...", "deliverables": [...]}（新契约）
      - 纯字符串：旧形态或非结构化文本（deliverables 留空）
    """
    if isinstance(t, dict):
        desc = str(t.get("step") or t.get("desc") or "").strip()
        raw_dl = t.get("deliverables") or []
        deliverables = [str(x).strip() for x in (raw_dl if isinstance(raw_dl, (list, tuple)) else []) if x]
    else:
        desc = str(t).strip()
        deliverables = []
    return {"desc": desc, "deliverables": deliverables, "status": "pending"}


# LM Studio（localhost_openai）只接受 json_schema（拒绝 json_object，会 400）；
# Zhipu/GLM（remote_openai）用 json_object。最终按实际调用模型的 provider 再适配一次
# （升级子 agent 可能换 provider：lfm2.5 主模型用 json_schema，升级到 glm-4.7 时转 json_object）。
_PLANNER_JSON_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
        "json_schema": {
            "name": "execution_plan",
            "strict": "true",
        "schema": {
            "type": "object",
            "properties": {
                "language": {"type": "string", "description": "目标实现语言（go / python / node）。依据【开发任务】判断：含 golang/Go→go；含 python→python；含 node/js/ts→node；未指明且无法判断→python。"},
                "summary": {"type": "string", "description": "一句话方案概述"},
                "modules": {
                    "type": "array",
                    "description": "模块划分：文件级白名单契约，Executor 只允许创建或修改这些文件及测试文件，不得新增契约外模块；CLI 任务需规划带 main 入口的可执行文件。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "模块相对路径（如 src/game.py）；后缀与技术栈一致（python 用 .py、go 用 .go、node 用 .js 或 .ts）。"},
                            "purpose": {"type": "string", "description": "该模块的功能描述"},
                            "public": {"type": "array", "items": {"type": "string"}, "description": "该模块实现的类/函数签名列表（与 interface 逐字一致）"},
                        },
                        "required": ["path", "purpose", "public"],
                        "additionalProperties": False,
                    },
                },
                "interface": {"type": "array", "items": {"type": "string"}, "description": "接口定义：本任务全部公开函数最终签名（参数名、类型、返回值），与各 modules.public 完全一致、与任务描述函数名一一对应；禁止自行改名或改参数个数；CLI 入口需显式含 main 函数。"},
                "forbidden": {"type": "array", "items": {"type": "string"}, "description": "注意事项列表：明确禁止事项，如不要引入第三方依赖、不要做 GUI。"},
                "tasks": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 8,
                    "description": "明确可实现的任务列表（有序步骤，共 1 到 8 步）；每步 = step（描述）+ deliverables（本步真正落盘的相对路径文件清单，含测试文件，只读调试用空数组）。遵循 TDD：测试必须覆盖完整验收规格；测试文件至少一个、放在工作区根目录、不建子目录；最后一步必须是运行测试并确认全绿。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string", "description": "具体实现步骤描述"},
                            "deliverables": {"type": "array", "items": {"type": "string"}, "description": "本步会创建或修改的相对路径文件清单，含测试文件；只读调试用空数组"},
                        },
                        "required": ["step", "deliverables"],
                        "additionalProperties": False,
                    },
                },
                "verify_points": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 6,
                    "description": "自然语言描述的检查点列表（最少1点、最多3点）：供独立、只读的 tester 逐条验证需求是否被真正满足，比测试更偏用户视角。每条含 id 与 point（验收标准描述）；check_hint 给 tester 具体验证手段（读哪个文件或跑哪条命令）。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "检查点序号（整数）"},
                            "point": {"type": "string", "description": "一句自然语言描述的验收标准"},
                            "check_hint": {"type": "string", "description": "给 tester 的验证建议：读哪个文件或跑哪条命令"},
                        },
                        "required": ["id", "point"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["language", "summary", "modules", "interface", "forbidden", "tasks", "verify_points"],
            "additionalProperties": False,
        },
    },
}


def _planner_response_format() -> Optional[Dict[str, Any]]:
    """按 Planner 后端选择结构化输出格式（与 GLM 的 json_object 不同，LM Studio 用 json_schema）。

    - LM Studio（localhost_openai）：json_schema（该版本拒绝 json_object，400）；
    - Zhipu/GLM（remote_openai）：json_object；
    - 本地（openai_local）：弱模型不稳定，None 退回纯 prompt 约束。
    升级子 agent（如 glm-4.7）走 models.chat_text 内的 provider 适配，自动转 json_object。
    """
    spec = M.role_spec("planner")
    if not spec:
        return None
    transport = M.PROVIDERS.get(spec["provider"], {}).get("transport")
    if transport == "localhost_openai":
        return _PLANNER_JSON_SCHEMA
    if transport == "remote_openai":
        return {"type": "json_object"}
    return None


def _make_plan(user_task: str, replan_context: str = "",
              research_findings: str = "") -> Optional[Dict[str, Any]]:
    """调用当前角色分配的 Planner 模型生成执行契约。replan_context 非空=故障路由重规划。
    research_findings 非空=Analyzer 只读调研阶段的发现摘要，注入以避免 Planner 重复探查。"""
    user = f"【开发任务】\n{user_task}\n\n【当前工作区文件清单】\n{_workspace_listing()}"
    if research_findings:
        user += (f"\n\n【Analyzer 调研发现（只读探查得到的事实，请据此规划，不要重复探查）】\n"
                 f"{research_findings}")
    if replan_context:
        user += f"\n\n【重规划上下文（上一版契约执行失败，请改方案）】\n{replan_context}"
    # 走统一 Agent（JSON 模式）：prompt/tools/mode 全部来自 roles_config 的 RoleConfig，
    # 避免散落的 _PLANNER_SYSTEM / _planner_response_format 字面量重复。
    from . import roles_config as RC
    agent = RC.make_agent("planner")
    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw = agent.run(messages)
    if not raw:
        return None
    plan = _extract_json_object(raw)
    if plan is None:
        logger.info('%s', '[planner] 输出无法解析为 JSON 契约，降级为本地自规划。')
        return None
    norm = _normalize_plan(plan)
    if norm is None:
        logger.info('%s', '[planner] 契约结构不合格（缺 tasks/modules），降级为本地自规划。')
        return None
    return norm


def _apply_plan(plan: Dict[str, Any]) -> None:
    """把契约写入 GLOBAL_STATE：plan + 任务列表 + planning_done（跳过本地自规划）。

    任务已先经 _normalize_plan 规范为 {'desc', 'deliverables', 'status'} 对象。
    完成标准只看工作区测试是否全绿，不再有隐藏测试/交付物校验。
    """
    GLOBAL_STATE["plan"] = plan
    GLOBAL_STATE["tasks"] = [
        {"desc": t["desc"], "deliverables": t.get("deliverables", []), "status": "pending"}
        for t in plan["tasks"]
    ]
    GLOBAL_STATE["planning_done"] = True
    GLOBAL_STATE["done_list"] = []
    # 持久化 verify_points 到工作区（独立 tester 读取，且可人工复核）
    _save_verify_points(plan.get("verify_points") or [])


def plan_contract_section() -> str:
    """把 GLOBAL_STATE['plan'] 渲染为系统提示的『顶层契约』段（无 plan 则空串）。"""
    plan = GLOBAL_STATE.get("plan")
    if not plan:
        return ""
    lines = ["# 顶层 Planner 契约（远程大模型制定，必须遵守）",
             f"方案概述：{plan.get('summary', '')}", ""]
    if plan.get("modules"):
        lines.append("文件白名单（只允许创建/修改以下文件及测试文件，禁止新增契约外模块）：")
        for m in plan["modules"]:
            pub = ("；接口：" + "；".join(m["public"])) if m.get("public") else ""
            lines.append(f"  - {m['path']}：{m.get('purpose', '')}{pub}")
        lines.append("")
    if plan.get("interface"):
        lines.append("必须实现的接口：")
        lines += [f"  - {x}" for x in plan["interface"]]
        lines.append("")
    if plan.get("forbidden"):
        lines.append("禁止事项：")
        lines += [f"  - {x}" for x in plan["forbidden"]]
        lines.append("")
    lines.append("你不需要再输出 plan 动作（任务清单已由顶层 Planner 给出），"
                 "直接从第一个任务开始【填空式实现】：按任务顺序逐个完成，每轮一个动作。")
    return "\n\n" + "\n".join(lines)


def run_planner(user_task: str, research_findings: str = "") -> bool:
    """Planner 闸门。成功返回 True（契约已写入 GLOBAL_STATE）。失败返回 False（降级本地自规划）。
    research_findings 来自 Analyzer 阶段，注入 Planner 以免重复探查。"""
    mid = M.role_model_id("planner")
    if not mid:
        return False
    logger.info('%s', f'[planner] 调用 Planner（{mid}）生成执行契约……')
    plan = _make_plan(user_task, research_findings=research_findings)
    if plan is None:
        logger.info('%s', '[planner] Planner 不可用，回退本地自规划（优雅降级）。')
        return False
    _apply_plan(plan)
    logger.info('%s', '=====plan json======\n' + json.dumps(plan))
    GLOBAL_STATE["lang"] = plan.get("language") or "python"
    _telemetry("glm_plan")
    logger.info('%s', f"[planner] 契约已生成：{len(plan['modules'])} 个模块、{len(plan['tasks'])} 个任务、{len(plan['forbidden'])} 条禁止事项。")
    for i, t in enumerate(plan["tasks"], 1):
        dl = t.get("deliverables") or []
        dl_s = (" → " + ", ".join(dl)) if dl else ""
        logger.info('%s', f"    {i}. {t['desc']}{dl_s}")
    vps = plan.get("verify_points") or []
    logger.info('%s', f'[planner] 已生成 {len(vps)} 条语义验收点（verify_points，供独立 tester 验证）：')
    for v in vps:
        logger.info('%s', f"    - [{v.get('id')}] {v.get('point')}" + (f"  (hint: {v.get('check_hint')})" if v.get('check_hint') else ''))
    return True


def _save_verify_points(vps: List[Dict[str, Any]]) -> None:
    """把语义验收点持久化到工作区 .swe_verify.json（独立 tester 读取，可人工复核）。"""
    try:
        import json as _json
        from . import config as _C
        (_C.WORKSPACE).mkdir(parents=True, exist_ok=True)
        (_C.WORKSPACE / ".swe_verify.json").write_text(
            _json.dumps({"verify_points": vps}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception as e:
        logger.info('%s', f'[planner] verify_points 持久化失败（忽略）：{e}')


# ---------------- 漂移检测（磁盘 vs plan 契约） ----------------

_drift_reported: set = set()      # 已回灌过的漂移项（避免重复刷屏）


def drift_issues() -> List[str]:
    """磁盘实际代码文件 vs 契约白名单：契约外新建文件 = 漂移。返回新出现的漂移项。"""
    plan = GLOBAL_STATE.get("plan") or {}
    planned = {m.get("path") for m in (plan.get("modules") or []) if m.get("path")}
    if not planned:
        return []   # 无契约（本地自规划）不做漂移检测
    issues: List[str] = []
    try:
        ws = C.WORKSPACE.resolve()
        for p in sorted(ws.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in C.DRIFT_CODE_EXTS:
                continue
            if any(part in _HEAVY_DIRS for part in p.parts):
                continue
            try:
                rel = str(p.relative_to(ws))
            except Exception:
                continue
            # 测试文件不算漂移（契约允许额外写测试）
            if C._TEST_FILE_RE.search(rel) or rel.startswith("tests/"):
                continue
            if rel not in planned:
                issues.append(rel)
    except Exception:
        return []
    return [i for i in issues if i not in _drift_reported]


# ---------------- 故障路由（严重失败 → 退回 Planner 重规划） ----------------

# ---------------- Analyzer：只读探查（tools-only，无 response_format） ----------------

_ANALYZER_SYSTEM = """你是软件需求分析师（Requirements Analyst）。
职责：你不写代码，你负责把用户的需求、问题、任务理解透彻，并补齐必要的领域知识，输出一份完整且【克制的】需求/问题/任务分析与关键资料摘要/核对/根因分析，供下游 Planner / 实现者使用。

当前目录就是工作目录， 禁止换目录。 

工作方式：
- 你被授权按需调用工具来辅助理解，仅在必要时调用工具，不要为了调用而调用。
- 先吃透用户需求。遇到你【不了解或无法确认】的需求（某算法的具体规则、某库 / API 的用法等），先收集资料再下结论：可借助联网检索，也可阅读当前代码来加深理解；不要凭猜测给出不准确的结论。
- 对于常识性需求、无需查证的需求（如常见标准库功能），无需联网，直接基于任务描述整理即可。
- 如果已有相关的代码，需要运行单元测试，分析需求当前代码的缺陷。

输出要求（理解充分后调用结束动作提交摘要）：
- 克制：只输出与用户描述直接相关的需求要点 + 关键参考资料，不要写实现方案、不要写代码。
- 输出应贴合用户描述。例如「用 Python 实现斐波那契」应输出：斐波那契数列的计算规则（F(0)=0、F(1)=1、F(n)=F(n-1)+F(n-2)）、常见的边界约定等；「实现康威生命游戏」应输出：生存 / 死亡规则（存活邻居数为 2 或 3 则存活、恰好 3 则新生等）。
- 若工作区已有相关代码，指出其设计思路、接口形状、测试约定、潜在坑点等事实。
"""



# Analyzer 工具集走与主集统一的 registry（glm_tools("analyzer")）：read_file / grep /
# glob / web_search / finish_analysis，全部由 ToolRegistry 自描述驱动，不再单独硬编码 schema。


def _analyzer_summarize_observation(act: str, args: dict, res) -> str:
    """把一次工具探查压成一行可核对的事实（兜底摘要的唯一来源，禁止推断/编造）。"""
    out = str(res)
    if act == "read_file":
        lines = out.count("\n") + 1
        return f"已读取 {args.get('path', '?')}（{lines} 行输出）"
    if act == "glob":
        names = [x.strip() for x in out.splitlines() if x.strip()]
        return (f"工作区文件：{', '.join(names[:12])}"
                if names else f"glob {args.get('pattern', '')!r} 无匹配")
    if act == "grep":
        hits = len([x for x in out.splitlines() if x.strip()])
        return f"grep {args.get('pattern', '')!r} → {hits} 行匹配"
    if act == "web_search":
        return f"web_search {args.get('query', '')!r} → {len(out)} 字符结果"
    return f"{act} → {out[:120]}"


def _analyzer_collect_observations(cm) -> "List[str]":
    """从对话 buffer 重建 analyzer 的工具观察事实（finish_analysis 等 meta 动作不计入）。"""
    msgs = cm.to_list()
    results = {}
    for m in msgs:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            results[m["tool_call_id"]] = m.get("content", "")
    obs = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {}).get("name")
                if fn in ("finish_analysis",):
                    continue
                try:
                    a = json.loads(tc.get("function", {}).get("arguments") or "{}")
                except Exception:
                    a = {}
                res = results.get(tc.get("id"), "")
                obs.append(_analyzer_summarize_observation(fn, a, res))
    return obs


def analyzer_on_iter_end(ctx, reason: str):
    """analyzer 收敛闸门（on_iter_end hook）：统一 loop 驱动，不再手写 per-role 主循环。

    逐轮判定（替代原 _run_pass 内散落的重复/软空/散文 judge 逻辑）：
      - all_done（finish_analysis 命中）→ 捕获结构化摘要 → 'done'
      - model_error（模型连续不可用）→ 'break'（外层 R1：换 fallback 模型重跑）
      - continue 且 empty_streak 超 ANALYZER_SOFT_EMPTY_LIMIT（软空）→ 'break'（R1）
      - continue/stuck 且本轮有未调工具的散文 → M.judge 评测：
          yes → 采纳为发现 → 'done'；no/notsure → 回灌反馈续轮（judge_no 超轮次 → 'break'→R4）
      - stuck（重复动作 / 无工具停滞）→ 'break'（外层 R4：用观察事实兜底）
    """
    g = ctx.guard
    md = ctx.metadata
    # 终止动作命中：捕获 finish_analysis 的结构化摘要
    if reason == "all_done":
        sr = md.get("stop_result")
        if sr:
            block = (sr.get("summary") or "").strip()
            kf = sr.get("key_findings") or []
            if isinstance(kf, list) and kf:
                block += "\n要点：\n" + "\n".join(f"  - {x}" for x in kf)
            if block.strip():
                md.setdefault("analyzer_findings", []).append(block.strip())
        md["analyzer_outcome"] = "submitted"
        return "done"
    # 模型不可用（chat_toolcalls 重试后仍 None）
    if reason == "model_error":
        md["analyzer_outcome"] = "model_dead"
        return "break"
    # 软空响应累计超阈值（finish_reason=stop 但无 tool_calls/content）
    if reason == "continue" and g.value("empty_streak") > C.ANALYZER_SOFT_EMPTY_LIMIT:
        md["analyzer_outcome"] = "model_dead"
        return "break"
    # 纯散文（未调工具）→ judge 模型判定是否算有效调研
    prose = md.get("last_no_tool_content", "")
    if reason in ("continue", "stuck") and prose:
        jres, jreason = M.judge("analyzer", prose)
        if jres == "yes":
            md.setdefault("analyzer_findings", []).append(prose)
            md["analyzer_outcome"] = "submitted"
            return "done"
        md["analyzer_judge_no"] = md.get("analyzer_judge_no", 0) + 1
        if md["analyzer_judge_no"] > C.ANALYZER_JUDGE_ROUNDS:
            # 不采纳被拒散文，保持显式空语义；外层 R4 兜底
            md["analyzer_outcome"] = "exhausted"
            return "break"
        if jres == "notsure" and md.get("analyzer_judge_unsure", 0) < C.JUDGE_MAX_RETRY:
            md["analyzer_judge_unsure"] = md.get("analyzer_judge_unsure", 0) + 1
            ctx.cm.append("user", "（系统）无法判定你的调研是否合格，请补充更具体的概念解释与需求说明后再次提交。")
        else:
            ctx.cm.append("user", f"（系统）你的调研还不合格（判定：{jres}）：{jreason}。"
                                  f"请补充更具体的概念解释与需求说明，或调用工具探查工作区后再提交。")
        return "continue"
    # 重复动作 / 无工具停滞且无可用散文 → 视为 exhausted（外层 R4 兜底）
    if reason == "stuck":
        md["analyzer_outcome"] = "exhausted"
        return "break"
    return "continue"


def run_analyzer(user_task: str, max_steps: int = 10) -> str:
    """只读调研阶段（tools-only，原生 OpenAI toolcall 协议，与 executor/tester 一致）。

    走统一 Agent loop（make_agent + single_loop + on_iter_end 收敛闸门），analyzer 不再
    手写 per-role 主循环。quota（repeat/soft-empty）已在 config 放宽（ANALYZER_MAX_ITER /
    ANALYZER_REPEAT_THRESHOLD）；工具调用失败默认回灌、不计调用次数（对齐「确认失败就不算
    幂等提交次数」）。

    终止条件：
      - 模型调用 finish_analysis → 汇总并返回摘要；
      - 模型散文经 judge 判 yes → 采纳为发现；
      - 主模型不可用 / 软空（model_dead）→ R1 换 fallback 模型重跑完整主循环；
      - 仍无发现 → R4 用【实际观察到的工具事实】合成最小摘要；一个事实都没有才返回空串。
    """
    if not M.role_model_id("analyzer"):
        return ""
    main_id = M.role_model_id("analyzer")
    fb_id = M.role_fallback("analyzer")
    logger.critical('analyzer_start model=%s fallback=%s task_len=%s max_steps=%s ws_lines=%s',
                    main_id, fb_id or '(none)', len(user_task), max_steps, _clip(_workspace_listing(), 120))

    def _base_msgs():
        return [
            {"role": "system", "content": _ANALYZER_SYSTEM},
            {"role": "user", "content": f"【开发任务】\n{user_task}\n\n【当前工作区文件清单】\n{_workspace_listing()}"},
        ]

    # —— 主循环：统一 Agent loop + analyzer_on_iter_end 收敛闸门 ——
    loop = RC.single_loop(
        max_iter=max(1, max_steps, C.ANALYZER_MAX_ITER),
        repeat_threshold=C.ANALYZER_REPEAT_THRESHOLD,
        no_tool_threshold=C.ANALYZER_NO_TOOL_ROUNDS + 1,
        on_iter_end=analyzer_on_iter_end,
    )
    agent = RC.make_agent("analyzer", loop=loop)
    agent.ctx.cm.set_system(_ANALYZER_SYSTEM)
    agent.ctx.cm.append("user", f"【开发任务】\n{user_task}\n\n【当前工作区文件清单】\n{_workspace_listing()}")
    reason = agent.run()
    findings = list(agent.ctx.metadata.get("analyzer_findings", []))
    outcome = agent.ctx.metadata.get("analyzer_outcome", reason)
    last_cm = agent.ctx.cm

    # —— R1：主循环未产出发现 且 model_dead（主模型不可用 / 软空）→ 换 fallback 模型重跑 ——
    # 统一收敛：主模型 exhausted（可达但不收敛）也走同一 fallback 重跑，等价于原 R1+R5，
    # 避免旧代码「同 prompt 死等 + 孤立 1 次强制调用」的旁路 bug。
    if not findings and outcome == "model_dead" and fb_id:
        logger.critical('analyzer_fallback from_model=%s to_model=%s reason=%s', main_id, fb_id, outcome)
        fb_loop = RC.single_loop(
            max_iter=max(1, max_steps, C.ANALYZER_MAX_ITER),
            repeat_threshold=C.ANALYZER_REPEAT_THRESHOLD,
            no_tool_threshold=C.ANALYZER_NO_TOOL_ROUNDS + 1,
            on_iter_end=analyzer_on_iter_end,
        )
        fb_agent = RC.make_agent("analyzer", loop=fb_loop, model_override=fb_id)
        fb_agent.ctx.cm.set_system(_ANALYZER_SYSTEM)
        fb_agent.ctx.cm.append("user", f"【开发任务】\n{user_task}\n\n【当前工作区文件清单】\n{_workspace_listing()}")
        fb_agent.ctx.cm.append("user", "请立即调用 finish_analysis 工具，给出简洁的调研发现摘要"
                                     "（已有接口/依赖/潜在坑点/测试约定）；不要写实现方案。若工作区为空，"
                                     "只说明这是从零开始的全新任务即可。")
        fb_agent.run()
        findings = list(fb_agent.ctx.metadata.get("analyzer_findings", []))
        outcome = fb_agent.ctx.metadata.get("analyzer_outcome", "exhausted")
        last_cm = fb_agent.ctx.cm

    if findings:
        logger.critical('analyzer_done outcome=%s source=%s findings=%s returned_len=%s',
                        outcome, 'findings', len(findings), len('\n\n'.join(findings).strip()))
        return "\n\n".join(findings).strip()

    # —— R4 兜底：始终不交卷时，用【实际观察到的事实】合成最小摘要 ——
    if last_cm is not None:
        observations = _analyzer_collect_observations(last_cm)
        if observations:
            seen, uniq = set(), []
            for o in observations:
                if o not in seen:
                    seen.add(o)
                    uniq.append(o)
            synth = ("（analyzer 未提交结构化结论；以下为探查过程中实际观察到的事实，供规划参考）\n"
                     + "\n".join(f"- {o}" for o in uniq[:20]))
            logger.critical('analyzer_r4_synth n_obs=%s returned_len=%s snippet=%s',
                            len(uniq), len(synth), _clip(synth, 200))
            return synth
    logger.critical('analyzer_done outcome=%s source=%s findings=%s obs=%s', outcome, 'empty', 0, 0)
    return ""

