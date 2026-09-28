#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/registry.py —— 自描述（self-describing）工具与传感器注册中心

这是分层架构的核心抽象层：

1) 工具层（Worker 工具集群）自描述
   每个工具用 ToolDef 声明：名称 / 类别 / 描述 / 参数 JSON Schema / 实现函数 /
   是否危险 / 示例 / 何时使用。系统提示词里的「工具清单」由 ToolRegistry.prompt_fragment()
   依据这些声明动态生成，模型看到的工具说明与代码实现单点定义、永不同步。

2) 客观校验层（Harness 传感器）自描述
   每个传感器继承 BaseSensor 并声明：名称 / 描述 / 严重级别 / 是否启用。
   统一输出 SensorFact 结构化事实，流水线由 SensorRegistry.run_pipeline() 强制触发，
   调度层只消费 fact，不感知底层技术栈差异（LSP / pytest / 导入校验等）。

设计原则（对齐 SWE-Agent 架构文档）：
- 感知层永远是代码（传感器），绝不交给 LLM；
- 调度模型只消费结构化 Fact，不读原始日志；
- 工具/传感器自描述，插件化扩展无需改动内核。
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from . import config as C
from .hooks import (
    HookPoint,
    GateAction,
    HookHub,
    HOOK_HUB,
    _legacy_adapter,
)
from swe_agent.log import logger

# 旧魔法字符串 phase → HookPoint 锚点（保留旧 register_global_hook 调用方零改动）。
_PHASE_TO_POINT = {
    "before": HookPoint.BEFORE_TOOL_CALL,
    "after": HookPoint.AFTER_TOOL_CALL,
    "on_success": HookPoint.AFTER_TOOL_CALL,
    "on_fail": HookPoint.AFTER_TOOL_CALL,
    "pre_loop": HookPoint.RUN_START,
    "post_loop": HookPoint.FINALLY_TOOL_CALL,
    "on_error": HookPoint.ERROR_TOOL_CALL,
}


# ======================================================================
# 零、执行上下文 + 钩子类型
# ======================================================================
@dataclass
class ActionContext:
    """传递给工具 run 与 hooks 的统一执行上下文（拼装/测试的单一接口）。

    - messages：当前对话历史（compact 等动作原地修改它）
    - system_extras：run_agent 注入的系统提示补丁（fault_route 用）
    - extra：自定义扩展位
    - action：dispatch 在执行钩子前填好，钩子可用 ctx.action 读取本次动作名
    - loop_break：BEFORE_TOOL_CALL 门控返回 GateAction.BREAK_LOOP 时由 dispatch 置 True，
      供 loop 层消费（Y-a：REJECT=回灌 reason 继续；BREAK_LOOP=回灌 reason 并终止当前 loop）。
    """
    messages: Optional[List[Dict[str, Any]]] = None
    system_extras: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    action: str = ""
    loop_break: bool = False


# 钩子签名统一为 (ctx, params, result) -> Optional[str]
# - before 阶段：result 固定为 None；返回字符串 = 拦截（短路，作为最终结果）；
#   返回 None = 放行。
# - after / on_success / on_fail 阶段：result 为当前结果；返回字符串 = 覆盖结果；
#   返回 None = 保持原结果。
HookType = Callable[[ActionContext, Dict[str, Any], Optional[str]], Optional[str]]


# ======================================================================
# 一、自描述工具（Worker 工具集群）
# ======================================================================
@dataclass
class ToolDef:
    """单个工具的自描述定义。"""
    name: str                                   # 动作名（JSON 里的 "action"）
    description: str                            # 给模型看的功能描述（用于生成提示词）
    category: str                              # 分类：fs / shell / web / interact / meta
    schema: Dict[str, Any]                      # 参数 JSON Schema
    run: Callable[..., str]                     # 实现：返回结果文本
    dangerous: bool = False                     # 是否需要危险命令确认
    examples: List[str] = field(default_factory=list)   # 调用示例
    when_to_use: str = ""                       # 何时使用（prompt 附加提示）
    # —— 可拼装钩子：有就执行，没有就不执行 ——
    before: List[HookType] = field(default_factory=list)       # 执行前（可拦截）
    after: List[HookType] = field(default_factory=list)        # 执行后（可覆盖结果）
    on_success: List[HookType] = field(default_factory=list)  # 成功时
    on_fail: List[HookType] = field(default_factory=list)     # 失败时


class ToolRegistry:
    """工具注册中心：登记、查询、生成提示词片段、分发执行。

    全局钩子（对所有动作生效）统一收口到 `hooks.HOOK_HUB`（按 HookPoint 锚点聚合），
    不再用魔法字符串 phase 字典。工具级钩子（before/after/on_success/on_fail）仍挂在
    ToolDef 上，作为派发内联逻辑（其行为与历史完全一致，test_registry_dispatch 锁定）。
    """

    _tools: Dict[str, ToolDef] = {}

    @classmethod
    def register(cls, tool: ToolDef) -> ToolDef:
        cls._tools[tool.name] = tool
        return tool

    @classmethod
    def unregister(cls, name: str) -> None:
        """移除已注册工具（插件从清单移除 / 重载前清理时使用）。"""
        cls._tools.pop(name, None)

    @classmethod
    def get(cls, name: str) -> Optional[ToolDef]:
        return cls._tools.get(name)

    @classmethod
    def all(cls) -> List[ToolDef]:
        return list(cls._tools.values())

    @classmethod
    def names(cls) -> List[str]:
        return list(cls._tools.keys())

    @classmethod
    def prompt_fragment(cls) -> str:
        """依据各工具的自我声明，动态生成「可用工具」提示词片段（自描述）。"""
        if not cls._tools:
            return "（当前未注册任何工具）"
        lines: List[str] = []
        for t in cls.all():
            if t.name in _INTERNAL_TOOLS:
                continue
            lines.append(f"- {t.name}（{t.category}）：{t.description}")
            if t.when_to_use:
                lines.append(f"    何时用：{t.when_to_use}")
            # 参数 schema -> 一行摘要
            props = t.schema.get("properties", {})
            req = set(t.schema.get("required", []))
            if props:
                params = ", ".join(
                    f"{k}:{v.get('type', 'any')}{'*' if k in req else ''}"
                    for k, v in props.items()
                )
                lines.append(f"    参数：{params}")
            if t.examples:
                for ex in t.examples[:3]:
                    lines.append(f"    示例：{ex}")
        return "\n".join(lines)

    @classmethod
    def dispatch(cls, action: Dict[str, Any], ctx: Optional["ActionContext"] = None) -> str:
        """把模型输出的单个动作对象路由到对应工具实现，并按注册顺序执行钩子。

        执行顺序（每个阶段都是「有就执行，没有就不执行」）：
          1. before 钩子（全局 + 工具级）：返回字符串 = 拦截短路；否则放行
          2. 工具本体 run（自动区分是否接受 ctx）
          3. after 钩子（全局 + 工具级）：返回字符串 = 覆盖结果
          4. 成功 → on_success；失败 → on_fail（全局 + 工具级）
        """
        if not isinstance(action, dict):
            return "error: 动作对象必须是 dict"
        name = action.get("action")
        t = cls.get(name)
        if t is None:
            return f"error: 未知动作 {name!r}，可用动作：{', '.join(cls.names())}"
        if ctx is None:
            ctx = ActionContext()
        ctx.action = name or ""
        params = {k: v for k, v in action.items() if k != "action"}

        # 1) before：全局(HookHub) 先跑（含拦截副作用），任一 REJECT/BREAK_LOOP/str 短路；
        #    工具级 before 随后跑（历史顺序：全局先于工具）。所有 before 副作用都先发生，
        #    即便被拦截（test_registry_dispatch「拦截前全局 before 仍执行」）。
        #    Y-a：first_block_decision 保留 GateAction 语义——REJECT 仅回灌 reason、loop 继续；
        #    BREAK_LOOP 置 ctx.loop_break、回灌 reason、由 loop 层终止当前 loop。
        block = HookHub.first_block_decision(
            HOOK_HUB.emit(HookPoint.BEFORE_TOOL_CALL, ctx=ctx, params=params, tool=name)
        )
        if block is not None:
            if block.action == GateAction.BREAK_LOOP:
                ctx.loop_break = True
            return block.reason or "rejected_by_hook"
        for h in t.before:
            block = h(ctx, params, None)
            if isinstance(block, str):
                return block

        # 2) 工具本体
        result = cls._invoke(t, ctx, params)

        # 3) after：全局(HookHub) + 工具级，str 覆盖结果
        result = HookHub.override_of(
            HOOK_HUB.emit(HookPoint.AFTER_TOOL_CALL, ctx=ctx, params=params, result=result, tool=name),
            result,
        )
        for h in t.after:
            ov = h(ctx, params, result)
            if isinstance(ov, str):
                result = ov

        # 4) on_success / on_fail（仅工具级，条件路由；全局 on_success/on_fail 历史为空，已并入 after）
        if cls._is_success(result):
            hooks = t.on_success
        else:
            hooks = t.on_fail
        for h in hooks:
            ov = h(ctx, params, result)
            if isinstance(ov, str):
                result = ov
        return result

    @classmethod
    def _invoke(cls, t: "ToolDef", ctx: "ActionContext", params: Dict[str, Any]) -> str:
        """调用工具 run；自动区分函数是否接受 ctx，并按签名过滤多余参数。"""
        try:
            sig = inspect.signature(t.run)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                                 for p in sig.parameters.values())
            if "ctx" in sig.parameters:
                if accepts_var_kw:
                    return t.run(ctx=ctx, **params)
                accepted = [p for p in sig.parameters if p != "ctx"]
                filtered = {k: v for k, v in params.items() if k in accepted}
                return t.run(ctx=ctx, **filtered)
            if accepts_var_kw:
                return t.run(**params)
            accepted = list(sig.parameters.keys())
            filtered = {k: v for k, v in params.items() if k in accepted}
            return t.run(**filtered)
        except TypeError as e:
            return f"error: 工具 {t.name} 参数错误：{e}"
        except Exception as e:
            # 已注册但运行时不可用（网络未通 / MCP 未配置 / 插件未启用 / 内部异常）：
            # 返回结构化回执，让模型拿到「工具暂时不可用」并自行换策略，而非让异常冒泡
            # 炸掉 agent loop 或退化为 None / 空响应。
            reason = f"{type(e).__name__}: {str(e)[:300]}"
            return f"error: 工具 {t.name} 暂时不可用：{reason}"

    @classmethod
    def _is_success(cls, result: str) -> bool:
        """以结果文本前缀粗略判定动作成败（供 on_success/on_fail 路由）。"""
        if not isinstance(result, str):
            return True
        fail_prefixes = (
            "error:", "denied:", "unknown", "未知动作",
            "write_error", "edit_error", "read_error", "shell_error",
            "grep_error", "glob_error", "web_fetch_error", "web_search_error", "ocr_error",
            "write_rejected", "compact_error", "task_error", "task_stopped:",
            "plan_mode_blocked:", "mcp_error", "skill_error", "lsp_error",
        )
        return not result.startswith(fail_prefixes)

    @classmethod
    def register_global_hook(cls, phase: str, fn: "HookType") -> None:
        """注册一个全局钩子。

        phase ∈ 工具级(before/after/on_success/on_fail) 或 循环级(pre_loop/post_loop/on_error)。
        工具级钩子签名 fn(ctx, params, result)；循环级钩子签名 fn(loop_ctx)。
        """
        point = _PHASE_TO_POINT.get(phase)
        if point is None:
            raise ValueError(
                f"未知钩子阶段：{phase}"
                f"（应为 before/after/on_success/on_fail/pre_loop/post_loop/on_error）")
        HOOK_HUB.on(point, _legacy_adapter(fn))

    @classmethod
    def run_loop_hooks(cls, phase: str, loop_ctx: Any = None) -> None:
        """运行循环级钩子（pre_loop/post_loop/on_error）。

        这是 GLOBAL 机制（补充③）：把跨切面的判断/功能（统计、重置、错误升级）
        从主循环里抽出来，统一在此注册与执行，便于维护。
        钩子签名 fn(loop_ctx) -> None（可改 loop_ctx 或读全局状态），返回值忽略。
        """
        point = _PHASE_TO_POINT.get(phase)
        if point is None:
            return
        HOOK_HUB.emit(point, loop_ctx=loop_ctx)

    @classmethod
    def clear_global_hooks(cls, phase: Optional[str] = None) -> None:
        """清空全局钩子；phase 为 None 时清空全部。"""
        if phase is None:
            for p in HookPoint:
                HOOK_HUB.clear(p)
        else:
            point = _PHASE_TO_POINT.get(phase)
            if point is not None:
                HOOK_HUB.clear(point)

    @classmethod
    def glm_tools(cls, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """按角色生成「每动作一个 function」的原生 OpenAI toolcall schema（自描述）。

        每个已注册 ToolDef 生成一个独立 function：name=工具名、description=工具描述、
        parameters=工具自有 schema。这样工具描述（含参数说明）真正进入 function-calling
        schema —— 模型在 ``tool_choice:"required"`` 下贴着用的是 schema，而非远在开头的系统提示，
        本地 7B 模型的调用准确率显著高于原来「单一巨型 agent_action + 扁平参数袋」的设计。

        role 指定时只返回该角色允许的工具子集（per-role toolset）；role=None 返回全部（向后兼容）。
        新增一个工具只需在 tools.py / supervisor.py 注册一次，文本提示与 schema 同步生成。
        """
        allowed = ROLE_TOOLS.get(role)  # None = 不裁剪
        # 无人值守模式：BUILD 层摘除 ask 工具（全角色），模型根本调不到，
        # 避免后台/batch 下 input() 卡死 stdin。交互模式下照常下发。
        gate_ask = C.UNATTENDED_MODE
        fns: List[Dict[str, Any]] = []
        for t in cls.all():
            if allowed is not None and t.name not in allowed:
                continue
            if gate_ask and t.name == "ask":
                continue
            fns.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                },
            })
        # 惰性校验：确认 allow-list 与 registry 一致（首次调用 glm_tools 时跑一次）
        cls._maybe_validate_role_tools(role)
        return fns

    @classmethod
    def _maybe_validate_role_tools(cls, role: Optional[str] = None) -> None:
        """惰性校验 allow-list 与 registry 的一致性（导入期之外、首次调用 glm_tools 时跑）。

        仅打印警告、不抛异常：
        - unknown 检查只针对【本次请求的角色】——不同入口（hy3 supervisor / forge）
          注册的工具集不同，验别的角色只会产生噪声；
        - 动态注入的工具（web_search/web_fetch/ocr 由插件运行时注册）列入白名单跳过；
        - orphan 检查针对当前已注册工具（发现「注册了却谁都用不了」的死角）。
        """
        global _ROLE_VALIDATED
        if _ROLE_VALIDATED:
            return
        _ROLE_VALIDATED = True
        registered = set(cls.names())
        # 1) 本角色 allow-list 名字必须已注册（动态工具除外，可能运行时才注入）
        allowed = ROLE_TOOLS.get(role) or set()
        unknown = sorted(n for n in allowed
                         if n not in registered and n not in _DYNAMIC_TOOL_NAMES)
        if unknown:
            logger.info('%s', f'[registry] 警告：角色 {role} 的 ROLE_TOOLS 引用了未注册的工具（将被静默忽略）：{unknown}')
        # 2) 已注册工具应至少被一个角色暴露（orphan 检查，避免注册了却谁都用不了）
        used = set(n for roleset in ROLE_TOOLS.values() for n in roleset)
        orphans = sorted(n for n in registered if n not in used)
        if orphans:
            logger.info('%s', f'[registry] 警告：以下已注册工具未被任何角色暴露（orphan，模型无法调用）：{orphans}')


# ======================================================================
# 每个 agent 角色允许的工具集（per-role toolset）
# 设计意图：不同角色只看到自己该用的工具，避免小模型在 28 个动作里迷路、
# 也避免 Planner 误拿到 write_file/shell、Executor 误拿到 plan 这类噪声/风险动作。
# 角色名与 models.py / supervisor.py 的 SUBAGENT_ALLOWED 保持一致。
# 插件动态注入的工具（web_search/web_fetch/lsp/mcp_tool 等）只要已注册即自动纳入。
# ======================================================================
ROLE_TOOLS = {
    # forge v2 的唯一角色：单 driver（qwen-7b）。工具面 = 通用软件工程师所需的最小全集：
    # fs + shell + 交互 + todo + 完成 + 插件扩展（skill/lsp/mcp/web）。
    "driver": {
        "read_file", "write_file", "edit_file", "shell", "grep", "glob",
        "ask", "todo_write", "todo_read", "done", "verify",
        "skill", "lsp", "mcp_tool", "list_mcp_resources", "read_mcp_resource",
        "reload_plugins", "local_search",
        # "web_search", "web_fetch", "ocr",
    },
    # forge v2 独立验收 agent（tester）：严格只读工具集 + 提交工具。
    # 故意不含任何 write/edit 类动作——BUILD 层硬控，杜绝「自己验自己」。
    "tester": {
        "read_file", "glob", "grep", "shell", "finish_verify", "local_search",
    },
    "planner": {
        "plan", "todo_write", "todo_read", "complete", "ask", "report",
        "read_file", "grep", "glob", "local_search",
        #   "web_search", "web_fetch",
        "compact", "reload_plugins", 'shell',
    },
    "executor": {
        "write_file", "edit_file", "read_file", "grep", "glob", "shell",
        "ask", "report", "complete", "verify", "mcp_tool", "lsp",
        "list_mcp_resources", "read_mcp_resource", "skill", "local_search",
        # "web_fetch", "web_search", # websearch, webfetch is running out of budget  
        "compact", "reload_plugins",
    },
    "analyzer": {
        # "read_file", "grep", "glob", "web_search", "finish_analysis",
        "read_file", "grep", "glob", "finish_analysis", "shell", "lsp", "local_search", # web_search is running out of budget
    },
    # 子智能体（与 supervisor.SUBAGENT_ALLOWED 对齐）
    "explore": {
        "read_file", "grep", "glob", "report", "local_search", #"web_fetch" # web_fetch is running out of budget, 
    },
    "plan": {"read_file", "grep", "glob", "local_search",
            #  "web_fetch", 
             "report", "agent"},
    "general-purpose": {"read_file", "write_file", "edit_file", "shell",
                        "grep", "glob", "local_search",
                        # "web_fetch", 
                        "report",
                        "task_output", "task_stop", "sleep"},
}


# allow-list 惰性校验用：已跑过一次后不再重复打印
_ROLE_VALIDATED = False
# 由插件运行时动态注入的工具，首次 glm_tools 调用时可能尚未注册，列入白名单避免误报。
_DYNAMIC_TOOL_NAMES = {"web_search", "web_fetch", "ocr"}
# 内部元动作：仅特定角色用，不进通用「工具清单」文本提示（避免非 analyzer 角色看到）
_INTERNAL_TOOLS = {"finish_analysis"}


def tool(name: str, description: str, category: str, schema: Dict[str, Any],
         dangerous: bool = False, examples: Optional[List[str]] = None,
         when_to_use: str = "",
         before: Optional[List["HookType"]] = None,
         after: Optional[List["HookType"]] = None,
         on_success: Optional[List["HookType"]] = None,
         on_fail: Optional[List["HookType"]] = None):
    """装饰器：把函数注册为自描述工具，并可选挂载 before/after/on_success/on_fail 钩子。"""
    def deco(fn: Callable[..., str]) -> Callable[..., str]:
        ToolRegistry.register(ToolDef(
            name=name, description=description, category=category,
            schema=schema, run=fn, dangerous=dangerous,
            examples=examples or [], when_to_use=when_to_use,
            before=before or [], after=after or [],
            on_success=on_success or [], on_fail=on_fail or [],
        ))
        return fn
    return deco


# Analyzer 专用「结束分析」动作。注册为普通 ToolDef，使其与所有工具走同一条
# registry -> glm_tools 路径（不再在 glm_tools 里硬编码 schema）。run 由 analyzer
# 主循环特判拦截，这里只提供占位实现以避免 dispatch 报错。
@tool(
    name="finish_analysis",
    category="meta",
    description="结束只读分析，给出结构化发现摘要（只摆事实，不要写实现方案）。",
    schema={"type": "object", "properties": {
        "summary": {"type": "string",
                    "description": "分析发现摘要：接口形状、相关实现位置、测试约定、潜在坑点等事实"},
        "key_findings": {"type": "array", "items": {"type": "string"},
                         "description": "要点列表"},
    }, "required": ["summary"]},
)
def _finish_analysis_tool(ctx: ActionContext, summary: str = "", key_findings=None) -> str:
    return f"[finish_analysis] 摘要已记录（{len(summary or '')} 字）。"


# Tester 专用「提交验收结论」动作。注册为普通 ToolDef 走 registry -> glm_tools 路径；
# run 由 verify.run_tester 主循环特判拦截（读取 results 后结束），这里只提供占位实现避免 dispatch 报错。
@tool(
    name="finish_verify",
    category="meta",
    description="结束独立验收，提交逐条判定（只读 agent 的唯一出口）。",
    schema={"type": "object", "properties": {
        "results": {"type": "array", "items": {"type": "object"},
                    "description": "每条验收点的判定：{id:int, verdict:'pass'|'fail'|'skipped', evidence:str}"},
    }, "required": ["results"]},
)
def _finish_verify_tool(ctx: ActionContext, results: Optional[List[Dict[str, Any]]] = None) -> str:
    return f"[finish_verify] 已收到 {len(results or [])} 条验收判定。"


# ======================================================================
# 二、自描述传感器（Harness 客观校验层）
# ======================================================================
@dataclass
class SensorFact:
    """单条结构化校验事实（调度层唯一消费对象）。"""
    sensor_name: str
    ok: bool
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"   # error / warning / info

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_name": self.sensor_name,
            "ok": self.ok,
            "message": self.message,
            "payload": self.payload,
            "severity": self.severity,
        }


@dataclass
class HarnessContext:
    """传感器运行上下文：工作区路径 + 当前 Plan 契约 + 额外可选信息。"""
    workspace: Any            # pathlib.Path
    plan: Optional[Dict[str, Any]] = None
    stack: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseSensor:
    """传感器基类：所有内置校验与用户插件统一实现该接口（自描述）。"""
    name: str = ""
    description: str = ""
    severity: str = "error"   # error=阻断级；warning=提示；info=信息
    enabled: bool = True

    def run(self, ctx: HarnessContext) -> SensorFact:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}[{self.severity}]: {self.description}"


class SensorRegistry:
    """传感器注册中心：登记、按启用状态枚举、强制跑全量流水线。"""
    _sensors: List[BaseSensor] = []

    @classmethod
    def register(cls, sensor: BaseSensor) -> BaseSensor:
        cls._sensors.append(sensor)
        return sensor

    @classmethod
    def all(cls) -> List[BaseSensor]:
        return [s for s in cls._sensors if s.enabled]

    @classmethod
    def names(cls) -> List[str]:
        return [s.name for s in cls.all()]

    @classmethod
    def run_pipeline(cls, ctx: HarnessContext) -> List[SensorFact]:
        """代码强制触发全量传感器，输出标准化事实集合（原始日志不向上透传）。"""
        facts: List[SensorFact] = []
        for s in cls.all():
            try:
                fact = s.run(ctx)
                if isinstance(fact, SensorFact):
                    facts.append(fact)
            except Exception as e:  # 传感器自身异常不能中断流水线，转为一条 error fact
                facts.append(SensorFact(
                    sensor_name=s.name, ok=False,
                    message=f"sensor internal error: {e}", severity=s.severity,
                ))
        return facts


# 便捷构造
def make_fact(sensor_name: str, ok: bool, message: str,
              payload: Optional[Dict[str, Any]] = None,
              severity: Optional[str] = None) -> SensorFact:
    # 未显式指定时，按事实成败推导阻断级别：通过=info（不阻断），失败=error（阻断级）。
    if severity is None:
        severity = "info" if ok else "error"
    return SensorFact(sensor_name=sensor_name, ok=ok, message=message,
                      payload=payload or {}, severity=severity)


def sensor(name: str, description: str, severity: str = "error",
           enabled: bool = True):
    """装饰器：把 BaseSensor 子类注册进 SensorRegistry（读取类级 name/description）。"""
    def deco(cls):
        inst = cls()
        inst.name = name
        inst.description = description
        inst.severity = severity
        inst.enabled = enabled
        SensorRegistry.register(inst)
        return cls
    return deco
