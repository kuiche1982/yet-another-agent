"""统一 hook 总线（HookHub）+ 生命周期锚点（HookPoint）+ 门控决策（GateAction/GateDecision）。

设计动机（2026-09-12 重构 #27）：
此前「门控 / hook」散落在三处、互不认识：
  - `ToolRegistry._global_hooks` 用魔法字符串 phase（before/after/on_success/on_fail/pre_loop/...）
    驱动派发拦截与循环级副作用；
  - `LoopConfig.scope` / `Guard` 用另一套字符串（attempt/round/step/run）驱动限制重置；
  - function-call 的观察点（before_functioncall/after/error）没有统一表达。

本模块把「生命周期锚点」与「门控决策」提升为一等公民：
  - `HookPoint`：可枚举的生命周期锚点（run 边界 / L1~L3 loop 起始 / tool_call 前后 / error / finally）。
    同时覆盖 REPL 与 unattend 两种执行方式——REPL 每轮用户输入 = 一次 RUN_START；
    unattend 整个 run = 一次 RUN_START，内层 L1/L2/L3 loop 各自 emit 对应锚点。
  - `GateAction`/`GateDecision`：tool_call 前的门控不再只是「观察」，可返回
    allow / reject(+reason) / break_loop(+reason) 三种语义，由派发层统一裁决。
  - `HookHub`：`on(point, fn)` 订阅、`emit(point, **fields)` 触发；取代 `_global_hooks` 魔法字符串，
    旧 hook 函数（签名 `fn(ctx, params, result) -> Optional[str]`）经 `_legacy_adapter` 零改动接入。

所有订阅者返回值的约定（经 `_legacy_adapter` 也兼容旧签名）：
  - `None`          → 放行（ALLOW），但可能有副作用（计数 / 记录）；
  - `str`           → BEFORE 阶段 = 拦截短路（reject，reason=该字符串）；
                      AFTER 阶段  = 覆盖工具结果（override）；
  - `GateDecision`  → 显式门控决策（ALLOW/REJECT/BREAK_LOOP/OVERRIDE）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # 仅用于 register() 的类型标注；运行时 import guards 会循环依赖，故置于 TYPE_CHECKING
    from .guards import Guard

# 订阅者返回值类型：旧式 str / None，或新式 GateDecision。
HookReturn = Union[None, str, "GateDecision"]
# 订阅者签名：接收 HookPayload，返回 HookReturn（None/str/GateDecision）。
HookFn = Callable[["HookPayload"], HookReturn]


class HookPoint(Enum):
    """可枚举的生命周期锚点（取代隐式魔法字符串 phase / scope）。

    覆盖 REPL 与 unattend 两种执行方式：
      - RUN_START：run 边界（REPL 每轮用户输入 / unattend 整个 run 起始）——全量重置点；
      - L1_LOOP_START / L2_LOOP_START / L3_LOOP_START：三层嵌套 loop 各自的迭代起始锚点，
        也是对应层级限制（attempt / round / step）的自动重置点；
      - BEFORE_TOOL_CALL / AFTER_TOOL_CALL / ERROR_TOOL_CALL / FINALLY_TOOL_CALL：
        单次工具调用前后 / 异常 / 收尾的观察与门控点（取代旧 before_functioncall/after/error 三段）。
    """

    RUN_START = "run_start"
    L1_LOOP_START = "l1_loop_start"
    L2_LOOP_START = "l2_loop_start"
    L3_LOOP_START = "l3_loop_start"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    ERROR_TOOL_CALL = "error_tool_call"
    FINALLY_TOOL_CALL = "finally_tool_call"
    # 校验失败（supervisor 单杠/lint 闸门在 tick 连续失败计数后 emit；UnsolvableGuard 订阅，
    # 达阈值置 unsolvable 并返回 BREAK_LOOP，取代原先散落的「内联 unsolvable + return 'break'」）。
    VALIDATION_FAIL = "validation_fail"


class GateAction(Enum):
    """门控动作语义（BEFORE_TOOL_CALL 等门控点的裁决结果）。"""

    ALLOW = "allow"          # 放行
    REJECT = "reject"        # 拦截本次工具调用（reason 说明）
    BREAK_LOOP = "break_loop"  # 拦截并请求提前终止当前 loop
    OVERRIDE = "override"    # AFTER 阶段：用 reason 覆盖工具原始结果


@dataclass
class GateDecision:
    """工具调用门控决策。

    - action：门控动作（见 GateAction）；
    - reason：人类可读原因（reject/break_loop 时回灌模型；override 时作为新结果）。
    """

    action: GateAction = GateAction.ALLOW
    reason: str = ""

    @classmethod
    def allow(cls) -> "GateDecision":
        return cls(GateAction.ALLOW, "")

    @classmethod
    def reject(cls, reason: str = "") -> "GateDecision":
        return cls(GateAction.REJECT, reason)

    @classmethod
    def break_loop(cls, reason: str = "") -> "GateDecision":
        return cls(GateAction.BREAK_LOOP, reason)

    @classmethod
    def override(cls, new_result: str) -> "GateDecision":
        return cls(GateAction.OVERRIDE, new_result)

    @property
    def is_allow(self) -> bool:
        return self.action == GateAction.ALLOW


@dataclass
class HookPayload:
    """emit 时构造、传给每个订阅者的载体。

    fields 约定键（调用方与订阅者约定，非强制）：
      - ctx：ActionContext / RunState（视锚点而定）
      - params：动作参数字典
      - result：工具结果（AFTER 阶段有值）
      - tool：工具名（tool_call 阶段）
      - loop_ctx：循环级 hook 的 RunState
    """

    point: HookPoint
    fields: Dict[str, Any] = field(default_factory=dict)


class HookHub:
    """统一 hook 总线：订阅（on）+ 触发（emit），取代魔法字符串 phase 注册表。

    设计约束：
      - emit 必须「跑完所有订阅者」再裁决——即便某个订阅者要拦截，其它订阅者的
        副作用（如动作计数）也必须先发生（test_registry_dispatch 的「拦截前全局 before 仍执行」）。
      - 订阅者异常 fail-open：单个订阅者炸了不影响其它订阅者与主流程，仅记日志。
    """

    def __init__(self) -> None:
        self._subs: Dict[HookPoint, List[HookFn]] = {p: [] for p in HookPoint}

    def on(self, point: HookPoint, fn: HookFn) -> "HookHub":
        self._subs.setdefault(point, []).append(fn)
        return self

    def clear(self, point: Optional[HookPoint] = None) -> "HookHub":
        if point is None:
            for p in self._subs:
                self._subs[p] = []
        elif point in self._subs:
            self._subs[point] = []
        return self

    def register(self, guard: "Guard") -> "HookHub":
        """注册一个自描述 guard（swe_agent.guards.Guard 子类）。

        按 guard.hook_points 把 guard.check 挂到对应 HookPoint；调用方只丢实例，
        接入点由 guard 自声明（取代 on(point, fn) 散写 hook 点）。guard 仅 duck-typed
        （读 .hook_points / .check），本模块不 import guards，避免循环依赖。
        """
        for p in guard.hook_points:
            self._subs.setdefault(p, []).append(guard.check)
        return self

    def subscribers(self, point: HookPoint) -> Tuple[HookFn, ...]:
        return tuple(self._subs.get(point, ()))

    def emit(self, point: HookPoint, **fields: Any) -> List[HookReturn]:
        """触发某锚点的全部订阅者，返回各自的原始返回值（None/str/GateDecision）。

        调用方（派发层）按锚点语义解释返回值；本方法只负责「跑完」+ fail-open。
        """
        out: List[HookReturn] = []
        for fn in self._subs.get(point, []):
            try:
                out.append(fn(HookPayload(point=point, fields=fields)))
            except Exception as e:  # fail-open：单订阅者异常不影响主流程
                logger.info('[hook:%s] subscriber error ignored: %s', point.value, e)
                out.append(None)
        return out

    # ---- 门控裁决辅助（供派发层使用）----

    @staticmethod
    def first_block(decisions: List[HookReturn]) -> Optional[str]:
        """从 BEFORE_TOOL_CALL 的返回值中取第一个「拦截」决定；无则 None。

        优先级：首个非 ALLOW 胜出（REJECT / BREAK_LOOP / 旧式 str 均视为拦截）。
        返回拦截原因字符串（供回灌模型），或 None 表示全部放行。
        """
        for d in decisions:
            if d is None:
                continue
            if isinstance(d, str):
                return d
            if isinstance(d, GateDecision) and d.action != GateAction.ALLOW:
                return d.reason or "rejected_by_hook"
        return None

    @staticmethod
    def first_block_decision(decisions: List[HookReturn]) -> Optional["GateDecision"]:
        """同 first_block，但返回首个拦截的 GateDecision 本体（含 action 语义）。

        旧式 str 拦截被包装成 GateDecision.reject(str)，便于派发层区分
        REJECT（回灌 reason、loop 继续）与 BREAK_LOOP（回灌 reason、请求终止当前 loop）。
        全部放行时返回 None。
        """
        for d in decisions:
            if d is None:
                continue
            if isinstance(d, str):
                return GateDecision.reject(d)
            if isinstance(d, GateDecision) and d.action != GateAction.ALLOW:
                return d
        return None

    @staticmethod
    def override_of(decisions: List[HookReturn], original: str) -> str:
        """从 AFTER_TOOL_CALL 的返回值中取最后一个覆盖结果；无覆盖则 original。"""
        result = original
        for d in decisions:
            if d is None:
                continue
            if isinstance(d, str):
                result = d
            elif isinstance(d, GateDecision) and d.action == GateAction.OVERRIDE:
                result = d.reason
        return result


def _legacy_adapter(fn: Callable[..., Optional[str]]) -> HookFn:
    """把旧式 hook `fn(ctx, params, result) -> Optional[str]` 适配成 HookHub 订阅者。

    旧 hook 只认位置参数，不认 HookPayload；这里从 payload.fields 取出它要的参数再调用，
    让既有 `_hook_*` 函数零改动接入 HookHub。返回值（str/None）原样透传，由派发层解释。
    """

    def _sub(payload: HookPayload) -> HookReturn:
        return fn(
            payload.fields.get("ctx"),
            payload.fields.get("params"),
            payload.fields.get("result"),
        )

    return _sub


# 模块级单例：全局 hook 总线（取代 ToolRegistry._global_hooks）。
HOOK_HUB = HookHub()
