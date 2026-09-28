"""循环防护（fences）的**单一住所**：具名限制 + 自声明重置锚点 + hook 驱动重置。

设计动机（2026-09-12 重构 #27）：
以前各级别的「兜底限制」散在 4 个互不相认的地方——`LoopConfig.max_iter`（预算）、
`ContextManager.guard`（三个 stall 计数）、`ctx.metadata` 的裸计数器（bar/lint/drift）、
以及模块级全局（`harness._VAL_*` / `GLOBAL_STATE`）。结果是：
  - 想看清「到底有哪些限制」要翻 4 个文件；
  - **重置时机靠人记**（`_l2_start` 手敲一行 `cm.reset_guard()`），漏一处就跨轮串味
    （REPL 的 guard 因此从不重置）；
  - 阈值在 `LoopConfig`、计数在 `cm`，两处分居。

本模块把「限制」提升为一等对象：
  - `Limit(name, reset_at, threshold, on_trip)` —— **reset_at 即重置时机的自声明**（用 `HookPoint`）；
  - `Guard` 是这些限制的注册表 + 计数，自己知道「谁该在什么时候被清零」。

重置由 loop 的 hook 自动执行，不再由调用方手工敲：
  - `Agent._run_loop(loop)` 在**每层循环的每次迭代起始**按 `loop.hook_point` 调
    `guard.reset_at(loop.hook_point)`，清零与该锚点同 `reset_at` 的限制；
  - `Agent.run()` 在**轮次边界**调 `guard.reset_at(HookPoint.RUN_START)` 全量重置。

因此「L1 级的计数不能重置」与「守卫要重置」不再矛盾：把它们声明成不同的 `reset_at`
（RUN_START / L1/L2/L3_LOOP_START）即可，各自的重置时机写在声明处、由同一套 hook 统一执行。

越界「决策」（stuck / unsolvable）不再由本类标签表达，而由 guards.Guard 子类经 hook 总线
返回 GateDecision（REJECT / BREAK_LOOP），由 loop 层统一消费（见 swe_agent/agent.py _run_loop
与 supervisor.py 校验闸门）。本模块只持有计数器与重置时机。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Tuple

from .hooks import HookPoint


@dataclass(frozen=True)
class Limit:
    """一条具名限制。

    - name：计数键名（沿用历史命名，日志与测试语义不变）。
    - reset_at：**重置时机（自声明）**，值为 `HookPoint` 枚举。
      含义：当某层 loop.hook_point == 本 reset_at 时，该层每次迭代起始自动清零本限制；
      其余 loop 不动它。据此把「跨轮累计的 stall 计数」(L2_LOOP_START) 与「跨 run 累计的熔断计数」(RUN_START)
      声明成不同锚点，重置时机各写各处、由同一套 hook 统一执行，不再靠人记。
    - threshold：阈值（0 = 只计数不设闸门）。越界「决策」（stuck / unsolvable）不再由本类
      标签表达，而由 guards.Guard 子类订阅对应 HookPoint 返回 GateDecision（REJECT / BREAK_LOOP）。
    """

    name: str
    reset_at: HookPoint = HookPoint.RUN_START
    threshold: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.reset_at, HookPoint):
            raise TypeError(f"reset_at must be a HookPoint, got {type(self.reset_at).__name__}")


class Guard:
    """具名限制的注册表 + 计数。限制未声明即报错（防拼写漂移悄悄失效）。"""

    def __init__(self, limits: Iterable[Limit] = ()) -> None:
        self._limits: Dict[str, Limit] = {}
        self._counts: Dict[str, int] = {}
        self.declare_all(limits)

    # ---------- 声明 ----------
    def declare(self, limit: Limit) -> "Guard":
        """登记（或覆盖）一条限制；同名再声明即以新阈值/锚点为准。"""
        self._limits[limit.name] = limit
        self._counts.setdefault(limit.name, 0)
        return self

    def declare_all(self, limits: Iterable[Limit]) -> "Guard":
        for lim in limits:
            self.declare(lim)
        return self

    # ---------- 只读 ----------
    @property
    def limits(self) -> Dict[str, Limit]:
        return dict(self._limits)

    def limit(self, name: str) -> Limit:
        return self._limits[name]

    def value(self, name: str) -> int:
        return self._counts.get(name, 0)

    def tripped(self, name: str) -> bool:
        """该限制是否已达到阈值（threshold<=0 视为不设闸门）。"""
        lim = self._limits[name]
        return lim.threshold > 0 and self._counts.get(name, 0) >= lim.threshold

    def snapshot(self) -> Dict[str, int]:
        return dict(self._counts)

    # ---------- 计数 ----------
    def tick(self, name: str, n: int = 1) -> int:
        """计数 +n 并返回新值。未声明的名字直接抛 KeyError（不许裸用）。"""
        if name not in self._counts:
            raise KeyError(f"undeclared limit {name!r}; declare it before ticking")
        self._counts[name] += n
        return self._counts[name]

    def set(self, name: str, value: int) -> "Guard":
        if name not in self._counts:
            raise KeyError(f"undeclared limit {name!r}; declare it before setting")
        self._counts[name] = value
        return self

    # ---------- 重置（时机由 reset_at 自声明） ----------
    def reset_at(self, point: HookPoint) -> Tuple[str, ...]:
        """重置 reset_at == point 的全部限制；返回被重置的名字（便于日志/测试断言）。"""
        hit = tuple(n for n, lim in self._limits.items() if lim.reset_at == point)
        for n in hit:
            self._counts[n] = 0
        return hit

    def reset_all(self) -> Tuple[str, ...]:
        """全量重置（轮次边界用，对应 HookPoint.RUN_START）。"""
        names = tuple(self._counts)
        for n in names:
            self._counts[n] = 0
        return names

    def reset(self, name: str) -> "Guard":
        if name in self._counts:
            self._counts[name] = 0
        return self


# ======================================================================
# 规范限制集（BUILD 层声明；阈值可被 loop 覆盖）
# ======================================================================
def default_limits(repeat_threshold: int, no_tool_threshold: int) -> List[Limit]:
    """全部内置限制。阈值默认取全局常量，`Agent` 再用自身 loop 的阈值覆盖。

    reset_at 锚点约定（executor 三层嵌套 l1→l2→l3 分别挂 L1/L2/L3_LOOP_START）：
      - stall 三件套（consec_repeat / no_tool_streak / empty_streak）：reset_at=L2_LOOP_START
        → 每 round 起始清零；analyzer/tester 单层 loop（hook_point=L1_LOOP_START）无 L2 层
        → 整个 run 内累计，与重构前行为一致。
      - 熔断计数（bar_consec_fail / lint_consec_fail / drift_injections）：reset_at=RUN_START
        → 跨 attempt 累计，绝不逐轮重置；仅在各自 gate 判定成功时显式清零（保持「连续」语义）。
      - tool_call_count：reset_at=L3_LOOP_START → 每步清零（观察每步工具调用密度）。
    """
    from . import config as C

    return [
        Limit("consec_repeat", reset_at=HookPoint.L2_LOOP_START, threshold=repeat_threshold),
        Limit("no_tool_streak", reset_at=HookPoint.L2_LOOP_START, threshold=no_tool_threshold),
        Limit("empty_streak", reset_at=HookPoint.L2_LOOP_START, threshold=0),
        Limit("bar_consec_fail", reset_at=HookPoint.RUN_START, threshold=C.VAL_DOOMED_THRESHOLD),
        Limit("lint_consec_fail", reset_at=HookPoint.RUN_START, threshold=C.VAL_DOOMED_THRESHOLD),
        Limit("drift_injections", reset_at=HookPoint.RUN_START, threshold=C.DRIFT_MAX_INJECTIONS),
        Limit("tool_call_count", reset_at=HookPoint.L3_LOOP_START, threshold=0),
    ]


def default_guard() -> Guard:
    """RunState 的默认 guard：限制齐备（阈值取全局默认），保证独立调用点可用。"""
    from . import config as C

    return Guard(default_limits(C.LOOP_REPEAT_THRESHOLD, C.LOOP_REPEAT_THRESHOLD))


def apply_loop_thresholds(guard: Guard, repeat_threshold: int, no_tool_threshold: int) -> Guard:
    """用 Agent 自身 loop 的阈值覆盖 stall 限制（阈值仍由 LoopConfig 声明，计数只在 guard）。"""
    guard.declare(replace(guard.limit("consec_repeat"), threshold=repeat_threshold))
    guard.declare(replace(guard.limit("no_tool_streak"), threshold=no_tool_threshold))
    return guard
