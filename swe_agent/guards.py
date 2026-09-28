"""自描述行为护栏（guard）的一等抽象 + 内置护栏实现。

设计（对齐 2026-09-13 重构，用户明确要求）：
所有「行为护栏」都是 Guard 子类，自包含三件事：
  - hook_points：本 guard 挂在哪些 HookPoint（自声明接入点）；
  - applies_to：本 guard 管哪些工具（tool_call 类 guard 用；loop / validation 类留 None）；
  - check(payload)：具体判定逻辑，返回 GateDecision（REJECT=回灌 reason+继续；
    BREAK_LOOP=回灌 reason+终止 loop）或 None（放行）。

注册只需 `HOOK_HUB.register(guard_instance)` —— 只丢实例，hook 总线读其 hook_points
决定挂哪；loop 层 emit 对应 hook_point 后统一消费 first_block_decision（Y-a 结果约定）。
这取代原先散落的「工具内硬拒绝」与「内联 return 'stuck' / ctx.metadata['unsolvable']」两套机制，
所有护栏走同一套 hook 总线：新增护栏 = 写一个 Guard 子类 + 注册，loop / dispatch / 校验闸门零改动。

注意：计数仍由 guard.Guard（具名 Limit 注册表）持有；本模块的 Guard 只做「决策」，
阈值从 `ctx.guard.limit(name).threshold` 读取（含 per-role override，如 analyzer 放宽），
不重复存阈值。
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

from .hooks import HookPoint, HOOK_HUB, GateDecision, HookPayload
from . import config as C
from . import models as M


class Guard:
    """行为护栏基类（自描述 + 自注册）。

    子类覆盖类属性 hook_points / applies_to 与方法 check()。注册方式：
        HOOK_HUB.register(WriteSizeGuard())   # 或 guards.register_all_builtin()
    hook 总线按 hook_points 把 check 挂到对应 HookPoint；emit 时统一消费其返回值。
    """

    hook_points: Tuple[HookPoint, ...] = ()
    applies_to: Optional[Set[str]] = None  # None = 不按工具过滤（loop / validation 级）

    def check(self, payload: HookPayload) -> Optional[GateDecision]:
        return None

    def _applies(self, payload: HookPayload) -> bool:
        if not self.applies_to:
            return True
        return payload.fields.get("tool") in self.applies_to


# ---------------------------------------------------------------------------
# 工具调用前护栏（BEFORE_TOOL_CALL）
# ---------------------------------------------------------------------------
class WriteSizeGuard(Guard):
    """write_file 单次写入行数上限（弱 500 / 强 1000 源码；测试放宽到 1000 / 2000）。

    仅管 write_file；超限 REJECT（回灌 reason、loop 继续、工具不执行）。
    """

    hook_points = (HookPoint.BEFORE_TOOL_CALL,)
    applies_to = {"write_file"}

    def check(self, payload: HookPayload) -> Optional[GateDecision]:
        if not self._applies(payload):
            return None
        params = payload.fields.get("params") or {}
        content = params.get("content") or ""
        path = params.get("path") or ""
        nlines = content.count("\n") + 1 if content else 0
        test_match = bool(C._TEST_FILE_RE.search(path))
        if M.is_weak_executor():
            cap = C.WEAK_MAX_TEST_WRITE_LINES if test_match else C.WEAK_MAX_WRITE_LINES
        else:
            cap = C.STRONG_MAX_TEST_WRITE_LINES if test_match else C.STRONG_MAX_WRITE_LINES
        if nlines > cap:
            return GateDecision.reject(
                reason=(f"write_rejected: 文件过大（{nlines} 行，超过单次写入上限 {cap} 行），已拒绝写入。"
                        f"请拆分：先用 write_file 写一个【最小可运行版本】（只覆盖核心功能，建议 ≤ {cap} 行），"
                        f"用 pytest 确认通过后，再用 edit_file 分批追加其余内容。"
                        f"禁止一次性写出上百行/上百个高度重复的测试函数。")
            )
        return None


class ReadSizeGuard(Guard):
    """read_file 单次读取行数上限（MAX_READ_LINES，默认 100）。

    仅管 read_file；limit 给了看 limit，没给则读目标文件实际行数（只读、经 _safe_rel 防逃逸）；
    超限 REJECT（回灌 reason、loop 继续）。这是「加个新 guard 很容易」的范式样板：
    仅一个类 + 注册，loop / dispatch 零改动即获 REJECT 语义。
    """

    hook_points = (HookPoint.BEFORE_TOOL_CALL,)
    applies_to = {"read_file"}

    def check(self, payload: HookPayload) -> Optional[GateDecision]:
        if not self._applies(payload):
            return None
        params = payload.fields.get("params") or {}
        limit = params.get("limit")
        cap = C.MAX_READ_LINES
        if limit is not None:
            try:
                nlines = int(limit)
            except (TypeError, ValueError):
                nlines = 0
        else:
            # 未给 limit = 读全文件：需自己算实际行数（只读、路径走工具安全解析，防 ../ 逃逸）
            from .tools import WORKSPACE, _safe_rel
            try:
                p = WORKSPACE / _safe_rel(params.get("path") or "")
                nlines = (p.read_text(errors="replace").count("\n") + 1) if p.exists() else 0
            except Exception:
                nlines = 0
        if nlines > cap:
            return GateDecision.reject(
                reason=(f"read_rejected: 单次读取不能超过 {cap} 行，请用 limit 参数分批读取"
                        f"（如 limit={cap}）。")
            )
        return None


# ---------------------------------------------------------------------------
# 循环级护栏（L1 / L2 / L3_LOOP_START，loop 在每步结束后 emit 消费）
# ---------------------------------------------------------------------------
class StallGuard(Guard):
    """连续重复动作 / 连续未调工具 → 停滞（REJECT=回灌 reason、loop 继续）。

    阈值从 ctx.guard.limit(name).threshold 读取（含 per-role override，如 analyzer 放宽），
    本 guard 不存阈值。订阅 L1 / L2 / L3 各 loop hook 点；loop 在每步结束后 emit，
    超阈即 REJECT（回灌建设性指引，提示换策略）。取代原先散落在 agent._apply_toolcall
    的内联 return 'stuck'。
    """

    hook_points = (HookPoint.L1_LOOP_START, HookPoint.L2_LOOP_START, HookPoint.L3_LOOP_START)
    applies_to = None

    def check(self, payload: HookPayload) -> Optional[GateDecision]:
        ctx = payload.fields.get("ctx")
        if ctx is None:
            return None
        g = ctx.guard
        rep_lim = g.limit("consec_repeat").threshold
        no_tool_lim = g.limit("no_tool_streak").threshold
        rep = g.value("consec_repeat")
        no_tool = g.value("no_tool_streak")
        if rep_lim > 0 and rep >= rep_lim:
            return GateDecision.reject(
                reason=("（系统）检测到本步连续重复相同动作、未产生进展，已触发停滞保护并暂停本步。"
                        "请先 read_file 查看文件现状，再换一种实质性策略推进"
                        "（例如改用不同文件或命令），不要重复刚才的动作。")
            )
        if no_tool_lim > 0 and no_tool >= no_tool_lim:
            return GateDecision.reject(
                reason=("（系统）连续多轮未产出可执行工具调用（模型只回空响应 / 散文），"
                        "已触发停滞保护并暂停本步。请通过调用工具（write_file / edit_file / shell / complete 等）"
                        "推进任务，不要空响应。")
            )
        return None


# ---------------------------------------------------------------------------
# 校验级护栏（VALIDATION_FAIL，supervisor 闸门在 tick 后 emit）
# ---------------------------------------------------------------------------
class UnsolvableGuard(Guard):
    """单杠（pytest）/ lint 连续失败达 VAL_DOOMED_THRESHOLD → 任务不可解（BREAK_LOOP）。

    supervisor._l1_gate / _l2_gate 在 tick bar / lint 计数后 emit VALIDATION_FAIL；
    本 guard 读计数达阈值即置 ctx.metadata['unsolvable']=True、回灌原因、返回 BREAK_LOOP
    （loop 逐层终止）。取代原先散落在 supervisor 校验闸门的「内联 unsolvable + return 'break'」。
    """

    hook_points = (HookPoint.VALIDATION_FAIL,)
    applies_to = None

    def check(self, payload: HookPayload) -> Optional[GateDecision]:
        ctx = payload.fields.get("ctx")
        if ctx is None:
            return None
        g = ctx.guard
        bar = g.value("bar_consec_fail")
        lint = g.value("lint_consec_fail")
        detail = payload.fields.get("detail", "")
        if bar >= C.VAL_DOOMED_THRESHOLD:
            ctx.metadata["unsolvable"] = True
            ctx.cm.append("user",
                f"⚠️ 连续 {bar} 次单杠校验未通过（当前为：{detail}），"
                "疑似任务对当前模型不可解（测试期望或实现始终无法通过）。已停止重试以避免空耗预算。")
            return GateDecision.break_loop(reason="unsolvable: bar consec fail")
        if lint >= C.VAL_DOOMED_THRESHOLD:
            ctx.metadata["unsolvable"] = True
            ctx.cm.append("user",
                f"⚠️ 连续 {lint} 次静态校验（lint）失败，"
                "疑似任务对当前模型不可解（源码始终无法编译通过）。已停止重试以避免空耗预算。")
            return GateDecision.break_loop(reason="unsolvable: lint consec fail")
        return None


# ---------------------------------------------------------------------------
# 内置护栏集中注册（生产 import 即注册；测试 conftest autouse 重注册）
# ---------------------------------------------------------------------------
_BUILTIN_GUARDS: Tuple[Guard, ...] = (
    WriteSizeGuard(),
    ReadSizeGuard(),
    StallGuard(),
    UnsolvableGuard(),
)
# 内置 guard 订阅的 hook 点（定向 clear，避免误伤 register_global_hook 等其它订阅者）
_BUILTIN_POINTS = (
    HookPoint.BEFORE_TOOL_CALL,
    HookPoint.L1_LOOP_START,
    HookPoint.L2_LOOP_START,
    HookPoint.L3_LOOP_START,
    HookPoint.VALIDATION_FAIL,
)


def register_all_builtin() -> None:
    """（重新）注册全部内置护栏。先定向清空其订阅点，避免重复注册累积。"""
    for p in _BUILTIN_POINTS:
        HOOK_HUB.clear(p)
    for g in _BUILTIN_GUARDS:
        HOOK_HUB.register(g)


# 模块 import 即注册（生产路径：supervisor / agent 顶层 import swe_agent.guards 触发）
register_all_builtin()
