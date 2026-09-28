#!/usr/bin/env python3
"""离线契约测试 —— 模型与工具「双 fake」，不连 LLM / 网络 / 文件系统。

把 harness 的两条核心契约编成 model-free + tool-free 回归测试，专门防止上一轮那种
「靠跑完整 e2e 才撞出来、且日志没有任何根因」的流程 bug：

  Bug A 类（装饰器错挂）：write_file 的 @tool 被挂到内部 _write_verify，
    导致 executor 写不出文件。回归：检查注册表 —— write_file 已注册且其实现函数
    就是 write_file 本体（而非内部 helper）。纯注册表检查，无需真实写盘。
  Bug B 类（循环不终止）：tester 命中 finish_verify 后，single_loop 无 on_iter_end
    把 all_done 当 continue 重跑，弱模型空转到 max_iter。回归：fake 模型按「验证点脚本」
    只发一次 finish_verify，fake 工具只记录调用；run_tester 必须在 1 次模型调用内
    干净终止并捕获 results。

双 fake 设计（本文件核心）：
  - ToolFreeRig：monkeypatch ToolRegistry.get，对【任意工具名】返回 fake ToolDef，
    其 run 只记录 (name, kwargs) 并返回脚本回执。完整保留 dispatch / hook / 角色
    allowed 校验流水线，但工具体绝不执行真实副作用（不写盘、不联网、不跑 shell）。
  - ScriptedModel：monkeypatch chat_toolcalls，按测试用例给定的「验证点脚本」
    （step = (工具名, 参数)）依次返回 toolcall，引导循环按测试关注的流程点走完；
    脚本耗尽后返回空（让循环按各自逻辑终止/卡住）。模型本身也是假的。

这样测试只验证 harness 的控制流契约，确定性、零副作用、秒级跑完。真实工具的行为
冒烟（write_file 真落盘等）交给离线脚本 scripts/harness_selfcheck.py，二者分层。

运行：uv run pytest tests/test_harness_contracts.py -q
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from swe_agent.registry import ToolRegistry, ToolDef
import swe_agent.models as M
import swe_agent.verify as V
import swe_agent.llm_lmstudio as L


# ======================================================================
# 双 fake 装配
# ======================================================================
class ToolFreeRig:
    """工具全 fake：monkeypatch ToolRegistry.get，对【任意工具名】返回 fake ToolDef。

    - 完整保留 dispatch / before-after 钩子 / 角色 allowed 校验流水线；
    - fake run 只记录 (name, kwargs) 并返回脚本回执或通用成功串；
    - 绝不执行真实工具副作用（不写盘、不联网、不跑 shell）。
    """

    def __init__(self, monkeypatch):
        self.calls: list = []                       # [(name, kwargs), ...] 按调用顺序
        self.scripts: dict = {}                     # name -> 静态回执字符串 或 callable(kwargs)->str
        self._orig_get = ToolRegistry.get
        monkeypatch.setattr(ToolRegistry, "get", self._fake_get)

    def _fake_get(self, name):
        # 仍引用真实注册表里的声明（schema/描述），仅把 run 换成 fake
        real = ToolRegistry._tools.get(name)
        schema = real.schema if real is not None else {"type": "object", "properties": {}}
        return ToolDef(
            name=name,
            description=f"[fake] {name}",
            category="fs",
            schema=schema,
            run=self._make_run(name),
        )

    def _make_run(self, name):
        def _run(ctx=None, **kwargs):
            self.calls.append((name, kwargs))
            script = self.scripts.get(name)
            if callable(script):
                return script(kwargs)
            if isinstance(script, str):
                return script
            return f"[fake:{name}] ok (args={sorted(kwargs)})"
        return _run

    def set_script(self, name, script) -> None:
        self.scripts[name] = script

    def called(self, name) -> list:
        return [kw for (n, kw) in self.calls if n == name]

    def count(self, name) -> int:
        return len(self.called(name))

    def sequence(self) -> list:
        """调用到的工具名序列（按时间）。"""
        return [n for (n, _) in self.calls]


class ScriptedModel:
    """模型 fake：按「验证点脚本」返回 toolcall，引导循环按测试关注的流程点走完。

    steps：有序列表，每项是一「轮」脚本，两种形态（向后兼容）：
      - 单调用：(工具名, 参数dict)
      - 多调用：[(工具名, 参数dict), (工具名, 参数dict), ...]
        模拟支持 parallel tool calls 的模型在一轮内下发多个 tool_call。
    依次 pop 返回；耗尽后：
      - fallback 非空 → 一直返回该轮（模拟弱模型反复重发同一动作）；
      - fallback 为空 → 返回空 tool_calls（让循环按各自逻辑终止/卡住）。
    """

    def __init__(self, steps, fallback=None):
        self.steps = list(steps)
        self.fallback = fallback                      # 一「轮」脚本：单调用或多调用
        self.model_calls = 0

    @staticmethod
    def _as_calls(item):
        """把一「轮」脚本归一化成 [(name, args), ...]（兼容单调用 tuple 形态）。"""
        return list(item) if isinstance(item, list) else [item]

    def __call__(self, role="tester", messages=None, temperature=0.3, max_tokens=None,
                 tools=None, model_override=None, tool_choice=None):
        self.model_calls += 1
        if self.steps:
            item = self.steps.pop(0)
        elif self.fallback:
            item = self.fallback
        else:
            return {"type": "toolcalls", "actions": [], "tool_calls": []}
        calls = self._as_calls(item)
        return {
            "type": "toolcalls",
            "actions": [{"action": n} for (n, _a) in calls],
            # 每个 tool_call 的 id 必须唯一（多调用场景用序号后缀），否则回执无法一一对应
            "tool_calls": [{
                "id": f"c{self.model_calls}_{i}",
                "name": n,
                "arguments": json.dumps(a),
            } for i, (n, a) in enumerate(calls)],
        }


# ======================================================================
# Bug A：工具注册完整性（tool-free，无文件系统 I/O、无模型）
# 只需检查注册表即可定位「装饰器错挂」根因，无需真实写盘。
# ======================================================================
def test_write_file_registered_and_not_mis_hung():
    import swe_agent.tools as T  # 触发全部 @tool 注册

    wf = ToolRegistry.get("write_file")
    assert wf is not None, "write_file 未注册（@tool 被错挂到别的内部函数）"
    # 核心断言：实现函数必须就是 write_file 本体，而非内部 _write_verify
    assert wf.run.__name__ == "write_file", (
        f"write_file 实现错挂到 {wf.run.__name__!r}（装饰器 @tool(name='write_file') "
        f"被挂到了内部函数，executor 将写不出文件）")
    # 反向保证：任何以 '_' 开头的内部函数都不应成为对外工具名
    assert "_write_verify" not in ToolRegistry.names()
    for name in ToolRegistry.names():
        assert not name.startswith("_"), f"内部函数 {name!r} 误注册为对外工具"


def test_no_internal_helper_registered_as_public_tool():
    import swe_agent.tools as T
    for name in ToolRegistry.names():
        assert not name.startswith("_"), f"内部函数 {name!r} 被误注册为对外工具"


# ======================================================================
# Bug B：tester 循环必须在 finish_verify 命中后终止（model-free + tool-free）
# ======================================================================
def test_tester_terminates_on_first_finish_verify(monkeypatch):
    # 工具全 fake：finish_verify 只记录调用、不真正执行任何副作用
    rig = ToolFreeRig(monkeypatch)
    # 模型按验证点脚本：第一次就提交 finish_verify（带 1 条 pass 验收点）。
    # fallback 令「脚本耗尽后被继续追问」时仍重发 finish_verify——忠实复现弱模型在缺
    # on_iter_end 时会反复重发终止动作的行为（而非返回空 tool_calls 触发别的崩溃）。
    model = ScriptedModel(
        [("finish_verify", {"results": [{"id": 1, "verdict": "pass", "evidence": "read src, ok"}]})],
        fallback=("finish_verify", {"results": [{"id": 1, "verdict": "pass", "evidence": "read src, ok"}]}),
    )
    monkeypatch.setattr(M, "chat_toolcalls", model)
    # 注入 1 条验收点，绕开 _load_points 对 GLOBAL_STATE/磁盘的依赖
    monkeypatch.setattr(
        V, "_load_points",
        lambda: ("goal", [{"id": 1, "point": "p", "check_hint": ""}]))

    verdict, detail, results = V.run_tester("fake-model", max_iter=5)
    # 关键断言：模型只被调用 1 次（若 on_iter_end 缺失会重跑到 max_iter=5）
    assert model.model_calls == 1, (
        f"tester 在 finish_verify 后未终止，模型被调用 {model.model_calls} 次（预期 1）。"
        " 根因：single_loop 未传 on_iter_end，_run_loop 把 all_done 当 continue 重跑。")
    assert rig.count("finish_verify") == 1, "finish_verify 被调用次数异常"
    assert verdict == "pass", f"tester 未干净通过：{verdict} / {detail[:120]}"
    assert results and results[0]["verdict"] == "pass"


def test_tester_walks_verification_points_in_order(monkeypatch):
    """演示「按验证点返工具调用信息，引导循环按流程点走完」：

    脚本令模型先 read_file（探查实现），再 finish_verify（提交判定）。
    fake 工具记录调用顺序，断言循环确实按 read_file → finish_verify 走完并在 finish_verify
    后干净终止（模型仅在 2 个验证点被调用）。
    """
    rig = ToolFreeRig(monkeypatch)
    model = ScriptedModel(
        [("read_file", {"path": "src/add.py"}),
         ("finish_verify", {"results": [
             {"id": 1, "verdict": "pass", "evidence": "read src/add.py 确认签名正确"}]})],
        fallback=("finish_verify", {"results": [
            {"id": 1, "verdict": "pass", "evidence": "read src/add.py 确认签名正确"}]}),
    )
    monkeypatch.setattr(M, "chat_toolcalls", model)
    monkeypatch.setattr(
        V, "_load_points",
        lambda: ("goal", [{"id": 1, "point": "p", "check_hint": ""}]))

    verdict, detail, results = V.run_tester("fake-model", max_iter=5)
    # 模型按脚本发了 2 次调用，循环在 finish_verify 后干净终止（不再多调）
    assert model.model_calls == 2, f"模型被调用 {model.model_calls} 次（预期 2）"
    assert rig.sequence() == ["read_file", "finish_verify"], (
        f"流程顺序错误：{rig.sequence()}（预期 read_file → finish_verify）")
    assert verdict == "pass", f"tester 未干净通过：{verdict} / {detail[:120]}"


def test_tester_on_iter_end_guard_contract():
    """闸门自身契约（无需模型/工具）：命中 all_done 必须返回 done，否则 _run_loop 会当
    continue 重跑。"""
    assert V._tester_on_iter_end(None, "all_done") == "done"
    assert V._tester_on_iter_end(None, "continue") is None


# ======================================================================
# Bug B / 2026-09-02 修复：tester 未提交 finish_verify → skipped（model-free + tool-free）
# 根因 = executor/tester 共用同一 _apply_toolcall，对「无 content + 无 toolcall +
# finish_reason=stop」统一当 continue（executor 该 continue，tester 该终止）。共享代码把
# tester 也当 continue → 弱 tester 被反复 nudge 空转、又被 supervisor 当 fail 重跑 attempt，
# 即 case 失败下 tester 不停重试的空转。修复：run_tester 只要没拿到 finish_verify 就判
# skipped（没有 failed_case → 不重跑）。以下用双 fake 锁死这条不变量。
# ======================================================================
def test_tester_skips_when_no_finish_verify_committed(monkeypatch):
    """tester 全程未提交 finish_verify（模型只返回空 tool_calls / finish_reason=stop）→
    run_tester 必须返回 skipped（而非 fail），_l1_gate 据此不重跑 attempt。"""
    rig = ToolFreeRig(monkeypatch)
    # 模型始终返回空 tool_calls（= finish_reason=stop，未调任何工具），模拟弱 tester 不交卷
    model = ScriptedModel([], fallback=None)
    monkeypatch.setattr(M, "chat_toolcalls", model)
    monkeypatch.setattr(
        V, "_load_points",
        lambda: ("goal", [{"id": 1, "point": "p", "check_hint": ""}]))
    verdict, detail, results = V.run_tester("fake-model", max_iter=5)
    assert verdict == "skipped", (
        f"未提交 finish_verify 应判 skipped（不重跑），实际 {verdict}: {detail[:120]}")
    assert results == [], "skipped 不应带任何 results"


def test_tester_maxiter_cap_degrades_to_skipped_not_fail(monkeypatch):
    """tester 撑满 max_iter 仍未提交 finish_verify → verdict 必须是 skipped 而非 fail。

    语义：触顶是【tester 自身收敛失败】，不是代码缺陷。旧代码返回 fail 会让 supervisor
    把「tester 没交卷」误判成「代码有缺陷」而重跑整个 attempt——惩罚正确代码。
    max_iter=2 < LOOP_REPEAT_THRESHOLD=3，确保走的是 limit 触顶分支（而非 stuck 分支）。
    """
    rig = ToolFreeRig(monkeypatch)
    # 模型每轮只发 read_file（非终止动作），永不提交 finish_verify → 循环走到 max_iter 上限
    step = ("read_file", {"path": "src/x.py"})
    model = ScriptedModel([step], fallback=step)
    monkeypatch.setattr(M, "chat_toolcalls", model)
    monkeypatch.setattr(
        V, "_load_points",
        lambda: ("goal", [{"id": 1, "point": "p", "check_hint": ""}]))
    verdict, detail, results = V.run_tester("fake-model", max_iter=2)
    assert verdict == "skipped", (
        f"max_iter 触顶未交卷应判 skipped（不重跑），实际 {verdict}: {detail[:120]}")
    assert "max_iter" in detail, f"detail 应说明是 max_iter 触顶：{detail[:120]}"
    assert model.model_calls <= 2, f"触顶后不应继续空转：{model.model_calls} 次模型调用"
    assert rig.count("finish_verify") == 0, "tester 从未提交 finish_verify"
    assert results == [], "skipped 不应带任何 results"


def test_tester_default_max_iter_comes_from_build_constant():
    """BUILD 层常量收口：run_tester / verify_gate 默认 max_iter 必须取 C.MAX_TESTER_ITER，
    不得在调用处写死魔法数字（否则调参时改一处漏一处）。"""
    import inspect
    import swe_agent.config as C
    assert hasattr(C, "MAX_TESTER_ITER"), "config 缺 MAX_TESTER_ITER（BUILD 常量）"
    for fn in (V.run_tester, V.verify_gate):
        sig = inspect.signature(fn)
        default = sig.parameters["max_iter"].default
        assert default == C.MAX_TESTER_ITER, (
            f"{fn.__name__} 默认 max_iter={default}，与 C.MAX_TESTER_ITER="
            f"{C.MAX_TESTER_ITER} 不一致")


def test_tester_failed_case_triggers_rerun_verdict(monkeypatch):
    """tester 提交 finish_verify 含 fail 用例 → verdict=fail（failed_case>0 → _l1_gate 重跑）。"""
    rig = ToolFreeRig(monkeypatch)
    payload = {"results": [
        {"id": 1, "verdict": "fail", "evidence": "fib(0) 应为 []"},
        {"id": 2, "verdict": "pass", "evidence": "fib(1)==[1]"}]}
    model = ScriptedModel([("finish_verify", payload)], fallback=("finish_verify", payload))
    monkeypatch.setattr(M, "chat_toolcalls", model)
    monkeypatch.setattr(
        V, "_load_points",
        lambda: ("goal", [{"id": 1, "point": "p1"}, {"id": 2, "point": "p2"}]))
    verdict, detail, results = V.run_tester("fake-model", max_iter=5)
    assert verdict == "fail", f"含 fail 用例应判 fail，实际 {verdict}: {detail[:120]}"
    failed = [r for r in results if r["verdict"] == "fail"]
    assert len(failed) == 1, f"failed_case 计数错误：{results}"


def test_tester_skipped_verdict_not_counted_as_fail(monkeypatch):
    """tester 显式标 skipped 的用例不计入失败 → verdict=pass（不重跑）。"""
    rig = ToolFreeRig(monkeypatch)
    payload = {"results": [
        {"id": 1, "verdict": "pass", "evidence": "fib(1)==[1]"},
        {"id": 2, "verdict": "skipped", "evidence": "无法确认边界"}]}
    model = ScriptedModel([("finish_verify", payload)], fallback=("finish_verify", payload))
    monkeypatch.setattr(M, "chat_toolcalls", model)
    monkeypatch.setattr(
        V, "_load_points",
        lambda: ("goal", [{"id": 1, "point": "p1"}, {"id": 2, "point": "p2"}]))
    verdict, detail, results = V.run_tester("fake-model", max_iter=5)
    assert verdict == "pass", (
        f"含 skipped 用例（无 fail）应判 pass（不重跑），实际 {verdict}: {detail[:120]}")
    assert any(r["verdict"] == "skipped" for r in results), "skipped 用例未被保留"


def test_tester_missing_point_completes_as_skipped_not_fail(monkeypatch):
    """tester 漏提交某验收点 → 补判 skipped（不计失败）→ 无 fail 时 verdict=pass。"""
    rig = ToolFreeRig(monkeypatch)
    # 只提交 id=1，漏掉 id=2
    payload = {"results": [{"id": 1, "verdict": "pass", "evidence": "ok"}]}
    model = ScriptedModel([("finish_verify", payload)], fallback=("finish_verify", payload))
    monkeypatch.setattr(M, "chat_toolcalls", model)
    monkeypatch.setattr(
        V, "_load_points",
        lambda: ("goal", [{"id": 1, "point": "p1"}, {"id": 2, "point": "p2"}]))
    verdict, detail, results = V.run_tester("fake-model", max_iter=5)
    assert verdict == "pass", (
        f"漏提交点应补 skipped（不计入失败），无 fail 应判 pass，实际 {verdict}: {detail[:120]}")
    ids = {r["id"] for r in results}
    assert ids == {1, 2}, f"漏提交点未被补判：{results}"
    assert results[1]["verdict"] == "skipped", "漏提交点未补判 skipped"


def test_tester_skips_when_read_tools_unregistered(monkeypatch):
    """工具闸门（2026-09-02 复盘）：tester 只读工具（read_file/glob/grep/shell）未注册齐全时，
    run_tester 必须在进入模型循环【之前】直接返回 skipped——绝不调一次 LLM、绝不空转、
    绝不把「工具缺失」误判成「代码缺陷」而触发 supervisor loop1 重跑。

    复现真实现状：ToolRegistry 仅注册 finish_verify（只读工具缺失），tester 若真跑会盲判
    （凭任务描述 fabricate 证据、把错代码也判 pass）。guard 把它降级为 skipped。
    """
    # 模拟「只读工具未注册」：names() 只返回 finish_verify
    monkeypatch.setattr(ToolRegistry, "names", classmethod(lambda cls: ["finish_verify"]))
    # 即便强行塞一个会调 finish_verify 的模型，也绝不该被调用（guard 短路）
    model = ScriptedModel(
        [("finish_verify", {"results": [{"id": 1, "verdict": "pass", "evidence": "x"}]})],
        fallback=("finish_verify", {"results": [{"id": 1, "verdict": "pass", "evidence": "x"}]}))
    monkeypatch.setattr(M, "chat_toolcalls", model)
    monkeypatch.setattr(
        V, "_load_points",
        lambda: ("goal", [{"id": 1, "point": "p", "check_hint": ""}]))
    verdict, detail, results = V.run_tester("fake-model", max_iter=5)
    assert verdict == "skipped", (
        f"只读工具缺失应直接 skipped（不进模型循环），实际 {verdict}: {detail[:120]}")
    assert model.model_calls == 0, (
        f"工具缺失时仍调用了模型 {model.model_calls} 次——guard 未短路，会空转/盲判")
    assert results == [], "skipped 不应带 results"


# ======================================================================
# 通用 role 契约：每个带 stop_actions 的角色，其 RoleConfig 必须正确产出 stop_actions
# （纯 config 层检查，无模型/无工具）
# ======================================================================
def test_role_configs_carry_stop_actions():
    import swe_agent.roles_config as RC

    for role in ("analyzer", "tester", "executor"):
        rc = RC.make_role_config(role)
        assert rc.stop_actions, f"[{role}] RoleConfig 缺少 stop_actions（终止动作未声明）"


def test_empty_tool_calls_does_not_crash(monkeypatch):
    """回归（tool-free）：模型返回 type=toolcalls 但 tool_calls 为空时，harness 不得
    IndexError 崩溃（agent.py 旧代码在 tcs[0] 上越界），而应走「未调用工具」兜底
    （nudge + no_tool_streak），由循环防护在阈值后判定 stuck。"""
    import swe_agent.roles_config as RC
    from swe_agent.agent import Agent, RunState

    rig = ToolFreeRig(monkeypatch)
    # 模型始终返回空 tool_calls（退化响应）；fallback=None -> 返回空 tool_calls 体
    model = ScriptedModel([], fallback=None)
    monkeypatch.setattr(M, "chat_toolcalls", model)

    rc = RC.make_role_config("executor")
    ctx = RunState(role="executor")
    agent = Agent(rc, RC.single_loop(max_iter=2), ctx=ctx)

    # 关键：不得抛 IndexError；循环防护会把它当「未调用工具」处理
    reason = agent.run([{"role": "user", "content": "task"}])
    assert reason in ("continue", "stuck", "limit", "done"), f"意外终止原因：{reason}"


def test_respect_stop_policy_drives_nudge_per_role(monkeypatch):
    """回归（stop 约定配置化）：finish_reason=stop 且无工具调用时，是否灌「请调用工具」nudge
    由 RoleConfig.respect_stop 决定，不在 _apply_toolcall 里写死角色名（OOP 反模式）。

    - executor（respect_stop=True）：尊重 stop，不灌 nudge，交外层 _l1_gate 单杠判定
      （lint+pytest+tester 过则收尾、不过回灌失败继续修）——直接修掉 jsonl 里观察到的
      「executor 说'全部完成'仍被逼'调工具'拖着走」。
    - tester（respect_stop=False）：不遵守 stop，必须逼出 stop tool，且 nudge 直呼其名
      （finish_verify），避免 jsonl 实测的「被逼'调工具'后错调 shell 重跑」拖拽。
    """
    import swe_agent.roles_config as RC
    from swe_agent.agent import Agent, RunState
    from swe_agent.management import ContextManager

    # 模拟「模型以 finish_reason=stop 收尾、未调工具、带了实质完成摘要」
    stop_meta = {"type": "content",
                 "content": "全部完成，pytest 7 个用例全部通过（exit code 0，无失败）。",
                 "finish_reason": "stop"}

    # ---- executor：尊重 stop，不灌 nudge ----
    rc_exec = RC.make_role_config("executor")
    assert rc_exec.respect_stop is True, "executor 必须 respect_stop=True"
    ctx_exec = RunState(role="executor")
    ctx_exec.cm = ContextManager()
    agent_exec = Agent(rc_exec, RC.single_loop(max_iter=1), ctx=ctx_exec)
    reason_exec = agent_exec._apply_toolcall(stop_meta)
    assert reason_exec == "continue", f"executor 尊重 stop 应返回 continue，实际 {reason_exec}"
    nudges_exec = [m for m in ctx_exec.cm.to_list()
                   if m.get("role") == "user" and "请调用" in (m.get("content") or "")]
    assert not nudges_exec, f"executor 尊重 stop 不应灌'调工具'nudge，实际：{nudges_exec}"

    # ---- tester：不遵守 stop，逼 finish_verify 且指名 ----
    rc_test = RC.make_role_config("tester")
    assert rc_test.respect_stop is False, "tester 必须 respect_stop=False"
    ctx_test = RunState(role="tester")
    ctx_test.cm = ContextManager()
    agent_test = Agent(rc_test, RC.single_loop(max_iter=1), ctx=ctx_test)
    reason_test = agent_test._apply_toolcall(stop_meta)
    assert reason_test == "continue", (
        f"tester 不遵守 stop 应返回 continue（等阈值判 stuck），实际 {reason_test}")
    nudges_test = [m for m in ctx_test.cm.to_list()
                   if m.get("role") == "user" and "请调用" in (m.get("content") or "")]
    assert nudges_test, "tester 不遵守 stop 应灌'调工具'nudge"
    assert "finish_verify" in nudges_test[-1]["content"], (
        f"tester nudge 应直呼 finish_verify，实际：{nudges_test[-1]['content']}")


def test_executor_stall_detected_through_rejected_edits(monkeypatch):
    """P0 回归（2026-09-02）：executor 陷入 read→被拒edit→read 循环时，空转护栏
    consec_repeat 必须仍累积到 LOOP_REPEAT_THRESHOLD 并判定 stuck——不能因「被拒 edit」
    是不同指纹而反复把 consec_repeat 重置回 1（旧行为：stuck 永不触发，executor 空转到
    1800s 被 SIGKILL；fizzbuzz e2e 实测即此死法）。

    锁死不变量：被 harness 拒绝/无实质改动的工具调用（edit_error 空转编辑）视为「未前进」，
    连续计数 +1 且不更新 last_fp，使后续真实动作能继续累积连续计数。
    """
    import swe_agent.roles_config as RC
    from swe_agent.agent import Agent, RunState

    rig = ToolFreeRig(monkeypatch)
    # 钉死 executor 工具白名单，避免测试 rig 替换 ToolRegistry.get 后 allowed 集被清空
    # （否则所有工具走「不允许」早退路径，consec_repeat 永不累加 → stuck 永不触发，测试失真）
    monkeypatch.setattr(
        "swe_agent.agent.ROLE_TOOLS_ALLOWED",
        {"executor": {"read_file", "edit_file"}},
    )
    # edit_file 被 harness 拒绝（old==new 空转编辑）→ 返回 edit_error 前缀串
    rig.set_script(
        "edit_file",
        lambda kw: "edit_error: old_string 与 new_string 完全相同，这是无效的空转编辑。"
                   "请让 new_string 与 old_string 有实质差异。")
    # 模型序列：严格交替 read → 被拒edit → read → 被拒edit …（共 10 步，恰好覆盖 max_iter=10），
    # 无 fallback：用尽后若未终止走空响应兜底（不干扰本测试关注点）。
    # 这正是 fizzbuzz e2e 里 executor 卡死的形态：read 测试 → 提交 old==new 空转 edit 被拒 → 再 read。
    # 关键：被拒 edit 必须夹在 read 之间，才能检验它是否会把 consec_repeat 计数重置。
    # 新逻辑（P0）：被拒 edit 视为「未前进」→ read 累积 3 次即 stuck（iter3 触发）；
    # 旧逻辑：read/edit 指纹不同 → 每次交替都重置计数 → 跑到 max_iter=10 → limit（不 stuck）。
    step_read = ("read_file", {"path": "test_fizzbuzz.py"})
    step_edit = ("edit_file", {"path": "test_fizzbuzz.py",
                               "old_string": "x", "new_string": "x"})
    model = ScriptedModel([step_read, step_edit] * 5, fallback=None)
    monkeypatch.setattr(M, "chat_toolcalls", model)

    rc = RC.make_role_config("executor")
    ctx = RunState(role="executor")
    # 叶循环闸门：步内 _apply_toolcall 返回 stuck 必须让本层循环立即终止并向上传播
    # 终点原因（真实 harness 里由 supervisor 的 L1/L2 on_iter_end 消费；单元测试直接收口，
    # 否则 single_loop 无 on_iter_end 时会把 stuck 当 last 吞掉、一路跑到 max_iter，
    # 导致「新逻辑 iter3 即 stuck」与「旧逻辑跑满 limit」在 agent.run() 层无法区分）。
    def _exec_stuck_gate(ctx, reason):
        return "break" if reason == "stuck" else None
    agent = Agent(rc, RC.single_loop(max_iter=10, on_iter_end=_exec_stuck_gate), ctx=ctx)
    reason = agent.run([{"role": "user", "content": "task"}])
    # 第 3 次 read 时 consec_repeat 触顶 → stuck；旧代码会一路跑到 max_iter=10 → limit
    assert reason == "stuck", (
        f"read→被拒edit→read 应判定 stuck（空转护栏生效），实际 {reason}。"
        f"若返回 limit 说明被拒 edit 仍在打断 consec_repeat 计数（P0 未生效）。")
    # 必须早于 max_iter=10 触顶（iter3 即 stuck）；若 ==10 说明跑满了 limit 分支（旧逻辑）
    assert 3 <= model.model_calls < 10, (
        f"stuck 应在第 3 次 read 左右触发（model_calls≈3），实际 {model.model_calls}")


def test_l1_gate_caps_consecutive_pytest_failures_as_unsolvable(monkeypatch):
    """P1 回归（2026-09-02）：loop_1 单杠（pytest）连续失败达 VAL_DOOMED_THRESHOLD 次，
    supervisor._l1_gate 必须判定 unsolvable 并 break 终止本任务——而非 continue 一路空耗
    attempt 预算（fizzbuzz e2e 实测：弱模型反复写错测试期望 → 每轮 pytest 失败 → 旧代码
    一路 continue 到 MAX_ATTEMPTS 耗尽 / 烧满 1800s 被 SIGKILL）。

    锁死不变量：
      - 第 1 次 pytest 失败 → continue（给一次重跑机会，对齐 MAX_ATTEMPTS 语义）；
      - 第 2 次（达阈值）→ break + ctx.metadata["unsolvable"]=True；
      - pytest 通过 → consec_bar_fail 清零（保持「连续」语义，历史失败不误触发）。
    测试只用 monkeypatch 钉死 _run_test_bar / verify_gate，不连 LLM / 网络 / 文件系统。
    """
    import swe_agent.supervisor as S
    import swe_agent.config as C
    from swe_agent.agent import RunState
    from swe_agent.management import ContextManager

    # 钉死 pytest 单杠始终失败（signature-agnostic：任意连续失败都计）
    monkeypatch.setattr(S._harness, "_run_test_bar",
                        lambda msgs: ("fail", "3 failed", "traceback..."))

    ctx = RunState(role="executor")
    ctx.cm = ContextManager()  # _l1_gate 调用 ctx.cm.to_list()

    # 第 1 次 pytest 失败：仍 continue（给一次重跑），不置 unsolvable
    r1 = S._l1_gate(ctx, "continue")
    assert r1 == "continue", f"首次 pytest 失败应 continue（给重跑），实际 {r1}"
    assert not ctx.metadata.get("unsolvable"), "单次失败不应判 unsolvable"
    assert ctx.guard.value("bar_consec_fail") == 1, "consec_bar_fail 应为 1"

    # 第 2 次 pytest 失败（达 VAL_DOOMED_THRESHOLD=2）：判定 unsolvable + break
    r2 = S._l1_gate(ctx, "continue")
    assert r2 == "break", (
        f"连续 {C.VAL_DOOMED_THRESHOLD} 次失败应 break 终止，实际 {r2}")
    assert ctx.metadata.get("unsolvable") is True, "连续失败应置 unsolvable"
    assert ctx.guard.value("bar_consec_fail") == 2

    # 反向：pytest 通过后连续计数清零（独立 ctx，避免 unsolvable 终态标志干扰）
    monkeypatch.setattr(S._harness, "_run_test_bar",
                        lambda msgs: ("pass", "all green", ""))
    monkeypatch.setattr(S._verify, "verify_gate", lambda: ("pass", "ok"))
    ctx2 = RunState(role="executor")
    ctx2.cm = ContextManager()
    r_pass = S._l1_gate(ctx2, "continue")
    assert r_pass == "done", f"pytest 通过 + tester 通过应 done，实际 {r_pass}"
    assert ctx2.guard.value("bar_consec_fail") == 0, "pytest 通过后连续计数应清零"


def test_l1_gate_caps_consecutive_no_tests_as_unsolvable(monkeypatch):
    """P1 扩展回归（2026-09-02 e2e 验证）：弱模型在「写了但 1 失败」(fail) 与
    「根本没写测试」(no_tests) 间摆动时，no_tests 必须也计入 doomed 连续计数，
    否则连续链被反复清零、永远凑不到阈值 → 烧满 MAX_ATTEMPTS 才判 failed。只有 bar 真正
    pass 才清零。这里钉死 _run_test_bar=no_tests，验证连续 2 次即 unsolvable。"""
    import swe_agent.supervisor as S
    from swe_agent.agent import RunState
    from swe_agent.management import ContextManager
    import swe_agent.config as C

    monkeypatch.setattr(S._harness, "_run_test_bar",
                        lambda msgs: ("no_tests", "工作区尚无任何测试文件", ""))

    ctx = RunState(role="executor")
    ctx.cm = ContextManager()

    # 第 1 次 no_tests：continue（给一次重跑），不置 unsolvable
    r1 = S._l1_gate(ctx, "continue")
    assert r1 == "continue", f"首次 no_tests 应 continue（给重跑），实际 {r1}"
    assert not ctx.metadata.get("unsolvable"), "单次 no_tests 不应判 unsolvable"
    assert ctx.guard.value("bar_consec_fail") == 1, "consec_bar_fail 应为 1"

    # 第 2 次 no_tests（达阈值）：判定 unsolvable + break（证明 fix：曾是被清零）
    r2 = S._l1_gate(ctx, "continue")
    assert r2 == "break", (
        f"连续 {C.VAL_DOOMED_THRESHOLD} 次 no_tests 应 break 终止，实际 {r2}")
    assert ctx.metadata.get("unsolvable") is True, "连续 no_tests 应置 unsolvable"

    # 反向（摆动不互相清零）：上一段证明 fail→no_tests 累加；这里独立验证
    # fail(1)→no_tests(2) 在第 2 次即 break（而非被 no_tests 重置回 continue）。
    monkeypatch.setattr(S._harness, "_run_test_bar",
                        lambda msgs: ("fail", "1 failed", "tb"))
    ctx3 = RunState(role="executor"); ctx3.cm = ContextManager()
    assert S._l1_gate(ctx3, "continue") == "continue"   # fail(1) → continue
    monkeypatch.setattr(S._harness, "_run_test_bar",
                        lambda msgs: ("no_tests", "no tests", ""))
    r3 = S._l1_gate(ctx3, "continue")
    # 关键：no_tests 不清零 fail 累计的 1，第 2 次即达阈值 → break（若被清零会是 continue）
    assert r3 == "break" and ctx3.metadata.get("unsolvable") is True, \
        "fail→no_tests 应连续累加（不互相清零）至 unsolvable"


def test_executor_prompts_enforce_classification_and_language_agnostic():
    """P2 回归（2026-09-02）：executor 两条 prompt 必须满足以下约束：
    1. 含「代码逻辑 vs 测试逻辑」分流诊断引导（测试失败必须先分类，禁止无脑改到全绿）；
    2. 语言无关：不含 python 专属命令（python -m / py_compile），改用 compile / 测试套件；
    3. 以 task spec 为改测试的唯一权威（两条 prompt 均须点名 task spec）。
    WEAK_SYSTEM_PROMPT 于 2026-09-12 改为 fixbug 导向版，不再点名 verify_points
    （弱模型上下文预算留给 debug 主链路），故「verify_points 不可作改测试依据」
    一条仅对 SYSTEM_PROMPT 生效。
    纯静态字符串断言，不连 LLM / 网络 / 文件系统。
    """
    import swe_agent.config as C
    for name in ("SYSTEM_PROMPT", "WEAK_SYSTEM_PROMPT"):
        p = getattr(C, name)
        # 约束 1：分流诊断
        assert "代码逻辑" in p and "测试逻辑" in p, f"{name} 缺「代码逻辑/测试逻辑」分流引导"
        # 约束 2：语言无关（禁用 py_compile 与 python -m 专属命令）
        assert "py_compile" not in p, f"{name} 仍含 py_compile（应语言无关，改用 compile）"
        assert "python -m" not in p, f"{name} 仍含 python 专属命令（应语言无关）"
        # 约束 3：task spec 为改测试权威（两条 prompt 均须点名）
        assert "task spec" in p, f"{name} 未以 task spec 为改测试权威"
    # 约束 3b：verify_points 不可作改测试依据 —— 仅 SYSTEM_PROMPT（WEAK 版有意省略）
    assert "verify_points" in C.SYSTEM_PROMPT, \
        "SYSTEM_PROMPT 应显式点名 verify_points 并声明其不可作为改测试依据"


# ======================================================================
# Bug B / 同构风险：executor 的 complete 由外层 L1/L2 gate 按测试结果终止（有界，非无限空转）
# 同样 model-free + tool-free。
# ======================================================================
def test_executor_completes_via_outer_gates(monkeypatch):
    """证明 executor 的 stop_action `complete` 不靠 L3 的 on_iter_end，而是靠嵌套外层的
    _l2_gate（lint）/ _l1_gate（pytest+tester）按测试结果判定 done——因此是有界终止，
    不会像 tester 那样在没有 on_iter_end 时无限空转。

    与 tester 的对比：tester 的循环是扁平 single_loop（无外层 gate），所以缺 on_iter_end
    就真空转；executor 的 L3 虽也无 on_iter_end，但外层 gate 会在 MAX_STEPS 次重抽后接管并
    按测试结论终止。本测试锁死这条「有界」不变量，若有人把外层 gate 改坏成不终止，这里会红。

    工具全 fake：complete 等只记录调用、不真正执行任何副作用；模型按 fallback 一直想
    complete（模拟弱模型反复提交 complete），验证外层 gate 有界接管。
    """
    import swe_agent.supervisor as SUP
    import swe_agent.config as C

    # 外层闸门依赖的 lint / pytest 单杠 / tester 全部 stub 为通过（避免真实子进程）
    monkeypatch.setattr("swe_agent.harness.run_lint", lambda lang: ("pass", "ok", None))
    monkeypatch.setattr("swe_agent.harness._run_test_bar", lambda msgs: ("pass", "ok", ""))
    monkeypatch.setattr(V, "verify_gate", lambda: ("pass", "ok"))
    # build_context 读 GLOBAL_STATE['goal']，测试环境无 state；stub 掉（对终止逻辑无关）
    monkeypatch.setattr(SUP, "build_context", lambda subtask: subtask)

    rig = ToolFreeRig(monkeypatch)
    model = ScriptedModel([], fallback=("complete", {"summary": "done"}))
    monkeypatch.setattr(M, "chat_toolcalls", model)
    # 把 L3 上限压到很小，验证「有界」：complete 最多被重抽 MAX_STEPS 次即被外层 gate 接管
    monkeypatch.setattr(C, "MAX_STEPS", 3)

    from swe_agent.agent import RunState
    ctx = RunState(role="executor")
    ctx.metadata["exec_load_unload"] = False  # 关闭测试中的模型装卸副作用
    agent = SUP.build_executor_agent(ctx)
    agent.ctx.cm.compress_if_needed = lambda c: None  # 关闭压缩副作用

    reason = agent.run([{"role": "user", "content": "task"}])

    assert rig.count("complete") <= C.MAX_STEPS + 1, (
        f"executor 未在 L3 MAX_STEPS 内有界终止（complete 被重抽 {rig.count('complete')} 次），"
        "若外层 _l1_gate/_l2_gate 不再接管会退化为空转。")
    assert reason == "done", f"executor 外层 gate 未判定完成：{reason}"


def test_render_task_hardens_unreadable_file_skipped_rule():
    """提示词固化回归（model-free + tool-free）：_render_task 必须明确告诉 tester——
    单条验收点因「文件读不到 / 只读工具不可用」无法核对时，verdict 标 skipped 且 evidence 写 20 字内原因。
    防止「以事实为基础 + 不可读→skipped + ≤20字原因」这一层固化被静默回退。
    """
    task = V._render_task("实现 fib", [
        {"id": 1, "point": "fib(0) 返回空列表", "check_hint": "assert fib(0) == []"},
    ])
    # 以事实为基础
    assert "以事实" in task, "提示词未固化「以事实为基础」"
    # 因不可读 → skipped
    assert "skipped" in task, "提示词未固化 skipped 判定"
    assert "文件不可读" in task or "读不到" in task or "无法核对" in task, \
        "提示词未固化「文件读不到/无法核对 → 该条 skipped」"
    # 20 字内简要原因
    assert "20" in task, "提示词未固化 evidence 20 字内约束"


def test_tester_system_prompt_hardens_unreadable_file_skipped_rule():
    """提示词固化回归（model-free + tool-free）：_TESTER_SYSTEM 必须含「不可读→skipped + 20字内原因」。"""
    sys_p = V._TESTER_SYSTEM
    assert "以事实" in sys_p, "_TESTER_SYSTEM 未固化「以事实为基础」"
    assert "文件不可读" in sys_p or "读不到" in sys_p, \
        "_TESTER_SYSTEM 未固化「文件读不到 → skipped」"
    assert "20" in sys_p, "_TESTER_SYSTEM 未固化 evidence 20 字内约束"
    assert "不得猜" in sys_p, "_TESTER_SYSTEM 未固化「不得猜 pass/fail」"


def test_write_file_does_not_inline_compile_check(monkeypatch):
    """回归（2026-09-02）：写工具返回值只描述「写入本身」，绝不内联编译/lint 失败。

    这是 fizzbuzz e2e 死循环的根因之一——tools.py 曾在每次 write/edit 后追加
    `[编译检查未通过]`，把「写成功」与「lint 失败」揉成一团，弱模型分不清
    「文件已落盘仅 line N 坏」还是「写入失败」，于是把整件事当失败重写整个文件
    （premature+punitive 反馈节奏）。按架构 lint 只在 loop_2 收尾闸门跑，不在写工具内联。
    """
    import swe_agent.tools as T
    # 跳过真实写盘（不触碰 brokered sandbox 的 FS），只验证返回值语义
    monkeypatch.setattr(T, "_write_verify", lambda p, text: None)
    # 绕过弱模型行数上限分支：直接 patch swe_agent.models（与 guards.M 同一模块对象，
    # WriteSizeGuard/ReadSizeGuard 经 M.is_weak_executor 读取）。tools.py 已不再暴露 M。
    monkeypatch.setattr(M, "is_weak_executor", lambda: False)
    # 1) 语法错误的 .py：写工具只回报写入成功，绝不内联编译检查
    bad = "def f(:\n    pass\n"   # SyntaxError
    r = T.write_file("impl_bad.py", bad)
    assert r.startswith("write_success:"), f"应报 write_success，实际：{r}"
    assert "[编译检查未通过]" not in r, f"写工具不应内联编译检查：{r}"
    assert "py_compile" not in r, f"写工具返回值不应含 py_compile：{r}"
    # 2) 合法 .py：同样只回报写入成功
    ok = "def f():\n    return 1\n"
    r2 = T.write_file("impl_ok.py", ok)
    assert r2.startswith("write_success:"), f"合法文件应报 write_success，实际：{r2}"
    assert "[编译检查未通过]" not in r2, f"合法文件不应出现编译检查：{r2}"


def test_l2_gate_caps_consecutive_lint_failures_as_unsolvable(monkeypatch):
    """回归（2026-09-02）：loop_2 lint 连续失败熔断 → unsolvable。

    弱模型写不出可编译代码时（如 lfm2.5-2.6b 把 print("\\n".join) 写成真实换行），
    loop_2 会陷入「lint失败→重编码→lint失败」空转。复用 VAL_DOOMED_THRESHOLD(=2)：
    连续 lint 失败达阈值即判 unsolvable，提前终止；lint 通过即清零（保持「连续」语义）。
    """
    import swe_agent.supervisor as S
    from swe_agent.agent import RunState
    from swe_agent.management import ContextManager
    # 钉死 run_lint = 失败
    monkeypatch.setattr(S._harness, "run_lint", lambda lang: ("fail", "syntax error at line 1", ""))
    ctx = RunState(role="executor"); ctx.cm = ContextManager()
    r1 = S._l2_gate(ctx, "limit")
    assert r1 == "continue", f"第 1 次 lint 失败应 continue，实际 {r1}"
    assert ctx.guard.value("lint_consec_fail") == 1, "连续 lint 失败计数应为 1"
    r2 = S._l2_gate(ctx, "limit")
    assert r2 == "break", f"连续 2 次 lint 失败应 break（unsolvable），实际 {r2}"
    assert ctx.metadata.get("unsolvable") is True, "应置 unsolvable"
    # 反向：lint 通过 → 计数清零，返回 break（出 round 进 pytest）
    monkeypatch.setattr(S._harness, "run_lint", lambda lang: ("pass", "", ""))
    r3 = S._l2_gate(ctx, "limit")
    assert r3 == "break", f"lint 通过应 break，实际 {r3}"
    assert ctx.guard.value("lint_consec_fail") == 0, "lint 通过后连续计数应清零"


# ======================================================================
# 单轮多工具调用（parallel tool calls）—— harness 必须全部执行，不得丢弃
# ======================================================================
def _run_one_turn(monkeypatch, role, turn, allowed):
    """驱动 Agent 严格跑【一轮】模型响应，返回 (reason, rig, msgs, ctx)。

    on_iter_end 对任何 reason 都 break，确保只跑一轮，便于断言「一轮内发生了什么」。
    """
    import swe_agent.roles_config as RC
    from swe_agent.agent import Agent, RunState

    rig = ToolFreeRig(monkeypatch)
    monkeypatch.setattr(
        "swe_agent.agent.ROLE_TOOLS_ALLOWED",
        {role: set(allowed)},
    )
    model = ScriptedModel([turn], fallback=None)
    monkeypatch.setattr(M, "chat_toolcalls", model)

    rc = RC.make_role_config(role)
    ctx = RunState(role=role)
    agent = Agent(rc, RC.single_loop(max_iter=1, on_iter_end=lambda c, r: "break"), ctx=ctx)
    reason = agent.run([{"role": "user", "content": "task"}])
    return reason, rig, ctx.cm.to_list(), ctx


def _tool_result_ids(msgs):
    """消息序列里所有 tool 结果的 tool_call_id（按顺序）。"""
    return [m.get("tool_call_id") for m in msgs if m.get("role") == "tool"]


def _assistant_call_ids(msgs):
    """消息序列里 assistant 声明的 tool_call_id（按顺序）。"""
    ids = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids += [tc["id"] for tc in m["tool_calls"]]
    return ids


def test_single_turn_executes_every_tool_call(monkeypatch):
    """核心回归（2026-09-03）：模型一轮下发多个 tool_call 时，harness 必须**全部执行**。

    旧行为只跑 tcs[0]、其余回「（已忽略：每轮只执行第一个工具调用）」假消息——强模型
    （Ling 等）一轮多步计划被砍成一步，模型再聪明也白搭。这是「模型聪明但 harness
    基础能力欠缺」的直接来源，必须锁死。
    """
    turn = [("read_file", {"path": "src/a.py"}),
            ("read_file", {"path": "src/b.py"}),
            ("grep", {"pattern": "def main", "path": "src"})]
    reason, rig, msgs, _ctx = _run_one_turn(monkeypatch, "executor", turn,
                                            {"read_file", "grep"})
    # 三个调用全部真正派发到工具层（顺序执行）
    assert rig.sequence() == ["read_file", "read_file", "grep"], (
        f"一轮内的多个 tool_call 应全部执行，实际只跑了 {rig.sequence()}")
    assert [kw.get("path") for (_n, kw) in rig.calls] == ["src/a.py", "src/b.py", "src"], (
        "每个调用的参数必须原样透传，不得串位")
    # 协议完整性：每个声明的 tool_call_id 都必须有且仅有一条回执
    assert _tool_result_ids(msgs) == _assistant_call_ids(msgs), (
        "每个 tool_call_id 都必须有一条对应的 tool 结果消息（否则 OpenAI 协议下一轮报 400）")


def test_turn_cap_truncated_but_every_id_answered(monkeypatch):
    """超出 C.MAX_ACTIONS_PER_RESPONSE 的调用不执行，但仍必须回执（协议完整性）。

    防模型一次吐 N 个调用烧爆上下文；被截断的调用要拿到可读原因，而不是静默消失
    或让 provider 因缺 tool 回执报 400。
    """
    import swe_agent.config as C
    cap = C.MAX_ACTIONS_PER_RESPONSE
    turn = [("read_file", {"path": f"f{i}.py"}) for i in range(cap + 2)]
    reason, rig, msgs, _ctx = _run_one_turn(monkeypatch, "executor", turn, {"read_file"})
    assert len(rig.calls) == cap, (
        f"单轮执行数应被 C.MAX_ACTIONS_PER_RESPONSE 收口到 {cap}，实际 {len(rig.calls)}")
    assert _tool_result_ids(msgs) == _assistant_call_ids(msgs), (
        "被上限截断的调用也必须各有一条 tool 回执")
    truncated = [m for m in msgs if m.get("role") == "tool" and "未执行" in (m.get("content") or "")]
    assert len(truncated) == 2, f"应有 2 条「未执行」说明回执，实际 {len(truncated)}"
    assert "上限" in truncated[0]["content"], "截断回执必须说明原因（供模型下一轮自纠）"


def test_stop_action_halts_remaining_calls_in_same_turn(monkeypatch):
    """一轮内终止动作（finish_verify/complete）命中后，同轮剩余调用不再执行，
    但仍各回一条说明——终止语义 + 协议完整性同时成立。
    """
    payload = {"results": [{"id": 1, "verdict": "pass", "evidence": "ok"}]}
    turn = [("read_file", {"path": "src/a.py"}),
            ("finish_verify", payload),
            ("read_file", {"path": "src/b.py"})]
    reason, rig, msgs, ctx = _run_one_turn(monkeypatch, "tester", turn,
                                           {"read_file", "finish_verify"})
    # 终止动作之后的调用不得执行（它的结果对已提交的验收结论没有意义）
    assert rig.sequence() == ["read_file", "finish_verify"], (
        f"终止动作后的同轮调用不应执行，实际 {rig.sequence()}")
    # 但每个 id 仍要有回执，否则下一轮 provider 报 400
    assert _tool_result_ids(msgs) == _assistant_call_ids(msgs), (
        "被跳过的调用也必须有一条 tool 回执")
    assert reason == "all_done", f"终止动作命中应返回 all_done，实际 {reason}"
    # 终止动作参数必须被捕获上送（tester 的验收结论走这条路）
    assert ctx.metadata.get("stop_result", {}).get("results"), (
        "finish_verify 的 results 必须捕获进 ctx.metadata['stop_result']")


def test_multi_call_turn_fingerprint_covers_whole_turn(monkeypatch):
    """多调用轮次的循环防护指纹必须覆盖整轮动作（而非只有第一个），
    否则「(A,B) 与 (A,C)」会被误判为同一动作，触发虚假 stuck。
    """
    import swe_agent.roles_config as RC
    from swe_agent.agent import Agent, RunState

    rig = ToolFreeRig(monkeypatch)
    monkeypatch.setattr("swe_agent.agent.ROLE_TOOLS_ALLOWED",
                        {"executor": {"read_file", "grep"}})
    # 三轮：首元素相同、次元素不同 → 整轮指纹必须不同 → consec_repeat 恒为 1（不触发 stuck）。
    # 若指纹只覆盖本轮第一个动作（旧形态），三轮指纹全同 → iter3 触顶 LOOP_REPEAT_THRESHOLD
    # → 误判 stuck（虚假停滞，会白白砍掉模型的一次多步计划）。
    turn1 = [("read_file", {"path": "a.py"}), ("grep", {"pattern": "x", "path": "."})]
    turn2 = [("read_file", {"path": "a.py"}), ("grep", {"pattern": "y", "path": "."})]
    turn3 = [("read_file", {"path": "a.py"}), ("grep", {"pattern": "z", "path": "."})]
    model = ScriptedModel([turn1, turn2, turn3], fallback=None)
    monkeypatch.setattr(M, "chat_toolcalls", model)

    rc = RC.make_role_config("executor")
    ctx = RunState(role="executor")
    # 跑满 3 轮（只有 stuck 才 break），确保指纹差异真的被检验到
    agent = Agent(rc, RC.single_loop(max_iter=3,
                                     on_iter_end=lambda c, r: "break" if r == "stuck" else None),
                  ctx=ctx)
    reason = agent.run([{"role": "user", "content": "task"}])
    assert reason != "stuck", (
        "三轮的第二个调用各不相同（pattern x/y/z），整轮指纹应不同，不得误判重复停滞")
    assert len(rig.calls) == 6, f"三轮 × 2 调用应全部执行，实际 {len(rig.calls)}"
    assert ctx.guard.value("consec_repeat") == 1, (
        f"不同指纹应把 consec_repeat 重置为 1，实际 {ctx.guard.value('consec_repeat')}")


# ======================================================================
# Tester Judge 预判（2026-09-03）：交付有效性用结构化 judge 把关，取代 F3 类长度启发式
# ======================================================================
def test_tester_judge_preflight_rejects_empty_delivery(monkeypatch):
    """客观前置（非空实现/测试文件 + pytest 收集>0）不达标 → 直接 fail。

    不调用模型 judge、不进深度 tester agent（省调用、符合「不能是空文件不能是空测试」）。
    """
    import swe_agent.models as M
    called = {"judge": 0, "run_tester": 0}
    monkeypatch.setattr(V, "_delivery_summary",
                        lambda: ("实现文件 0 非空 / 1 总；测试文件 0 非空；pytest 收集 0", False))
    monkeypatch.setattr(M, "judge", lambda kind, content, model_override=None: called.__setitem__("judge", called["judge"] + 1) or ("yes", "ok"))
    monkeypatch.setattr(V, "run_tester", lambda *a, **k: called.__setitem__("run_tester", called["run_tester"] + 1) or ("pass", "x", []))

    verdict, detail = V.verify_gate(model="qwen2.5.1-coder-7b-instruct", max_iter=2)
    assert verdict == "fail", f"空交付应 fail，实际 {verdict}: {detail}"
    assert called["judge"] == 0, "客观不达标时不应调用模型 judge"
    assert called["run_tester"] == 0, "客观不达标时不应进深度 tester agent"


def test_tester_judge_preflight_objective_ok_but_judge_no_fails(monkeypatch):
    """客观达标但模型 judge 判 no（无效单测/实现）→ fail，且不进深度 tester agent。"""
    import swe_agent.models as M
    called = {"judge": 0, "run_tester": 0}
    monkeypatch.setattr(V, "_delivery_summary",
                        lambda: ("实现文件 1 非空；测试文件 1 非空；pytest 收集 3", True))
    def fake_judge(kind, content, model_override=None):
        called["judge"] += 1
        return ("no", "测试只是占位符，无真实断言")
    monkeypatch.setattr(M, "judge", fake_judge)
    monkeypatch.setattr(V, "run_tester", lambda *a, **k: called.__setitem__("run_tester", called["run_tester"] + 1) or ("pass", "x", []))

    verdict, detail = V.verify_gate(model="qwen2.5.1-coder-7b-instruct", max_iter=2)
    assert verdict == "fail", f"judge 判 no 应 fail，实际 {verdict}: {detail}"
    assert called["judge"] == 1, "客观达标时应调用一次模型 judge"
    assert called["run_tester"] == 0, "judge 判 no 时不应进深度 tester agent（快失败）"
    assert "judge" in detail, f"fail 原因应说明是 judge 判定，实际：{detail}"


# ======================================================================
# 传输层回归（2026-09-03）：Ling 作 judge 的 thinking_off 必须经 extra_body 透传
# ======================================================================
def test_ling_judge_thinking_off_passthrough_via_extra_body(monkeypatch):
    """Ling 特有的回归：judge 用 Ling-3.0-Tiny（thinking_off=True）时，

    `chat_template_kwargs={"enable_thinking": False}` 必须经 OpenAI SDK 的 `extra_body`
    特殊参数合进请求体顶层；绝不能泄漏成 `client.chat.completions.create()` 的【顶层 kwarg】
    （否则 OpenAI SDK 抛 `Completions.create() got an unexpected keyword argument
    'chat_template_kwargs'`，judge 整段 TypeError 失败 → analyzer/tester 两道闸全退化）。

    为何 Ling 特有：只有 judge_params 命中 "ling" 子串才设 thinking_off=True；qwen/Spark 路径
    extra_body 为空、等价 create(**payload) 不传该 kwarg，故只有 Ling 在真实 loop 撞上。
    单测 mock + 非 Ling 路径都抓不到，必须 fake client 拦截 create 的 kwarg 形状才能守。
    """
    captured = {}

    class _FakeMsg:
        def __init__(self):
            self.role = "assistant"
            self.content = '{"result": "yes", "reason": "ok"}'   # judge 可解析的合法 JSON
            self.tool_calls = None
    class _FakeChoice:
        def __init__(self):
            self.message = _FakeMsg()
    class _FakeResp:
        def __init__(self):
            self.choices = [_FakeChoice()]
            self.usage = None
    class _FakeCompletions:
        def create(self, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return _FakeResp()
    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()
    class _FakeClient:
        def __init__(self):
            self.chat = _FakeChat()

    # 替换 LM Studio 客户端工厂（绕过单例缓存 + 真实联网），只抓 kwarg 形状
    monkeypatch.setattr(L, "_get_client", lambda: _FakeClient())
    monkeypatch.setattr(L, "_client", None)

    # 走真实 judge 链路：model_override=Ling-3.0-Tiny → judge_params 命中 ling 子串 → thinking_off=True
    res, reason = M.judge("analyzer", "工作区为空，需创建 src/fib.py 与 test_fib.py。",
                          model_override="Ling-3.0-Tiny")
    assert res == "yes", f"fake judge 应解析为 yes，实际 ({res!r},{reason!r})"

    # 核心断言：chat_template_kwargs 绝不能成为 create() 的顶层 kwarg
    assert "chat_template_kwargs" not in captured, \
        "回归！chat_template_kwargs 泄漏成 create() 顶层 kwarg → SDK 会 TypeError。应走 extra_body。"
    # 必须且只能经 extra_body 嵌套透传（LM Studio 读取位置）
    assert captured.get("extra_body") == {"chat_template_kwargs": {"enable_thinking": False}}, \
        f"Ling thinking_off 未正确经 extra_body 透传：captured.extra_body={captured.get('extra_body')!r}"

    # 对照：同一链路下模型名确为 Ling（证明走的是 Ling 特有覆盖，而非默认 qwen 路径）
    assert captured.get("model") == "Ling-3.0-Tiny", \
        f"judge 应打向 Ling-3.0-Tiny，实际 {captured.get('model')!r}"
