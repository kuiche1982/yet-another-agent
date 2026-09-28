#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/agent.py —— 统一 Agent 抽象（重构核心）

设计目标（来自与用户的 OOP 重构讨论，已在项目记忆归档）：
- 一个 Agent 类 + 两条正交轴参数化，不做角色子类爆炸：
    * RoleConfig（who）：system_prompt / tools / mode / parser / model_override / max_tokens
    * LoopConfig（how）：max_iter / child / on_iter_end / before_step / hooks
- 五角色（planner/analyzer/tester/executor/compact）与三层 loop
  （loop_1/loop_2/loop_3，实为同一 executor 角色在三种 LoopConfig 粒度上的嵌套）
  都是 Agent 的实例：executor = Agent(R_exec, L1(L2(L3)))。
- 两个横切子系统（ModelManager / ContextManager）全部走 **per-agent hook**，
  且由工厂在**组装时条件绑定**（GLM 不要求 load/unload → 不传 hook → 零调用零开销）：

      pre_loop  → ModelManager.load      （仅 load_unload=True 的模型才绑）
      post_loop → ModelManager.unload
      before_step → ContextManager.compress_if_needed（按目标模型预算长度感知压缩）

- 核心 Agent 类 **不直接引用 ModelManager**（模型生命周期走回调 / hook）；
  但 ContextManager 作为「对话 buffer 拥有者」是 Agent 运行态（RunState.cm）的一部分，
  Agent 经 ctx.cm 读写对话与循环防护状态。符合「BUILD 层不该耦合」「prompt 约束进 BUILD 层」原则。

复用现有地基（非平地起楼）：
- models.chat_text_messages / chat_text_escalating / chat_toolcalls 负责按 role 路由到 provider；
- registry.ToolRegistry.glm_tools / dispatch 负责工具 schema 与执行；
- 循环防护（重复动作 / 空响应 / 未调工具）逻辑从 supervisor._run_loop3_executor 原样收敛进 _apply_toolcall。
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import models as M
from . import config as C
from .guard import Guard, default_guard
from .hooks import HookPoint, HOOK_HUB, GateAction
from .management import ContextManager
from .registry import ToolRegistry, ActionContext, ROLE_TOOLS as ROLE_TOOLS_ALLOWED
from swe_agent.log import logger


# ======================================================================
# 交互模式（与 models.FCCapability 正交：fc 描述「能否 toolcall」，mode 描述「怎么调」）
# ======================================================================
class AgentMode(str, Enum):
    TEXT = "text"          # 摘要/压缩：chat_text_messages（无 tools）
    JSON = "json"          # planner：chat_text_escalating(+response_format)
    TOOLCALL = "toolcall"  # executor/analyzer/tester：chat_toolcalls（原生 toolcall 循环）


# ======================================================================
# 配置数据形状
# ======================================================================
@dataclass
class RoleConfig:
    """角色「是谁」：提示词 + 工具 + 交互模式 + 解析器 + 模型绑定。

    tools=None 表示「由 registry 按 role 名取该角色工具集」（默认行为）；
    显式传 list 则直接用（如 tester 用 ToolRegistry.glm_tools("tester")）。
    model_override 非空时忽略角色默认模型（compact 副驾 / 升级子 agent 用）。
    """
    name: str
    mode: AgentMode
    system_prompt: str = ""
    tools: Optional[list] = None
    parser: Optional[Callable] = None
    model_override: Optional[str] = None
    max_tokens: Optional[int] = None
    # 终止动作名（命中即视为本层循环结束条件之一），如 executor 的 complete / tester 的 finish_verify
    stop_actions: Tuple[str, ...] = ()
    # stop 信号约定（BUILD 层，按角色在 make_role_config 实例化时声明，不写死角色名）：
    #   True  = 尊重 finish_reason=stop 作「声明完成」信号，不逼模型调工具；交外层闸门判定
    #           （executor 的 lint+pytest+tester 单杠：过则收尾、不过回灌失败继续修）。
    #   False = 不遵守 stop，必须逼出对应工具（planner→JSON / tester→finish_verify 结构化验收）。
    respect_stop: bool = False
    # 工具结果压缩器（可选）：大体积工具结果入主上下文前先压缩；fail-open。
    # 例：supervisor 把 _sidecar.compress_content 注入此处，让 agent 核心不耦合压缩实现。
    result_compress: Optional[Callable[[str], Any]] = None
    # JSON 模式专用：结构化输出格式（可传 dict 或「返回 dict 的 callable」，如 _planner_response_format）
    # 与 require_json（主产出不可解析为 JSON 时升级子 agent）。仅 JSON 模式消费。
    response_format: Optional[Any] = None
    require_json: bool = False
    # 全量工具放行（REPL ChatAgent 用）：True 时 _apply_toolcall 跳过 ROLE_TOOLS 白名单，
    # 已注册工具一律可执行（含插件运行时注入的动态工具）。
    allow_all_tools: bool = False
    # 聊天/对话模式终止信号：True 且 respect_stop 为真时，模型返回纯文本（无 tool_calls）
    # 即视为本轮对话结束（不再逼它调工具），_apply_toolcall 返回 "all_done"。
    stop_on_no_tool: bool = False

    def model_id(self) -> Optional[str]:
        return self.model_override or M.role_model_id(self.name)


@dataclass
class RunState:
    """单次 run 内跨迭代共享的可变运行态（合并原 agent.LoopCtx 与 contracts.LoopCtx）。

    单一真相源：循环控制 + 跨层持久化状态 + 对话 buffer 一并收口于此，
    彻底消除上一轮「两个不兼容 LoopCtx 需手动回写 early_stop」的 bug 类。

    - role / iteration / early_stop / max_iter：循环控制
    - metadata：跨嵌套层持久化**结果与标志**（stop_result / unsolvable / last_bar /
      last_verify / exec_load_unload 等）与各角色自定义运行态。**不放计数器**——
      计数器一律进 guard（见下），避免「限制散落、重置时机靠人记」。
    - guard：具名限制 registry（`swe_agent/guard.py`）。每条 Limit 自带 reset_at（HookPoint 重置时机），
      由 `Agent._run_loop` 在每层迭代起始按 hook_point 自动重置、`Agent.run()` 在轮次边界全量重置。
    - cm：ContextManager 拥有的对话 buffer（压缩=内部不变量，回调经 ctx.cm.append 写入）
    """
    role: str = ""
    iteration: int = 0
    early_stop: bool = False
    max_iter: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)
    cm: Optional[ContextManager] = None
    guard: Guard = field(default_factory=default_guard)


@dataclass
class LoopConfig:
    """循环「怎么跑」：迭代上限 + 嵌套子循环 + 每轮收尾闸门 + 单步前中间件 + 条件 hook。

    on_iter_end(ctx, reason) -> "done"|"break"|"continue"：
      - "done"：整条嵌套链结束（如 tester 全过 → Agent 完成）
      - "break"：结束本层循环，向上传播（如 model_error → 提前终止）
      - "continue"/None：带反馈重跑下一轮
    hooks 为 per-agent，键为 "pre_loop"/"post_loop"/"before_step"；
      工厂按模型配置条件绑定（GLM 不绑 load/unload → 列表为空 → 零开销）。
    回调统一以 RunState 为唯一入参（ctx）；消息 buffer 经 ctx.cm 访问，不再单独透传 messages。

    repeat_threshold / no_tool_threshold：循环防护阈值，默认取全局 LOOP_REPEAT_THRESHOLD，
      允许各角色单独放宽（如 analyzer 用更宽松的阈值，配合「失败不计次数」让弱模型自愈）。
      阈值在 Agent 构造时登记进 ctx.guard（计数只在 guard 一处，不再散落）。

    hook_point：本层循环的生命周期锚点（`HookPoint`），同时决定**哪些限制在这层的迭代起始被重置**
      （`Agent._run_loop` 自动调 `guard.reset_at(loop.hook_point)`）。executor 三层嵌套约定
      l3→L3_LOOP_START / l2→L2_LOOP_START / l1→L1_LOOP_START；单层角色（analyzer/tester/chat）
      默认 L1_LOOP_START（无 L2/L3 层 → stall 限制在整个 run 内累计，与重构前行为一致）；
      RUN_START 保留给 `Agent.run()`（轮次边界，全量重置），loop 层不得使用。
    """
    max_iter: int
    child: Optional["LoopConfig"] = None
    on_iter_end: Optional[Callable[[RunState, str], Optional[str]]] = None
    on_iter_start: Optional[Callable[[RunState, int], None]] = None
    before_step: Optional[Callable[[RunState], None]] = None
    hooks: Dict[str, List[Callable]] = field(default_factory=dict)
    # 本层生命周期锚点：loop 迭代起始按它重置同 reset_at 的限制
    hook_point: HookPoint = HookPoint.L1_LOOP_START
    # 连续重复动作 → 停滞 阈值（默认全局值，可逐角色放宽）
    repeat_threshold: int = 3
    # 连续未调工具 → 停滞 阈值（默认全局值，可逐角色放宽）
    no_tool_threshold: int = 3

    def add_hook(self, phase: str, fn: Callable) -> "LoopConfig":
        self.hooks.setdefault(phase, []).append(fn)
        return self


# ======================================================================
# Agent 类
# ======================================================================
class Agent:
    """统一 Agent：六角色 + 三层 loop 都是它的实例。

    核心约束：run/_run_loop/_step 不直接感知 ModelManager（模型生命周期走回调 / hook），
    只跑 loop.hooks。对话 buffer 与循环防护状态由 RunState.cm（ContextManager）统一持有，
    循环体一律经 ctx.cm 读写，不再裸透传 messages。
    """

    def __init__(self, role: RoleConfig, loop: LoopConfig, ctx: Optional[RunState] = None):
        self.role = role
        self.loop = loop
        self.ctx = ctx or RunState(role=role.name)
        if not self.ctx.role:
            self.ctx.role = role.name
        if self.ctx.cm is None:
            self.ctx.cm = ContextManager()

    # ---- 对外入口 ----
    def run(self, messages: Optional[List[Dict[str, Any]]] = None) -> str:
        if messages is not None:
            self.ctx.cm.reset(messages)  # 初始对话注入 buffer（单发角色 / supervisor 主循环）
        # 轮次边界（REPL 每轮用户输入 / UNATTEND 整个 run 起始）：全量重置 guard——
        # 修 REPL 旧 bug：guard 从不重置导致跨轮串味。emit RUN_START 供观察型 hook 订阅。
        self.ctx.guard.reset_at(HookPoint.RUN_START)
        HOOK_HUB.emit(HookPoint.RUN_START, ctx=self.ctx)
        for h in self.loop.hooks.get("pre_loop", []):
            h(self.ctx)
        reason = self._run_loop(self.loop)
        for h in self.loop.hooks.get("post_loop", []):
            h(self.ctx)
        return reason

    # ---- 嵌套循环驱动 ----
    def _run_loop(self, loop: LoopConfig) -> str:
        last = "limit"
        for it in range(1, loop.max_iter + 1):
            self.ctx.iteration = it
            # 本层循环迭代起始：按 hook_point 重置同 reset_at 的限制（hook 驱动，替代手工 reset_guard）。
            # 例：L2_LOOP_START 的 consec_repeat/no_tool_streak/empty_streak 在每 round 开始时归零；
            # RUN_START 的限制（bar/lint/drift）不受影响，跨轮累计；L1/L3 同理。
            self.ctx.guard.reset_at(loop.hook_point)
            if loop.on_iter_start:
                loop.on_iter_start(self.ctx, it)
            reason = self._run_loop(loop.child) if loop.child else self._step()
            last = reason
            # 循环级 guard：本步结束后 emit 本层 hook_point，统一消费 Y-a 结果（REJECT / BREAK_LOOP）。
            # 放在 step 之后（而非起始）是为了让 stall 计数（本步 tick）被同一轮检测到，
            # 保证「连续 N 次 → stuck」在精确第 N 步触发，与旧内联 return 'stuck' 语义一致。
            # 阈值由 guard 从 ctx.guard.limit(name).threshold 读取（含 per-role override）。
            dec = HOOK_HUB.first_block_decision(
                HOOK_HUB.emit(loop.hook_point, ctx=self.ctx))
            if dec is not None:
                if dec.action == GateAction.BREAK_LOOP:
                    # 校验熔断（UnsolvableGuard 等）：回灌 reason 并逐层终止当前 loop
                    logger.info('%s', f'[agent] loop hook 请求终止（break_loop），role={self.role.name} iter={it}')
                    return "break_loop"
                # REJECT（停滞护栏）：回灌 reason（建设性指引），并把本层 reason 记为 "stuck"。
                # 关键：不在此短路 on_iter_end —— 对齐 HEAD 内联 return 'stuck' 的传播语义，
                # stuck 作为 reason 继续流入本层（若有）的 on_iter_end 闸门做重校验
                # （如 _l1_gate 重跑 pytest+tester、_l2_gate 重跑 lint 判定 done/break），
                # 否则会掐断外层 gate、把「重试预算兜底」误判成终止。gate-less 内层 loop
                # 落空、以 stuck 向上传播（被父层 reset_at 清零后才恢复）。
                self.ctx.cm.append("user", dec.reason)
                last = "stuck"
                if loop.on_iter_end is None:
                    # gate-less 内层 loop（如 executor 的 L3）：无父闸门可重校验，
                    # 必须阈值步立即 return 'stuck' 终止（复刻 HEAD 内联 return 'stuck' 的
                    # 精确停步语义），否则会空转到 max_iter 破坏精确调用数断言。stuck 作为
                    # 结果向上传播，被父层 reset_at 清零后才恢复。
                    return "stuck"
                # 有 on_iter_end 的 loop（L1/L2）：不在此短路，把 reason 置 'stuck'
                # 流入本层 on_iter_end 闸门做重校验（_l1_gate 重跑 pytest+tester /
                # _l2_gate 重跑 lint 判定 done/break），避免掐断外层 gate。
                reason = "stuck"
            if reason == "break_loop":
                # 工具门控（BEFORE_TOOL_CALL 的 GateAction.BREAK_LOOP）权威终止当前及外层 loop，
                # 不经 on_iter_end gate 覆盖（Y-a 的「REJECT+stop loop」）。逐层返回，直到根 loop。
                logger.info('%s', f'[agent] 工具门控请求终止 loop（break_loop），role={self.role.name} iter={it}')
                return "break_loop"
            if loop.on_iter_end:
                gate = loop.on_iter_end(self.ctx, reason)
                if gate == "done":
                    logger.debug('loop_end role=%s iter=%s reason=%s gate=%s', self.role.name, it, reason, 'done')
                    return "done"
                if gate == "break":
                    logger.debug('loop_end role=%s iter=%s reason=%s gate=%s', self.role.name, it, reason, 'break')
                    return reason
                # "continue" / None → 带反馈重跑下一轮
                logger.debug('loop_iter role=%s iter=%s reason=%s gate=%s', self.role.name, it, reason, 'continue')
            else:
                logger.debug('loop_iter role=%s iter=%s reason=%s gate=%s', self.role.name, it, reason, 'none')
                # 终止动作命中但本层 loop 未挂 on_iter_end：tester/analyzer 这类自包含叶角色
                # 缺闸门 = 必空转（弱模型反复重发 finish_* 耗满 max_iter）；executor 的 complete
                # 由外层 supervisor gate 消费属合法。统一以 monitor 事件暴露根因，开 SWE_TRACE=1
                # 即可见，无需重启；离线硬闸门在 harness_selfcheck / test_harness_contracts。
                if reason == "all_done":
                    if self.role.name in ("tester", "analyzer"):
                        logger.critical('loop_no_guard_for_stop_action role=%s iter=%s fix=%s', self.role.name, it, "调用 single_loop(on_iter_end=...) 对 'all_done' 返回 'done'")
                    else:
                        logger.debug('loop_all_done_no_guard role=%s iter=%s note=%s', self.role.name, it, 'stop_action 命中且本层无 on_iter_end（由外层 gate 消费，合法）')
        logger.debug('loop_end role=%s iter=%s reason=%s gate=%s', self.role.name, loop.max_iter, last, 'limit')
        return last

    # ---- 单步（最内层 loop_3 的一轮 toolcall 交换）----
    def _step(self) -> str:
        # before_step 中间件（如 ContextManager 长度感知压缩），无返回值，直接作用于 ctx.cm
        if self.loop.before_step:
            self.loop.before_step(self.ctx)
        role = self.role
        # 组装本轮 outgoing：RAG 回落 + 双 tier 压缩 + 模型上下文窗裁剪，
        # 全部由 ContextManager.prepare_messages 内部 auto 决策（只读 buffer，不改历史）。
        # user_input 取 buffer 末条 user 文本作 RAG query；model_context_length 按角色模型解析。
        user_input = _last_user_text(self.ctx.cm.to_list())
        msgs = self.ctx.cm.prepare_messages(
            model_context_length=M.model_context_length(role.model_id()),
            user_input=user_input,
        )
        if role.mode == AgentMode.TEXT:
            out = M.chat_text_messages(msgs, role=role.name,
                                       model_override=role.model_override,
                                       max_tokens=role.max_tokens)
            return out or ""
        if role.mode == AgentMode.JSON:
            system, user = _split_messages(msgs)
            rf = role.response_format
            rf_val = rf() if callable(rf) else rf
            return M.chat_text_escalating(role.name, system, user,
                                          model_override=role.model_override,
                                          response_format=rf_val,
                                          require_json=role.require_json) or ""
        # TOOLCALL
        tools = role.tools if role.tools is not None else ToolRegistry.glm_tools(role.name)
        meta = M.chat_toolcalls(role.name, msgs, tools=tools,
                                model_override=role.model_override)
        return self._apply_toolcall(meta)

    # ---- toolcall 解析 + 执行 + 循环防护（原 _run_loop3_executor 收敛）----
    def _apply_toolcall(self, meta: Optional[Dict[str, Any]]) -> str:
        if meta is None:
            # 清空上一轮散文内容：模型不可用不是「交卷」，避免 on_iter_end 误判为有效调研
            self.ctx.metadata["last_no_tool_content"] = ""
            logger.info('%s', '[agent] 模型连续不可用（chat_toolcalls 重试后仍 None），结束本步循环。')
            return "model_error"
        if M.is_empty(meta):
            # 软空响应（finish_reason=stop 但无 content/tool_call）= 模型本轮正常产出空，属良性轮次，非失败。
            # 仅把「本轮无内容」记入对话历史（让模型下一轮自纠），随后照常进入下一轮（lint/工具闸门），
            # 不升级为 stuck/error——避免把偶发空响应误判成停滞。整体轮次/步数上限由外层 loop guard 兜底。
            # 记录 finish_reason=stop，供 tester 在 run_tester 层判定 skipped（未提交 finish_verify → 跳过验收）。
            self.ctx.metadata["last_finish_reason"] = "stop"
            self.ctx.guard.tick("empty_streak")
            self.ctx.cm.append("user", "（系统）上一轮模型未返回任何内容或工具调用。"
                                      "请通过调用工具（write_file/edit_file/shell/complete 等）推进任务，不要空响应。")
            return "continue"
        # 退化响应（type=toolcalls 但 tool_calls 为空）与「非 toolcalls」同属「模型本轮未调工具」。
        # 各角色对 finish_reason=stop 的约定不同（BUILD 层规则，落在 RoleConfig.respect_stop，
        # 不在代码里写死角色名）：
        #   - respect_stop=True（executor）：stop = 声明完成 → 尊重，不逼调工具；
        #     交外层 _l1_gate 单杠三件套（lint+pytest+tester）判定，过则收尾、不过则闸门回灌失败继续修。
        #   - respect_stop=False（planner/tester/analyzer）：必须产出格式化工具结果 → 逼出对应 stop tool。
        if meta.get("type") != "toolcalls" or not meta.get("tool_calls"):
            # 记录 finish_reason=stop，供 tester 在 run_tester 层判定 skipped（未提交 finish_verify → 跳过验收）。
            self.ctx.metadata["last_finish_reason"] = "stop"
            # 暴露本轮「未调工具的纯文本」内容，供 on_iter_end hook（如 analyzer 的 judge 评测）读取。
            # 仅当确有文本内容时才记（空散文不记，避免 hook 误判为有效调研）。
            _ntc = (meta.get("content") or "").strip()
            if _ntc:
                self.ctx.metadata["last_no_tool_content"] = _ntc
            self.ctx.guard.tick("no_tool_streak")
            if not self.role.respect_stop:
                # 必须产出格式化工具结果 → 指名逼出 stop tool（tester 直呼 finish_verify，
                # 消除「被逼'调工具'后错调 shell 重跑」的拖拽；planner 走 response_format 独立路径不撞此处）。
                stop_tool = self.role.stop_actions[0] if self.role.stop_actions else None
                if stop_tool:
                    self.ctx.cm.append("user",
                        f"（系统）你必须通过【调用工具】来输出动作，不要只回复散文。"
                        f"请调用【{stop_tool}】工具提交结果。")
                else:
                    self.ctx.cm.append("user", "（系统）你必须通过【调用工具】来输出动作，不要只回复散文。请调用一个工具。")
            # 聊天/对话模式：模型给出纯文本答复（无 tool_calls）即视为本轮对话结束，
            # 不再逼它调工具；交由 _drive_chat 的循环自然终止（返回 all_done）。
            # 关键：必须把模型的纯文本答复先写进对话历史，否则既进不了 REPL 打印、
            # 也不进后续轮上下文（"发了话没回音"的根因）。仅聊天模式做此追加，
            # executor/planner/tester 的 no-tool 分支仍走原逻辑（不污染既有行为）。
            if self.role.respect_stop and self.role.stop_on_no_tool:
                if meta.get("content"):
                    _am = {"role": "assistant", "content": meta["content"]}
                    if meta.get("reasoning"):
                        _am["reasoning_content"] = meta["reasoning"]
                    self.ctx.cm.add(_am)
                return "all_done"
            # 停滞判定已迁至 StallGuard（订阅 Lx_LOOP_START，post-step emit 消费）：
            # 连续未调工具超阈即 REJECT（回灌 reason、loop 继续）。此处只计数，不决策。
            return "continue"

        self.ctx.guard.reset("no_tool_streak")
        self.ctx.guard.reset("empty_streak")
        # 本轮调了工具 → 清掉上一轮 prose 内容，避免 on_iter_end 误判为「未调工具的散文」
        self.ctx.metadata["last_no_tool_content"] = ""
        tcs = meta["tool_calls"]
        _am = {
            "role": "assistant",
            "content": meta.get("content") or None,
            "tool_calls": [{"id": tc["id"], "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                           for tc in tcs],
        }
        # 思维链(reasoning_content) 与 tool 轮的思考文本(content) 都只作 REPL 渲染用；
        # 出向发送前由 ContextManager.prepare_messages 剥除（reasoning_content 全剥；
        # 带 tool_calls 的 assistant 消息 content 归零为 None，符合 OpenAI 协议且 UNATTEND 回发
        # 历史与改动前一致），对 UNATTEND 净影响=0。
        if meta.get("reasoning"):
            _am["reasoning_content"] = meta["reasoning"]
        self.ctx.cm.add(_am)
        # 模型本轮调了工具（finish_reason=tool_calls）→ 记录，覆盖上一轮的 stop
        self.ctx.metadata["last_finish_reason"] = "tool_calls"

        # —— 单轮多工具调用（GLOBAL 收口点）——
        # 模型一轮可下发多个 tool_call，harness **全部顺序执行**（只读/写入有先后语义，
        # 不做并发）。旧行为只跑 tcs[0]、其余回「已忽略」假消息，等价于把强模型的一轮
        # 多步计划砍成一步——这是「模型聪明但 harness 拖后腿」的直接来源。
        # 硬约束（BUILD 层，不进 prompt）：
        #   1. 每个 tool_call_id 都必须有一条 tool 结果消息（被拒/被跳过的也要回），
        #      否则 OpenAI 协议下一轮报 400；
        #   2. 单轮执行数上限 C.MAX_ACTIONS_PER_RESPONSE，防模型一次吐 N 个调用烧爆上下文；
        #      （注：工具返回值不再做长度截断，完整回灌模型，保证弱模型可见全量信息。）
        allowed = None if self.role.allow_all_tools else (ROLE_TOOLS_ALLOWED.get(self.role.name) or set())
        cap = max(1, int(C.MAX_ACTIONS_PER_RESPONSE))
        exec_args: List[Dict[str, Any]] = []
        results: List[str] = []
        stop_hit = ""
        for idx, tc in enumerate(tcs):
            fn = tc["name"]
            if stop_hit:
                # 本轮已提交终止动作（complete/finish_verify）：剩余调用不再执行，
                # 但协议要求每个 id 都有回执 → 回一条说明，不静默丢弃。
                self.ctx.cm.append("tool", f"（未执行：本轮已提交终止动作 {stop_hit}，剩余调用不再执行）",
                                   tool_call_id=tc["id"])
                continue
            if idx >= cap:
                self.ctx.cm.append("tool", f"（未执行：单轮工具调用上限 {cap}，本轮共 {len(tcs)} 个；"
                                           f"请拆到下一轮或合并成更少的调用）", tool_call_id=tc["id"])
                continue
            try:
                args = json.loads(tc["arguments"] or "{}")
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            args["action"] = fn
            if allowed is not None and fn not in allowed:
                self.ctx.cm.append("tool", f"error: {self.role.name} 不允许调用 {fn}，可用：{', '.join(sorted(allowed))}",
                                   tool_call_id=tc["id"])
                continue
            actx = ActionContext(messages=self.ctx.cm.to_list())
            actx.extra = {"role": self.role.name}
            actx.action = fn
            try:
                res = ToolRegistry.dispatch(args, actx)
            except Exception as e:  # 单工具炸掉不应带走整轮其余调用
                res = f"error: 工具 {fn} 执行异常（{type(e).__name__}: {e}）"
            # 工具调用计数收口进 guard（取代散落的 STATS["actions"] 计数；STATS 仅留遥测）：
            # 每成功派发一次工具调用 +1，reset_at=L3_LOOP_START（每步清零，观察每步调用密度）。
            self.ctx.guard.tick("tool_call_count")
            # 大体积工具结果：先压缩再入主上下文（fail-open；由 RoleConfig.result_compress 注入）
            # orig 保留压缩前的原文，经 cm.append(original=...) 落进 _truth（完整会话史），
            # 保证「工作集被压缩/裁剪，但原文不丢失、后续可按 recall 取回」。
            orig = res
            if self.role.result_compress is not None:
                try:
                    res2 = self.role.result_compress(res)
                    res = res2[0] if isinstance(res2, tuple) else res2
                except Exception:
                    pass
            self.ctx.cm.append("tool", res, original=orig, tool_call_id=tc["id"])
            # Y-a：BEFORE_TOOL_CALL 门控返回 BREAK_LOOP 时，回灌 reason 并终止本步 loop
            # （ctx.loop_break 由 dispatch 在 BREAK_LOOP 时置位；REJECT 不置位、loop 继续）。
            if actx.loop_break:
                return "break_loop"
            # trace：记录本次 tool_call（工具名 + 参数 + 结果，逐参数截断到 200）
            logger.debug('role=%s tool=%s args=%s result=%s', self.role.name, fn, args, res)
            exec_args.append(args)
            results.append(res)
            if fn in self.role.stop_actions and self._is_done(fn):
                # 捕获终止动作的参数（如 tester 的 finish_verify results），供外层消费
                self.ctx.metadata["stop_result"] = args
                stop_hit = fn
        if not exec_args:
            # 一个都没真正执行（全部越权/畸形/被闸门拦下）→ 本轮未前进，计入未调工具 streak，
            # 防弱模型反复下发非法工具名空转到 max_iter（旧代码此处直接 continue 且 streak 归零）。
            self.ctx.guard.tick("no_tool_streak")
            # 停滞判定已迁至 StallGuard（post-step emit 消费）；此处只计数，不决策。
            return "continue"

        # 循环防护：动作指纹（同一文件不同写入必须区分，避免误判重复）
        fp = turn_fingerprint(exec_args)
        # 幂等原则（对齐「确认失败就不算幂等提交次数」）：工具调用【明确失败】
        # （参数问题 / 执行错误 / shell 非 0 退出 等，见 _is_tool_failure）默认回灌模型
        # （已在上面 cm.append），但【不计入调用次数】——不推进 consec_repeat、不刷新
        # last_fp，让模型能自我纠正（如 cd /testbed 失败后改回相对路径命令）而不被
        # 重复/停滞阈值提前截断。仅受 max_iter 硬上限约束（防无限循环）。
        # 仅当【本轮全部结果都失败】时才豁免计数；只要任一工具真正产出，仍正常计数。
        _all_failed = bool(results) and all(isinstance(r, str) and _is_tool_failure(r) for r in results)
        if not _all_failed:
            # last_fp 是「上次动作指纹」比对值（状态，非计数器）→ 存 metadata，不在 guard 计数。
            _last_fp = self.ctx.metadata.get("last_fp")
            if fp == _last_fp:
                self.ctx.guard.tick("consec_repeat")
            else:
                self.ctx.guard.set("consec_repeat", 1)
                self.ctx.metadata["last_fp"] = fp
        # 停滞判定已迁至 StallGuard（订阅 Lx_LOOP_START，post-step emit 消费）：
        # 连续重复动作超阈即 REJECT（回灌建设性指引、loop 继续）。此处只计数（consec_repeat 指纹比较），不决策。
        # 终止动作命中
        if stop_hit:
            logger.info('%s', f'[agent] 终止动作 {stop_hit} 命中，本步循环结束。')
            logger.debug('stop_action_hit role=%s action=%s', self.role.name, stop_hit)
            return "all_done"
        return "continue"

    # ---- 辅助 ----
    def _is_done(self, fn: str) -> bool:
        """complete 需所有任务标记完成才算 done；其余终止动作直接 done。"""
        if fn == "complete":
            try:
                from .supervisor import get_current_task
                return get_current_task() is None
            except Exception:
                return True
        return True


# ======================================================================
# 模块级辅助（从 supervisor 迁移的纯函数，避免循环 import）
# ======================================================================
# 动作指纹：本文件【不再自带实现】——历史上 agent.py 与 state.py 各存一份
# _turn_fingerprint，两份实现漂移（agent 版对 grep/glob 只取 path、丢 pattern，导致
# 「grep x」与「grep y」被判为同一动作 → 虚假 stuck；state 版的多动作分支又只用动作名
# 拼接，同样丢参数）。现统一收敛到 state.py 的单一实现（GLOBAL 收口，消除漂移）。
from .state import _turn_fingerprint as turn_fingerprint


# 工具调用「明确失败」标记集（对齐「确认失败就不算幂等提交次数」原则）。
# 凡命中以下任一前缀，或 shell 以非 0 退出（含 `cd /testbed` 这类幻觉根目录失败），
# 均视为「失败」：默认回灌模型（由 _apply_toolcall 的 cm.append 完成），但不计入
# consec_repeat / 调用次数，让模型能自我纠正而不被重复/停滞阈值提前截断。
_TOOL_FAILURE_PREFIXES = (
    "error:", "edit_error:", "shell_error:", "read_error:", "write_error:",
    "grep_error:", "glob_error:", "mcp_error:", "skill_error:", "subagent_error:",
    "task_error:", "compact_error:", "denied:", "sensor internal error:",
)


def _is_tool_failure(res) -> bool:
    """判断一条工具回执是否代表『明确失败』。

    命中条件：
      - 字符串以已知错误前缀开头（harness / 工具层的结构化错误回执）；
      - 或 shell 类结果含 `=== exit code: N ===` 且 N != 0（如 cd /testbed 失败）。
    注意：不修改工具原始返回文本（exec_shell 仍原样返回 stdout/stderr/exit code），
    仅在此处做判定，避免破坏 harness.py 等消费方对 `exit code:` 的解析。
    """
    if not isinstance(res, str):
        return False
    s = res.strip()
    if any(s.startswith(p) for p in _TOOL_FAILURE_PREFIXES):
        return True
    # shell 非 0 退出：cd /testbed && pytest 这类命令链 `&&` 短路，pytest 根本没跑，
    # 模型收到错误后能改回相对路径重跑；若计入重复计数会被提前截断。
    if "=== exit code: " in s and "exit code: 0 ===" not in s:
        return True
    return False


def _split_messages(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    """把 messages 拆成 (system, user_concat)；JSON 模式（planner）用。"""
    system = ""
    conv = list(messages)
    if conv and conv[0].get("role") == "system":
        system = conv[0].get("content", "") or ""
        conv = conv[1:]
    user = "\n".join(
        m.get("content", "") if isinstance(m.get("content"), str) else str(m.get("content"))
        for m in conv
    )
    return system, user


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    """取 buffer 末条 user 消息的内容，作为 RAG 检索 query（prepare_messages 据此做知识回落）。"""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            return c if isinstance(c, str) else ""
    return ""



