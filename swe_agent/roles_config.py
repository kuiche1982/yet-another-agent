#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/roles_config.py —— 角色配置中心化（重构：六角色 = Agent 的实例）

把五个角色（planner / analyzer / tester / executor / compact）的
「提示词 + 工具 + 交互模式 + 解析器 + 模型绑定」全部收进 RoleConfig，
由 make_role_config(role) 一把出实例。这是 OOP 重构的核心数据层：

  - prompt 与 tools 的差异，现在显式落在 RoleConfig 上（不再散落在 roles.py /
    verify.py / compact.py / supervisor.build_system_prompt 各处）。
  - 交互模式 AgentMode 直接对应 models.FCCapability 之外「怎么调」的维度：
      TEXT（压缩/摘要）→ chat_text_messages
      JSON（planner）→ chat_text_escalating(+response_format)
      TOOLCALL（executor/analyzer/tester）→ chat_toolcalls
  - 模型绑定（含 compact 副驾的 model_override）显式成字段，根治「副驾静默换模型」bug。

本模块只依赖 leaf 模块（config / models / roles / verify / compact / registry / agent），
不依赖 supervisor，避免循环导入。
"""

from typing import Optional

from . import config as C
from . import models as M
from . import roles as R
from .registry import ToolRegistry
from .agent import Agent, RoleConfig, AgentMode, LoopConfig, RunState


# ======================================================================
# 角色 → RoleConfig
# ======================================================================
def make_role_config(role: str, model_override: Optional[str] = None) -> RoleConfig:
    """按角色名产出 RoleConfig。model_override 用于升级子 agent 等场景。

    各角色差异（正是 OOP 重构要浮出来的「只有 prompt+tools 不同」）：
      planner   : JSON 模式，无工具，_PLANNER_SYSTEM
      analyzer  : TOOLCALL，只读工具集，_ANALYZER_SYSTEM，终止动作 finish_analysis
      tester    : TOOLCALL，只读工具集，_TESTER_SYSTEM，终止动作 finish_verify
      executor  : TOOLCALL，全量工具集，system_prompt 留空（由 supervisor 组装后注入 messages）
      compact   : TEXT 模式，model_override=SIDECAR_COMPRESS_MODEL（压缩副驾）
    """
    if role == "planner":
        return RoleConfig(
            name="planner", mode=AgentMode.JSON,
            system_prompt=R._PLANNER_SYSTEM,
            response_format=R._planner_response_format,
            require_json=True,
            model_override=model_override,
        )
    if role == "analyzer":
        return RoleConfig(
            name="analyzer", mode=AgentMode.TOOLCALL,
            system_prompt=R._ANALYZER_SYSTEM,
            tools=ToolRegistry.glm_tools("analyzer"),
            stop_actions=("finish_analysis",),
            # 必须透传：analyzer 的「弱模型产出不可用 → 升级强脑」靠 model_override 生效，
            # 丢掉它等于升级路径静默失效（日志说升级了、实际仍用原模型）。
            model_override=model_override,
        )
    if role == "tester":
        from . import verify as V  # 懒加载，避免与 verify 的顶层循环导入
        return RoleConfig(
            name="tester", mode=AgentMode.TOOLCALL,
            system_prompt=V._TESTER_SYSTEM,
            tools=ToolRegistry.glm_tools("tester"),
            stop_actions=("finish_verify",),
            # 必须透传：tester 不在 models.ROLE_ENV 里（没有 ANALYZER/PLANNER 那样的角色
            # env 绑定），模型只能靠 model_override（FORGE_TESTER_MODEL）解析。丢掉它
            # → chat_toolcalls 走 role_spec("tester") → None → 独立验收永远 error，
            # 单杠校验永远差第三件，任务只能耗到硬超时（2026-09-01 e2e 实测根因）。
            # 再兜底一层（对齐 compact 的 `model_override or C.SIDECAR_COMPRESS_MODEL`）：
            # 调用方忘了传也不会退化成无模型。
            model_override=model_override or C.TESTER_MODEL,
        )
    if role == "compact":
        from . import compact as CP  # 懒加载，避免与 compact 的顶层循环导入
        return RoleConfig(
            name="compact", mode=AgentMode.TEXT,
            system_prompt=CP.COMPACT_SYSTEM_PROMPT,
            model_override=model_override or C.SIDECAR_COMPRESS_MODEL,
        )
    if role == "executor":
        return RoleConfig(
            name="executor", mode=AgentMode.TOOLCALL,
            system_prompt="",  # TOOLCALL 模式不消费 system_prompt；由 supervisor 组装后注入 messages
            tools=ToolRegistry.glm_tools("executor"),
            stop_actions=("complete",),
            respect_stop=True,  # 尊重 finish_reason=stop；终止交 _l1_gate 单杠三件套判定（过则收尾）
            model_override=model_override,
        )
    if role == "chat":
        return RoleConfig(
            name="chat", mode=AgentMode.TOOLCALL,
            system_prompt=_CHAT_SYSTEM,
            # 全量工具：role=None → glm_tools 不裁剪，已注册工具一律可见（含插件动态注入的）
            tools=ToolRegistry.glm_tools(None),
            allow_all_tools=True,   # 跳过 ROLE_TOOLS 白名单，执行期放行全部已注册工具
            respect_stop=True,      # 尊重 finish_reason=stop（纯文本答复 = 本轮结束）
            stop_on_no_tool=True,  # 模型给纯文本即结束本轮对话（聊天语义）
            model_override=model_override or C.EXECUTOR_MODEL,
        )
    raise ValueError(f"未知角色：{role}")


# ChatAgent 的系统提示（REPL 交互模式专用）：全量工具 + 聊天语义
# _CHAT_SYSTEM = """你是一个运行在用户工作区里的交互式编程助手，拥有完整工具集（文件读写、shell、
# 搜索、规划、插件等）。

# 工作准则：
# - 用工具去查证事实（读文件、列目录、运行命令），不要凭空编造。
# - 需要多步操作时直接连续调用工具；当已经可以给用户一个完整的自然语言答复时，停止调用工具、用文字回答。
# - 不要输出「我无法…」之类推脱；尽力用工具把事情做完。
# - 用户用中文提问就用中文回答。"""
_CHAT_SYSTEM = """
你是运行在用户本地工作区内的交互式编程助手，具备完整工具集：文件读写、LSP、Shell执行、目录搜索、任务规划、插件调用。

## 核心工作准则
1. **事实优先，禁止臆造**：所有代码、配置、文件内容、项目现状必须通过工具查证（读文件、列目录、运行命令、检索源码），绝不凭空编造文件、接口、报错与项目上下文。
2. **工具调用策略**：可以在需要时调用工具。
3. **执行导向，拒绝推脱**：尽量利用现有工具完成用户诉求；不输出“我无法…”“做不到”这类推脱话术。遇到阻碍，优先通过Shell/文件工具排查问题，给出可行备选方案。
4. **语言对齐**：用户中文提问则输出中文；代码、报错、参数保留原始英文。
5. **修改前置校验**：写入、覆盖、删除文件前，优先读取目标文件确认现状；改动重要文件尽量输出变更diff，高危操作主动提醒风险。
6. **任务收敛规则**
   - 用户目标已达成 → 停止工具，交付最终结果
   - 多次工具尝试仍然无法推进 → 汇总已获取事实，说明卡点，给出可选路径，不再继续盲目调用工具
7. **输出规范**
   - 代码块完整可直接复制，不省略关键片段；
   - 结论基于工具返回的真实输出，区分【工具获取事实】和【你的推理】；
   - 不编造不存在的日志、堆栈、文件路径。

## 禁止行为
- 不要假设项目结构、依赖版本、已有代码；
- 不要在没有读取文件的前提下改写代码；
- 不要无限循环执行相同工具；
- 不要虚构命令执行结果。
"""


def _chat_loop_gate(ctx, reason: str) -> str:
    """ChatAgent 每轮收尾闸门：命中终态（纯文本答复 / 模型不可用 / 停滞）即结束本轮对话；
    工具调用类（continue）则继续下一轮（把观察结果回灌模型）。直接复用 Agent.run 的循环，
    避免 REPL 再手写一层 _step 循环。
    """
    if reason in ("all_done", "model_error", "stuck"):
        return "done"
    return "continue"


def make_chat_agent(model_override: Optional[str] = None) -> Agent:
    """REPL 交互模式专用 Agent 实例：全量工具 + 与执行器相同的 toolcall 执行路径，
    但带聊天语义（模型给纯文本答复即结束本轮）。打印由 REPL 自己的处理页负责，
    不在此处、也不影响 UNATTEND 的 run_agent 路径。

    循环用 chat_loop_gate：复用 Agent.run 的 L3 工具交换循环，遇 all_done/model_error/stuck
    自动收尾；无需 REPL 手写循环。
    """
    from .agent import Agent
    agent = make_agent("chat", loop=single_loop(max_iter=C.MAX_ITER, on_iter_end=_chat_loop_gate),
                       model_override=model_override)
    # 把聊天系统提示注入对话 buffer（Agent 自身不消费 system_prompt，由调用方注入，与 executor 同源）
    agent.ctx.cm.set_system(agent.role.system_prompt or "")
    return agent


# ======================================================================
# 循环配置：单步/单次 便捷构造
# ======================================================================
def single_loop(max_iter: int = 1, on_iter_end=None,
                repeat_threshold: int = 3, no_tool_threshold: int = 3) -> LoopConfig:
    """单步循环（planner/compact 单次产出；analyzer/tester 如需单次也可用）。

    on_iter_end 可选：自定义每轮收尾闸门（如 tester 命中 finish_verify 即终止）。
    repeat_threshold / no_tool_threshold：循环防护阈值（默认全局 LOOP_REPEAT_THRESHOLD），
    允许各角色单独放宽（如 analyzer 用更宽松值，配合失败不计次数让弱模型自愈）。
    """
    return LoopConfig(max_iter=max_iter, on_iter_end=on_iter_end,
                      repeat_threshold=repeat_threshold, no_tool_threshold=no_tool_threshold)


# ======================================================================
# 工厂：组装一个「条件 hook 已绑好」的 Agent
# ======================================================================
def make_agent(role: str, loop: Optional[LoopConfig] = None,
               model_override: Optional[str] = None,
               ctx: Optional[RunState] = None,
               before_step=None) -> Agent:
    """产出 Agent 实例，并按模型配置条件绑定横切 hook（load/unload / 压缩）。

    - 仅当角色模型 load_unload=True（本机 lmstudio）才把 ModelManager.load/unload
      绑进 pre_loop/post_loop；GLM/zhipu 不绑 → Agent.run 跑 hook 时列表为空 → 零开销。
    - before_step（ContextManager 压缩）由调用方按需传入（默认不传，避免每步都压）。
    """
    from .agent import Agent
    from .management import ModelManager, set_compress_backend, _default_compress_backend

    rc = make_role_config(role, model_override=model_override)
    lp = loop or single_loop()
    if before_step is not None:
        lp.before_step = before_step
    # 条件 hook：load_unload 为真才绑（走 models.model_load_unload 收口点，
    # 受 config.MODEL_LOAD_UNLOAD 全局开关控制，SWE_MODEL_LOAD_UNLOAD=0 时一律不绑）
    mid = rc.model_id()
    # 二级语义压缩兜底：仅本地 lmstudio 模型接 LFM compact；远程/无模型保持 None（纯 model-free）。
    # 具体压缩决策（何时升级到 LFM）由 ContextManager 按 hist_budget 自动判定。
    if (M.MODELS.get(mid) or {}).get("provider") == "lmstudio":
        set_compress_backend(_default_compress_backend)
    if M.model_load_unload(mid):
        lp.add_hook("pre_loop", ModelManager._load_hook(mid))
        lp.add_hook("post_loop", ModelManager._unload_hook(mid))
    return Agent(rc, lp, ctx=ctx)
