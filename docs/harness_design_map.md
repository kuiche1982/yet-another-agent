# litertlm / swe_agent —— 设计逻辑 & 调用脑图（参考用）

> **⚠️ 使用纪律（必读）**
> 本文件是**快速检索 / 参考**用的脑图，**不是代码真相源**。
> 动手（修 bug / 改逻辑 / 加功能）之前，**必须 re-read 实际代码验证本脑图的描述仍然正确**，
> 确认无误后再改。本文件可能过时，以实际源码为准。
> 结构有变化时**按需更新本文件**（不强制每次都更，但改了关键调用关系要补）。
> **发现本脑图与代码实际逻辑不一致时：以代码为唯一基准**，立即登记一个明确任务（Todo/任务系统）修复/更新本脑图并跟踪到完成——不默默带过，也绝不反向改代码迁就本脑图。
>
> **整合架构权威文档**：`docs/ARCHITECTURE.md`（2026-09-13 整理，全 22 块 + `file:line` 引用 + 关键不变量）。本脑图是它的 L2 速查补充；**三者冲突以「代码 > ARCHITECTURE.md > 本脑图」为准**。hook/guard 统一范式（#26–#45，2026-09-13 收口）的完整定义见 ARCHITECTURE.md §2.3/§2.4，本脑图 §0.5/§1/§5 仅补其落点。

---

## 知识分层（本项目认知体系的角色定位）

| 层 | 是什么 | 角色 | 是否真相源 |
|---|---|---|---|
| L1 设计文档 / 架构原则 | `MEMORY.md`「架构原则」等、本脑图中的「设计逻辑」段 | 定**大方向**（战略 / 契约） | 否 |
| L2 脑图（本文件） | `docs/harness_design_map.md` | **当前现状的索引**（导航 / 参考，快速检索） | 否 |
| L3 代码 | `swe_agent/*.py` 实际源码 | **当前现状的事实来源**（唯一真相） | **是** |

大方向靠 L1 锚定；现状到哪了靠 L2 找路；到底对不对永远回 L3 验证。L2 一旦与 L3 不符，以 L3 为准并登记任务修 L2。

## 0. 速查索引

| 你想找什么 | 去看哪节 |
|---|---|
| 整体架构 / 三层 loop | §1 |
| 某角色怎么跑、谁驱动 | §2 |
| 工具怎么注册/分发 | §3 |
| 模型接口返回什么 | §4 |
| 循环终止契约（stop_action） | §5 |
| hook 总线 / 自描述护栏（统一范式） | `ARCHITECTURE.md` §2.3/§2.4（本脑图 §0.5/§1/§5 补落点） |
| context（分级记忆 + RAG） | `ARCHITECTURE.md` §2.5 |
| mcp / plugins / skills / lsp 扩展层 | `ARCHITECTURE.md` §2.7/§2.8/§2.9/§2.21 |
| 怎么观测 / 自测 / 单测 | §6 |
| 已知坑 / 待裁定 | §7 |
| 关键文件职责 | 末尾「文件清单」 |

---

## 0.5 架构子模块心智模型（6 个，非「三层 loop」）

> ⚠️ **勿与「三层 loop」混淆**：下面的 6 个是**子模块（职责分块）**，不是循环层级。
> 「三层 loop」指 `Agent` 内部的 `L1(attempts)→L2(rounds)→L3(steps)`（见 §1），属于 `Agent cls` 子模块的内部机制。

把整个 harness 按职责切成 6 个子模块，由上（编排）到下（接入）：

| 子模块 | 职责 | 主要文件 | 现状 |
|---|---|---|---|
| **supervisor** | 组织 roles + harness，拼接端到端流程 | `supervisor.py`（`run_agent`/`build_executor_agent`/`_run_nested_loops`） | ✅ 编排已收敛，循环交给 Agent |
| **roles** | 自定义特性的 agent（per-role system/tools/hook） | `roles.py` + `roles_config.py`（`make_agent`/`single_loop`/`make_role_config`） | ✅ 六角色全走统一 `Agent`/`single_loop` + `on_iter_end` |
| **Agent cls** | 循环基础 + hooks（on_iter_end/pre_loop/post_loop/load/unload）+ 限制切入点 | `agent.py`（`_run_loop`/`_step`/`_apply_toolcall`） | ✅ 唯一循环驱动 |
| **hooks/guard** | 统一 hook 总线（生命周期锚点 HookPoint + 门控 GateAction/GateDecision）+ 自描述护栏（Limit 注册表 + Guard 子类） | `hooks.py`(`HOOK_HUB` 单例/`HookHub`/`HookPoint`/`GateDecision`) + `guard.py`(`Limit`/`Guard`) + `guards.py`(`WriteSizeGuard`/`ReadSizeGuard`/`StallGuard`/`UnsolvableGuard`) | ✅ **#26–#45 重构已收口（2026-09-13）：取代魔法字符串 `_global_hooks` + 内联 `return 'stuck'`/`ctx.metadata['unsolvable']`；所有护栏走同一总线，REJECT=回灌+继续 / BREAK_LOOP=回灌+终止** |
| **tool_registry** | 统一管理 function / plugins / mcp 带来的可用工具 + harness 限制（ROLE_TOOLS 白名单） | `registry.py`（`ToolRegistry`/`@tool`/`ROLE_TOOLS`/`dispatch`） | ✅ 双 fake 契约锁死 |
| **contextmgr** | 消息管理 / 压缩 / 知识库(RAG) 接入 | `management.py`（`ContextManager` + `RagEngine`/`_get_workspace_kb`） | ✅ 主链路已接电 |
| **providers / models** | 基础模型接入层，拉平模型使用方式 | `models.py`（`MODELS`/`PROVIDERS`/`role_provider`/`chat_toolcalls` 路由）+ `llm_lmstudio.py`/`llm_glm.py` | ⏳ **`providers` 独立成包：pending（仅记录，暂不动）** —— 拉平逻辑现嵌在 `models.py`，仅两个散装 `llm_*.py`，无 `providers/` 目录 |

> **`providers` 独立成包：pending（仅记录，暂不抽）**：用户心智里的 `providers` 独立子模块目前**不存在**——它与 `models` 融合在 `models.py`（`PROVIDERS` 字典 + `role_provider()` + `chat_toolcalls()` 按 provider transport 路由）+ 两个 `llm_*.py` 传输文件里。"拉平模型使用方式"的实际收口点是 `models.py:chat_toolcalls` 调度器。抽成 `providers/` 包是未来可选项，当前记为 pending，不影响功能（见 §7）。

> **双协议并存 ≠ 缺口**：六角色经 `Agent`（原生 tool_calls）统一；`supervisor.run_subagent` 走旧文本 JSON 动作协议（`chat_messages`+`parse_actions`）。这是**不同子模块各自合适的能力**，不是需要统一的缺口——`run_subagent` 保留旧协议是刻意设计，无动作。

> **`config`(C) 不是子模块/层，是 KV 配置源**：所有硬约束（`MAX_ACTIONS_PER_RESPONSE`、各角色阈值、`PARALLEL_TOOL_CALLS` 等）收在 `config.py`，本质是一份 KV（类比 xml/json/yml 配置文件），**最终应流入 `contextmgr` 与 `models`**——由这两者消费，而不是独立成层。上面的 6 个子模块**不含** config。

---

## 1. 整体架构：统一 Agent + 三层 loop

**核心抽象**（`swe_agent/agent.py`）：

```
Agent(RoleConfig, LoopConfig, RunState)
 ├─ RoleConfig  = 谁（system_prompt / tools / mode / stop_actions / model_override）
 ├─ LoopConfig  = 怎么跑（max_iter / child / on_iter_end / hooks）
 └─ RunState    = 跨迭代共享运行态（role / iteration / metadata / cm=ContextManager）
```

**三层 loop（executor 的本体）**：`L1(attempts) → L2(rounds) → L3(steps)`

- `agent._run_loop(loop)` 是唯一的循环驱动：
  - 若有 `child` → 递归跑子循环；否则跑 `_step()`（单步 toolcall 交换）。
  - 每轮收尾看 `on_iter_end(ctx, reason)`：
    - 返回 `"done"` → 整条链结束
    - 返回 `"break"` → 结束本层、向上传播 `reason`
    - 返回 `"continue"/None` → 带反馈重跑下一轮
- `_step()` 按 `RoleConfig.mode` 分流：
  - `TEXT` → `chat_text_messages`（压缩/摘要）
  - `JSON` → `chat_text_escalating(+response_format)`（planner）
  - `TOOLCALL` → `chat_toolcalls` → `_apply_toolcall`

**返回值语义**（贯穿整条链）：
`continue` / `done` / `all_done` / `break` / `limit`（跑满）/ `stuck`（循环防护）/ `model_error`（模型不可用）。

**🔴 hook/guard 统一驱动（#26–#45 收口，2026-09-13）**：上面的 `stuck` / `break_loop` 不再由内联 `return` 或 `ctx.metadata['unsolvable']` 产生，而由统一 hook 总线 `HOOK_HUB`（单例，`hooks.py`）驱动：
- `_run_loop` 在**每步之后（post-step）** emit 本层 `loop.hook_point`，统一消费 `first_block_decision`——放在 step 之后是为了让本步 `tick` 的 stall 计数同轮被检测到，保证「连续 N 次 → stuck」在精确第 N 步触发。
- `REJECT`（停滞护栏 `StallGuard`，订阅 L1/L2/L3_LOOP_START）→ 回灌 reason + loop 继续：gate-less 内层 loop 立即 `return "stuck"`（阈值步即停，不空转 `max_iter`）；有 `on_iter_end` 的 loop 置 `reason="stuck"` 流入父闸门重校验。**绝不 early-return 跳过父闸门**，否则外层 gate 永不触发、重试预算被误判终止。
- `BREAK_LOOP`（熔断护栏 `UnsolvableGuard`，订阅 `VALIDATION_FAIL`）→ 回灌 reason + 逐层终止当前 loop：由 supervisor `_l1_gate`/`_l2_gate` 在 `tick` 连续失败计数后 emit `VALIDATION_FAIL` 触发（单杠/lint 连续失败达 `VAL_DOOMED_THRESHOLD=2`）。
- 限制重置时机自声明：`Limit.reset_at=HookPoint`，由 `_run_loop` 按 `loop.hook_point` 自动清零（stall 三件套→L2_LOOP_START；bar/lint/drift→RUN_START；tool_call_count→L3_LOOP_START）。
- 完整定义与 `file:line` 见 `docs/ARCHITECTURE.md` §2.3/§2.4；新增护栏 = 写一个 `Guard` 子类 + `HOOK_HUB.register`，loop/dispatch/校验闸门零改动。

---

## 2. 角色（roles）与驱动方式

| 角色 | 模式 | 驱动方式 | stop_action | 终止是否安全 |
|---|---|---|---|---|
| **planner** | JSON | `make_agent("planner")` 单次 | — | 单次产出 |
| **analyzer** | TOOLCALL | `roles.run_analyzer()` → `RC.make_agent("analyzer")` + `single_loop(on_iter_end=analyzer_on_iter_end)`（2026-09-08 已改走统一 Agent，保留 R1 fallback 重跑 / R4 观察事实合成） | finish_analysis | ✅ `analyzer_on_iter_end` 闸门把 all_done→done、model_dead→break |
| **tester** | TOOLCALL | `verify.run_tester()` → `Agent` + `single_loop(on_iter_end=_tester_on_iter_end)` | finish_verify | ✅ 已修（`_tester_on_iter_end` 把 `all_done→done`） |
| **executor** | TOOLCALL | `supervisor.build_executor_agent()` → `Agent` 三层嵌套 | complete | ⚠️ 有界（外层 gate 兜住，见 §5） |
| **compact** | TEXT | `make_agent("compact")` 单次 | — | 单次产出（模型副驾） |

> **一致性缺口已消除（2026-09-08）**：`run_analyzer` 已改走 `make_agent` + `single_loop(on_iter_end=analyzer_on_iter_end)`，六角色全部收敛到统一 `Agent`，不再有手写 per-role 主循环。仅 `supervisor.run_subagent` 保留旧文本协议（见 §0.5）。

> **⚠️ 角色共享 Agent 的 finish_reason=stop 语义差异（2026-09-02 空转根因）**：executor 与 tester **共用 `_apply_toolcall`**，它对「无 content + 无 toolcall + finish_reason=stop」**统一按 `continue`（nudge 继续）**。但两者期望相反——executor 该 `continue`（继续干活），tester 该视为「没交卷」而**终止**。tester 被反复 nudge 空转、又被 supervisor 把旧代码的「未交卷→fail」当代码缺陷 → `_l1_gate` 回 `continue` 重跑 attempt → 弱 tester 又未交卷 → 死循环到 `MAX_ATTEMPTS=3`。**这就是 case 失败下 tester 不停重试的空转来源**。
> **修复收口在 run_tester/supervisor 层**（不打断统一 Agent 原则）：`_apply_toolcall` 仅在三个分支打标 `ctx.metadata["last_finish_reason"]`（"stop"/"tool_calls"）；`run_tester` 没拿到 `finish_verify`（`stop_result is None`）→ 一律 `("skipped", …)`；`_l1_gate` 见 `skipped` → 打印提醒并 `done`（退出、不重跑）。**下次别为「修 tester 空转」去给 `_apply_toolcall` 加角色特判**——那会破坏统一 Agent 抽象，且修不到 supervisor 视角的重跑回路。

**调用入口**：
- executor 主流程：`supervisor._run_nested_loops(messages, ctx)` → `build_executor_agent` → `agent.run(messages)`。
- tester：`verify.run_tester(model, max_iter)` → `make_role_config("tester", model_override=model)` → `Agent(rc, single_loop(on_iter_end=_tester_on_iter_end))` → 取 `ctx.metadata["stop_result"]` 归一化。
- analyzer：`roles.run_analyzer(user_task, max_steps)` → 直接 `M.chat_toolcalls`，手写 assistant/tool 交替消息，`finish_analysis` 命中即 `return` 摘要。

---

## 3. 工具系统（registry）

**注册**：`@tool(name=..., description=..., category=..., schema=..., ...)` 装饰函数 → `ToolRegistry.register(ToolDef)`。
- 注册名 `name` 与函数名可不同（`shell`→`exec_shell`、`grep`→`grep_files` 是合法的改名）。
- **历史 Bug A**：`@tool(name="write_file")` 被错挂到私有 helper `_write_verify`，导致 `write_file` 未注册、executor 写不出文件。检查要点：**公开工具的实现函数不能是 `_` 开头的私有 helper**（白名单除外）。

**分发** `ToolRegistry.dispatch(action, ctx)` 流水线：
`before 钩子(全局+工具)` → `run(自动按签名过滤参数)` → `after 钩子` → `on_success / on_fail`。

**角色工具集** `ROLE_TOOLS`（registry.py）：
- `executor`：write_file/edit_file/read_file/grep/glob/shell/ask/report/complete/verify/…+插件
- `tester`：`read_file/glob/grep/shell/finish_verify`（**严格只读**，BUILD 层硬控）
- `analyzer`：`read_file/grep/glob/web_search/finish_analysis`
- `planner`：`plan/todo_*/complete/ask/report/read_file/grep/glob/web_*/compact/…`

**双层校验**：
1. `glm_tools(role)` 生成 schema 时按 `ROLE_TOOLS[role]` 裁剪（首次调用惰性校验 orphan/unknown）。
2. `agent._apply_toolcall` 内再查 `ROLE_TOOLS_ALLOWED[role]`（= 同一份 `ROLE_TOOLS`），不在集内 → `error: 不允许调用` → continue。
**单轮多工具调用（2026-09-03 修复，GLOBAL 收口在 `agent._apply_toolcall`）**：
模型一轮可下发多个 tool_call（parallel tool calls），harness **全部顺序执行**
（只读/写入有先后语义，不做并发）。
- **旧行为（已废）**：只跑 `tcs[0]`，其余回「（已忽略：每轮只执行第一个工具调用）」假消息
  ——强模型（Ling 等）的一轮多步计划被砍成一步，是「模型聪明但 harness 拖后腿」的直接来源。
- **硬约束（BUILD 层 `config.py`，不进 prompt）**：
  - `MAX_ACTIONS_PER_RESPONSE`（默认 5）：单轮执行上限，超出**不执行但必须回一条说明**；
  - `OUTPUT_BUDGET_PER_TOOL_MIN`（1200）：单条结果预算 = `max(OUTPUT_BUDGET // 本轮调用数, 下限)`，
    防「一轮 N 个调用 × 每个 6000 字」撑爆弱模型上下文（单调用时等于 OUTPUT_BUDGET，行为不变）；
  - **每个 tool_call_id 都必须有一条 tool 结果消息**（被拒/被截断/被跳过也要回），
    否则 OpenAI 协议下一轮直接报 400；
  - 终止动作（`complete` / `finish_verify`）在轮中命中 → 同轮剩余调用不再执行，
    但同样各回一条说明；终止参数捕获进 `ctx.metadata["stop_result"]`；
  - 全轮一个都没真正执行（全越权/畸形）→ 计入 `no_tool_streak`（旧代码此处 continue 且 streak 归零，
    会让「反复下发非法工具名」空转到 max_iter）。
- **传输层配合**：`llm_lmstudio` 下发 `parallel_tool_calls = C.PARALLEL_TOOL_CALLS`（默认 True）。
  旧值硬编码 `False` = 主动要求服务端每轮只返回一个 tool call，是同类根因的传输层一半。
- 回归锁定：`test_harness_contracts.py` 4 个多工具契约测试 + `harness_selfcheck` 第 4 类源码级防回退闸门。

> 注：`verify.py` 里另有 `_TESTER_TOOLS` 常量，是冗余的二次声明，真闸门在 `ROLE_TOOLS`；别被它误导。

---

## 4. 模型层接口（models.py）

`chat_toolcalls(role, messages, tools=, model_override=, tool_choice=)` 返回**结构化 dict**：
- `{"type":"toolcalls","actions":[...],"tool_calls":[{"id","name","arguments"}]}` —— 正常
- `{"type":"content", ...}` —— 模型没走工具（只吐文本）
- `None` —— 调用失败 / 空响应（server 丢弃畸形 tool call）

`is_empty(meta)`：判定软空响应（`type=="empty"`，良性，交上层裁定）。

**单轮可返回多个 tool_call**：两个 provider 的解析层（`llm_glm.py` / `llm_lmstudio.py`）都已完整收集
全部 `tool_calls` 进 `meta["tool_calls"]`；能否被全部执行取决于消费端（`agent._apply_toolcall`，见 §3）。
`parallel_tool_calls` 由 `C.PARALLEL_TOOL_CALLS` 控制（默认 True；曾硬编码 False，见 §7）。

**退化响应边界（已修）**：`type=="toolcalls"` 但 `tool_calls:[]` 现在与「非 toolcalls」同属「本轮未调工具」，走 nudge + streak 兜底（`agent.py:236`）。**此前此处 `tcs[0]` 越界 `IndexError` 崩溃**，是根因级已修复的 bug。

`chat_text_escalating`（planner / JSON 模式）与 `chat_text_messages`（TEXT 模式）不返回 toolcall dict。

---

## 5. 终止契约（stop_action）—— 最易出 bug 的地方

> **两层终止机制并存（2026-09-13 澄清）**：
> - **stop_action 层（本节省）**：L3 单步命中 `stop_actions` → `_apply_toolcall` 返回 `all_done`，由 `on_iter_end` 闸门归一化（tester `finish_verify`→done / executor `complete`→外层 gate）。这是「模型主动交卷」路径。
> - **hook/guard 层（§1 已述 + `ARCHITECTURE.md` §2.3/§2.4）**：停滞（`stuck`，`StallGuard` via L1/L2/L3_LOOP_START emit）与熔断（`unsolvable`，`UnsolvableGuard` via `VALIDATION_FAIL`）由 `HOOK_HUB` 统一裁决，产出 `REJECT`(回灌+继续) / `BREAK_LOOP`(回灌+终止)。这是「护栏兜底」路径，与 stop_action 正交、互不替代。
> 两者最终都汇入 `on_iter_end` 闸门（`_l1_gate`/`_l2_gate`）做收敛判定；stuck 作为 reason 流入父闸门重校验（见 §1）。

- `_apply_toolcall` 命中 `stop_actions` 且 `_is_done(fn)` 为真 → 返回 `"all_done"`，并把参数捕获到 `ctx.metadata["stop_result"]`（tester 的 `finish_verify results` 由此上送）。
- `_run_loop` 收到 `all_done` 时：
  - **有 `on_iter_end`** → 由闸门决定（`_tester_on_iter_end` 返回 `"done"` 立即终止）。
  - **无 `on_iter_end`** → 被当 `"continue"` 重跑（**致命：弱模型反复重发 stop_action 直到 max_iter**）。
    - 监控：`agent.py` 对 `tester`/`analyzer` 触发 `logger.critical("loop_no_guard_for_stop_action")`；executor 则记 `loop_all_done_no_guard`（合法，外层 gate 消费）。

**三类角色处置（定位根因用）**：
1. **扁平循环无 on_iter_end**（tester 旧态）→ 灾难性空转 → 必须加 `on_iter_end`。
2. **嵌套循环有外层 gate**（executor）→ 有界 → 加回归测试锁住，别盲加 L3 `on_iter_end`（会改变嵌套交接）。
3. **手写非 Agent 循环**（analyzer）→ 不变量一致性缺口 → 标记待裁定。

**executor 终止链**：`complete` 在 L3 返回 `all_done` → L3 最多重抽 `MAX_STEPS` 次 → 外层 `_l2_gate`（lint）→ `_l1_gate`（pytest + tester）按测试结果判 `done`/`continue`。**有界，非无限空转**。

- **no-tool stop 分支（2026-09-03 配置化）**：`finish_reason=stop` 且无工具调用时，是否灌「请调用工具」nudge
  由 `RoleConfig.respect_stop` 决定（BUILD 层、实例化声明，**不写死角色名**，灭 OOP 反模式）：
  - **executor `respect_stop=True`**：尊重 stop，不逼调工具，交 `_l1_gate` 单杠判定（过则收尾、不过回灌失败继续修）。
  - **planner/tester/analyzer `respect_stop=False`**：不遵守 stop，逼出对应 stop tool；tester nudge 直呼 `finish_verify`。
  - 旧代码 `if role.name=="executor"` 写死角色名（已消除）。详见 `agent.py:_apply_toolcall` + `roles_config.make_role_config`。

**tester 验收结果三态 + 重跑闸门（2026-09-02）**：
- `finish_verify.results[].verdict` 三态：`pass` / `fail` / `skipped`。
- `run_tester` 归一化（`_normalize_results`）：verdict 非字面 `pass` 即 `fail`；显式 `skipped` 保留；**漏提交的验收点补判 `skipped`（不计失败、不触发重跑）**。
- harness 收敛只看 **`failed_case > 0`**：有失败用例 → `_l1_gate` 回 `continue` 重跑 attempt；`pass` / `skipped` / 漏提交（补 skipped）→ **不重跑**。
- **未提交 `finish_verify`**（模型 `finish_reason=stop` 不交卷 / 空转到 stuck / **max_iter 触顶**）→ `run_tester` 返回 `("skipped", …)`（非 `fail`）→ `_l1_gate` 直接 `done`。这是切断「tester 不停重试」死循环的关键：旧代码把未交卷当 `fail` 才引发重跑回路。
  - 三条 skipped 路径：`reason=="stuck"`（重复动作未收敛）/ `last_finish_reason=="stop"`（空响应不交卷）/ **max_iter 触顶**（`MAX_TESTER_ITER`）。三者语义一致：**tester 自身收敛失败 ≠ 代码缺陷**，都不重跑。契约由 `test_tester_maxiter_cap_degrades_to_skipped_not_fail` 锁死。
  - **2026-09-02 收口**：max_iter 触顶分支此前仍返回 `fail`（与另两条不一致），会让 supervisor 把「tester 没交卷」误判成代码缺陷而重跑 attempt——惩罚正确代码。现统一为 `skipped`，且 `MAX_TESTER_ITER=2` 由 BUILD 常量收口（`config.py`），test 用 `inspect.signature` 锁住两个入口默认值都取自该常量（防写死魔法数字）。
- `verify_gate` 在 `FORGE_TESTER_MODEL=off` 时也返 `("skipped", …)`，`_l1_gate` 同样 `done`（验收关闭=通过语义，仅确定性 pytest 生效）。
- **工具闸门（2026-09-02，收敛保护）**：`run_tester` 入口先 `_tester_tools_ready()` 校验 tester 只读工具（read_file/glob/grep/shell）是否都在 `ToolRegistry` 注册；**缺失直接返回 `("skipped", …)`，不进模型循环、不调一次 LLM**。guard 把「工具缺失」降级为 skipped，同时防住「空转」与「盲过」两种危害，避免 supervisor 把「工具缺失」误判成「代码缺陷」反复重跑 loop1。
  - **⚠️ 现状更正（2026-09-02 实测，推翻此前结论）**：`ToolRegistry.names()` 实测 = `ask/edit_file/finish_analysis/finish_verify/glob/grep/read_file/reload_plugins/shell/write_file`，**四个只读工具全部已注册**，`_tester_tools_ready()` 返回 `(True, [])`。**guard 不短路，tester 是真跑的**（会真调 LLM、真读代码）。此前脑图写的「只读工具未注册 → tester 永远 skipped、独立验收被旁路」是**过时描述**，已更正。guard 仅作防御保留。
- **提示词行为契约（2026-09-02，固化在 `_TESTER_SYSTEM` + `_render_task`）**：tester 必须**以事实为基础**——所有 verdict 都要指到实际读到的代码/输出，没读到的东西不判 pass。若某条验收点因【文件读不到 / 只读工具不可用 / 命令无有效输出】而**无法实际核对**，**该条 verdict 必须标 `skipped`，不得猜 pass/fail**，并在该条 `evidence` 写【20 字以内】简要原因（如「文件不可读，无法核对」），不展开。这是单条粒度（区别于上面的全局工具闸门整体 skipped）：已进模型循环、但单条文件读不到时的判定纪律。双 fake 测试 `test_render_task_hardens_...` / `test_tester_system_prompt_hardens_...` 锁死。

---

## 6. 观测层 / 自测 / 单测（动手前先跑这些）

**运行时监控** `swe_agent/log.py`（基于 Python 标准库 `logging`，取代旧的手搓 `trace.py`）：
- 级别开关：`logger.debug`(旧 `trace()`，默认不显示) / `logger.info`(正常进度) / `logger.warning`·`error`(调用失败/retry) / `logger.critical`(根因直报，永远显示)。
- 环境变量 `LOGLEVEL=DEBUG`(或向后兼容 `SWE_TRACE=1`) 启动即开详细追踪；运行期 `set_trace(True)` 即时开 debug，**无需重启**。
- 日志去向：控制台 `StreamHandler`(默认 INFO) + 文件 `logs/harness.log`(`FileHandler`，始终 DEBUG)。与 `logs/lmstudio_requests.jsonl`(I/O 抓包)、`logs/e2e_battery/`(电池日志) 同处 `logs/`。
- `critical()` 等价 `logger.critical`，已在 `agent.py`/`roles.py`/`verify.py` 等处替换；`trace()` 等价 `logger.debug`。
- **LM Studio 请求/回复落盘**：环境变量 `LMSTUDIO_DUMP=1`（默认关）把每次 LM Studio 调用的
  完整请求（messages/tools/max_tokens）与 server 回复（choices/tool_calls/content/usage/finish_reason）
  追加写到项目根 `logs/lmstudio_requests.jsonl`（按 ts 配成 request↔response 对）；调用失败/超时另落
  error 条。配套分析器 `scripts/analyze_lmstudio_dump.py` 直接读该文件，重点还原 error/response
  对应的完整输入与真实返回。glm（zhipu）传输层无此能力。2026-09-03 落地（用户要求请求+回复同开关、
  不加第二个变量）；已验证落盘为只读非破坏式（SDK 返回已解析的 `ChatCompletion` 对象，`_dump_response`
  仅 `getattr` 访问，不影响下游解析）。

**离线自检** `scripts/harness_selfcheck.py`：
- 导入期契约检查（write_file 注册且实现名正确、工具没挂到私有 helper、glm_tools 集合等）。
- 第 4 类（2026-09-03 新增）：**单轮多工具调用防回退闸门**——源码级检查 `_apply_toolcall`
  遍历全部 tool_calls 且不含旧的「只执行第一个」文案 + `MAX_ACTIONS_PER_RESPONSE` /
  `PARALLEL_TOOL_CALLS` 常量在位。
- `uv run python scripts/harness_selfcheck.py`。

**双 fake 契约单测** `tests/test_harness_contracts.py`（**model-free + tool-free**）：
- `ToolFreeRig` 把 `ToolRegistry.get` 换成 fake（工具只记录调用、零副作用）。
- `ScriptedModel` 把 `chat_toolcalls` 换成按「验证点脚本」返 toolcall，引导循环按流程点走完。
- **27 个测试**覆盖：tester 终止 / tester 流程顺序 / **tester 未提交 finish_verify→skipped** / **max_iter 触顶→skipped（非 fail，2026-09-02 新增）** / **`run_tester`/`verify_gate` 默认 max_iter 取自 `C.MAX_TESTER_ITER`（BUILD 常量收口）** / **含 fail 用例→fail（failed_case>0 重跑）** / **显式 skipped 不计失败→pass** / **漏提交点补 skipped→pass** / tester 只读工具未注册→skipped（guard 短路、零 LLM 调用；注：现状工具已注册，此测覆盖 guard 逻辑本身） / executor 有界终止 / role config / 空 tool_calls 不崩 / write_file 注册等。
- **单轮多工具调用 4 个（2026-09-03 新增）**：`test_single_turn_executes_every_tool_call`（一轮多调用全部执行 + 参数不串位 + 每个 id 都有回执）/ `test_turn_cap_truncated_but_every_id_answered`（超上限截断但仍回执）/ `test_stop_action_halts_remaining_calls_in_same_turn`（轮中终止动作后剩余不执行但仍回执 + stop_result 上送）/ `test_multi_call_turn_fingerprint_covers_whole_turn`（指纹覆盖整轮，防虚假 stuck）。
  - `ScriptedModel` 已扩展支持「一轮多调用」脚本（step 传 `[(name,args), ...]` 列表形态，单调用 tuple 形态向后兼容）。
- **负向实验纪律（改契约时必做）**：改完先跑绿 → 临时把被测分支改回错误行为 → 确认对应测试变红（证明检查器有效）→ 还原 → 再跑绿。本轮 max_iter cap 契约即用此法验证（`"skipped"→"fail"` 变异后 `test_tester_maxiter_cap_degrades_to_skipped_not_fail` 如期失败）。
- `uv run pytest tests/test_harness_contracts.py -p no:cacheprovider -q`。

> **铁律**：绝不用 e2e 当发现工具。修完流程 bug → 把契约编成离线检查 → **负向实验证明检查器能抓到** → 还原 → 全绿。

---

## 7. 已知坑 / 待裁定

| 项 | 状态 | 说明 |
|---|---|---|
| `@tool` 错挂私有 helper（Bug A） | 已修 + 回归测试锁死 | write_file 注册名==实现名 |
| tester 循环空转（Bug B） | 已修 + `_tester_on_iter_end` | all_done→done |
| 空 `tool_calls` 越界崩溃 | 已修（agent.py:236） | 走 nudge 兜底，非吞错 |
| tester 未提交 finish_verify 却 finish_reason=stop → 旧判 fail 致空转 | 已修（2026-09-02） | 未提交→`skipped`→`_l1_gate` done；根因=executor/tester 共用 `_apply_toolcall` 对 stop 统一 continue，语义按角色不同 |
| ~~tester 只读工具未注册~~ | **已推翻（2026-09-02 实测）** | 四个只读工具**均已注册**，`_tester_tools_ready()`=(True,[])，guard 不短路、tester 真跑。旧描述（「永远 skipped、验收被旁路」）已作废 |
| tester max_iter 触顶未交卷 → 旧判 `fail` | 已修（2026-09-02） | 统一为 `skipped`；`MAX_TESTER_ITER=2` 进 `config.py`，两个入口默认值由单测锁取自该常量 |
| analyzer 手写循环一致性缺口 | **已修（2026-09-08）** | `run_analyzer` 改走 `make_agent`+`single_loop`+`analyzer_on_iter_end`，六角色全统一到 Agent；仅 `run_subagent` 旧协议保留（见 §0.5） |
| analyzer/tester 产出靠长度阈值/启发式判定（F3 类） | **已修（2026-09-03 晚）** | 改用独立模型 `M.judge(kind, content)` 按显式标准评（analyzer: 概念解释+需求说明；tester: 有效单测+有效实现+非空），json_schema `{result:yes/no/notsure, reason}`。analyzer no_tool 分支调 judge，yes→采纳、no/notsure→回灌续轮（超 `ANALYZER_JUDGE_ROUNDS=3` 或不sure 耗尽 `JUDGE_MAX_RETRY=3` 降级→exhausted 且不采纳被拒散文）；tester `verify_gate` 先客观前置（非空文件+pytest 收集>0）不达标直接 fail，达标再 judge。硬 prompt 铁律不可省。取代原 F3 长度阈值 |
| executor/tester 共用 `_apply_toolcall` 对 finish_reason=stop 一律逼「调工具」，executor 真做完也被拖 | **已修（2026-09-03）** | `RoleConfig.respect_stop` 配置化：executor=True 尊重 stop 交 `_l1_gate` 单杠；tester=False 逼 `finish_verify` 且 nudge 直呼其名；灭 `agent.py` 写死角色名 |
| 单轮只执行第一个 tool_call（多工具被丢弃） | **已修（2026-09-03）** | 旧 `agent.py` 只跑 `tcs[0]`、其余回「已忽略」假消息，强模型一轮多步计划被砍成一步。现全部顺序执行 + 每个 id 必回执 + `MAX_ACTIONS_PER_RESPONSE` 收口；传输层 `parallel_tool_calls` 同步放开（旧值硬编码 False） |
| `_turn_fingerprint` 双实现漂移 | **已修（2026-09-03）** | agent.py 与 state.py 各存一份：agent 版对 grep/glob 只取 `path` 丢 `pattern`；state 版多动作分支只用动作名拼接。现统一收敛到 `state.py` 单一实现 |
| grep/glob 指纹丢 pattern（`a.get('path', a.get('pattern'))` 陷阱） | **已修（2026-09-03）** | dict.get 默认值只在键【不存在】时生效，而 path 几乎总存在（常为 "."）→ pattern 永远取不到 → 「同目录搜不同关键词」被判成同一动作 → 虚假 stuck |
| executor L3 冗余 `complete` 调用 | 有界，观察中 | 外层 gate 兜住；要彻底消除需单独验证 L3 on_iter_end |
| compact `model_override` 静默换模型 | **按设计，非缺陷（2026-09-08 确认）** | `model_override or C.SIDECAR_COMPRESS_MODEL` 是刻意行为，不视为坑；维持现状 |
| `providers` 子模块未独立成包 | pending（仅记录，暂不动） | 拉平逻辑嵌在 `models.py`（`PROVIDERS`+`role_provider`+`chat_toolcalls` 路由）+ `llm_*.py`，无 `providers/` 包（见 §0.5） |
| 双协议并存（`run_subagent` 旧文本 JSON） | **按设计，非缺口** | 不同子模块各自合适的能力；六角色走 Agent(tool_calls)，`run_subagent` 走 `chat_messages`+`parse_actions`，刻意保留，无动作（见 §0.5） |
| `ContextManager.to_list()` vs `prepare_messages` 入口 | 可尝试（须零回归） | agent 主循环入口是否改用 `prepare_messages` 仍在审议；可尝试切换，但**必须不影响现有功能**（需回归 `test_harness_contracts.py` + `test_agent_cm_contracts.py` 全绿） |

---

## 8. 文件清单（职责）

> 完整 22 块清单 + `file:line` 见 `docs/ARCHITECTURE.md` §1/§2。此处只列职责锚点。

- `swe_agent/agent.py` —— 统一 Agent / RoleConfig / LoopConfig / RunState / `_run_loop` / `_step` / `_apply_toolcall`（**单轮多工具调用 GLOBAL 收口点**）/ `_is_done`
- `swe_agent/hooks.py` —— **统一 hook 总线**：`HOOK_HUB`(单例) / `HookHub` / `HookPoint` / `GateAction` / `GateDecision`（#26–#45 落地）
- `swe_agent/guard.py` —— 具名限制 `Limit`(reset_at 自声明) / `Guard` 注册表+计数
- `swe_agent/guards.py` —— 自描述护栏 `WriteSizeGuard`/`ReadSizeGuard`/`StallGuard`/`UnsolvableGuard` + `register_all_builtin()`
- `swe_agent/state.py` —— GLOBAL_STATE + **`_turn_fingerprint` 单一实现** + 遥测/loop nudge-replan
- `swe_agent/roles_config.py` —— `make_role_config`（角色差异中心化）/ `single_loop` / `make_agent`
- `swe_agent/registry.py` —— `ToolRegistry` / `ToolDef` / `ROLE_TOOLS` / `@tool` / 传感器(`BaseSensor`/`SensorRegistry`) / `dispatch`
- `swe_agent/verify.py` —— `run_tester` / `verify_gate` / `_tester_on_iter_end` / `_normalize_results`
- `swe_agent/supervisor.py` —— `build_executor_agent` / `_run_nested_loops` / `_l1_*` `_l2_*` gates / `DEFAULT_TASK`
- `swe_agent/roles.py` —— 各角色 system prompt / `run_analyzer` / `_analyzer_collect_observations`
- `swe_agent/models.py` —— `PROVIDERS` / `MODELS` / `role_model_id` / `chat_toolcalls` / `judge`；`llm_glm.py` / `llm_lmstudio.py` 传输层；`model_swap.py` load/unload + 压缩副驾
- `swe_agent/management.py` —— `ContextManager`（对话 buffer + 真相日志）/ `ModelManager` / `LayeredKB` / `local_search`
- `contextmgr/`（根级包）—— 分级记忆引擎：store/layered/index/persist/manager/prepare/compress/retrieve/inject/distill/symbols/embedder/tokenize/llm_backends/types（深设计见 `contextmgr_design.md`）
- `swe_agent/compact.py` —— 对话压缩（autoCompact / summary）；`swe_agent/actions.py` —— `parse_actions` / `_sanitize_response_actions`
- `swe_agent/mcp.py` —— MCP 客户端/管理器（stdio + Streamable-HTTP）；`swe_agent/lsp.py` —— LSP 客户端（stdio JSON-RPC）
- `swe_agent/plugins.py` —— 插件扩展层装配 `load_plugins`（skills/.mcp/.lsp/agents 四路汇入）；`swe_agent/skills.py` —— `Skill` / `run_skill`
- `swe_agent/contracts.py` —— 层间契约 `Action` / `Observation` / `Subtask`
- `swe_agent/layers.py` —— 跨层遥测 `emit_event` / `layer_events`；`swe_agent/log.py` —— **统一 logging（取代旧 `trace.py`，2026-09-03 落地）**：`logger` / `critical` / `set_trace` 兼容别名
- `swe_agent/cli.py` / `__main__.py` —— 参数解析/入口；`swe_agent/dbg.py` —— 调试 CLI（`cmd_executor`/`cmd_planner`/`cmd_tester`/`cmd_mock_loop`/`cmd_e2e`）
- `swe_agent/config.py` —— BUILD 层 KV（loop 上限/熔断阈值/写入上限/上下文窗口/模型 env）
- `tests/test_harness_contracts.py` —— 双 fake 契约测试（model-free + tool-free）
- `scripts/harness_selfcheck.py` —— 导入期离线自检
