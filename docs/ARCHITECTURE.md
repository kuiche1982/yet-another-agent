# litertlm / swe_agent —— 架构总览（ARCHITECTURE）

> **代码是 source of truth。** 本文是整合架构参考，按 2026-09-13 代码现状整理；与
> `docs/harness_design_map.md`（L2 脑图 / 工作线索）互为补充，但凡本文与代码冲突，以代码为准。
> 所有 `file:line` 引用对应 `swe_agent/` 与根级 `contextmgr/` 包。
>
> 认知三层：`L1 设计原则 → L2 本文/脑图 → L3 代码（事实源）`。约束（阈值/路径/角色差异）
> 一律进 BUILD 层（`config.py` / `roles_config.py`），不塞进 LLM prompt。

---

## 0. 整体架构速览

```
                        ┌─────────────────────────────────────────────┐
                        │  supervisor.build_executor_agent(ctx)         │
                        │  装配三层嵌套 LoopConfig（l1→l2→l3）           │
                        └─────────────────────────────────────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼ L1 (attempt)           ▼ L2 (round)             ▼ L3 (step)
          on_iter_end=_l1_gate      on_iter_start=_l2_start     Agent._step()
          pytest+tester 闸门         on_iter_end=_l2_gate         单步 toolcall
                                     lint 闸门                    │
                 └────────────────────────┬────────────────────────┘
                                          ▼
                              Agent._run_loop(loop)
                               每步后 emit loop.hook_point
                               消费 HOOK_HUB.first_block_decision
                                 ├─ BREAK_LOOP → return "break_loop"
                                 └─ REJECT(stuck) →
                                      gate-less: return "stuck"
                                      有 on_iter_end: reason="stuck" 流入父闸门
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼ HOOK_HUB（统一总线，单例）                                  ▼ ToolRegistry.dispatch
  HookPoint: RUN_START / L1~L3_LOOP_START /                         BEFORE_TOOL_CALL →
            BEFORE/AFTER/ERROR/FINALLY_TOOL_CALL /                   内置 guard 经 HOOK_HUB
            VALIDATION_FAIL                                            REJECT/BREAK_LOOP
  订阅者：自描述 Guard 子类（写/读大小、stall、unsolvable）            AFTER_TOOL_CALL → OVERRIDE
            │                                                          global+工具级 hooks
            ▼
  guard.Guard（具名 Limit 注册表，reset_at=HookPoint 自声明重置）
                                          │
                            ┌─────────────┴──────────────┐
                            ▼ 扩展层（插件化，零改内核）   ▼ 感知层（代码，非 LLM）
              plugins(skills/.mcp/.lsp/agents)  mcp(LSP/MCP 服务器)
              skills(inline/fork)                harness sensors(LSP/pytest/import)
                            │
                            ▼ 上下文
              management.ContextManager（buffer+真相_log）
                 ↳ contextmgr 包（L0–L3 分级记忆 + RAG + 压缩）
```

---

## 1. 块清单（22 块）与归属

| # | 块 | 主文件 | 核心类/函数 |
|---|---|---|---|
| 1 | L1/L2/L3 loops | `supervisor.py` + `agent.py:_run_loop` | `build_executor_agent` / `LoopConfig` |
| 2 | agent class | `agent.py` | `Agent` / `LoopConfig` / `RoleConfig` / `RunState` / `AgentMode` |
| 3 | hooks | `hooks.py` | `HookHub`(单例 `HOOK_HUB`) / `HookPoint` / `GateAction` / `GateDecision` |
| 4 | guard / limits | `guard.py` + `guards.py` | `Limit` / `Guard` / `WriteSizeGuard`/`ReadSizeGuard`/`StallGuard`/`UnsolvableGuard` |
| 5 | context | `management.py:ContextManager` + 根级 `contextmgr/` 包 | `ContextManager` / `LayeredKB` |
| 6 | harness | `harness.py` | `detect_stack` / `_run_test_bar` / `run_lint` / `SensorRegistry` |
| 7 | mcp | `mcp.py` | `MCPClient` / `MCPHttpClient` / `MCPManager` |
| 8 | plugins | `plugins.py` | `load_plugins`（四路汇入扩展层） |
| 9 | skills | `skills.py` | `Skill` / `load_skills_from_dir` / `run_skill` |
| 10 | tools / registry | `tools.py` + `registry.py` | `ToolRegistry` / `ActionContext` / `tool` / 8 内置工具 |
| 11 | models / LLM 传输层 | `models.py` + `llm_glm.py` + `llm_lmstudio.py` + `model_swap.py` | `MODELS` / `role_model_id` / `judge` / `ModelManager` |
| 12 | roles | `roles.py` + `roles_config.py` | `make_role_config` / `RoleConfig` |
| 13 | compact / 上下文压缩 | `compact.py` | `maybe_auto_compact` / `compact_conversation` |
| 14 | config（BUILD 层） | `config.py` | 阈值/路径/模型 env 常量 |
| 15 | state | `state.py` | `GLOBAL_STATE` / 遥测 / loop nudge-replan |
| 16 | layers / 观测 | `layers.py` | `emit_event` / `layer_events` |
| 17 | management | `management.py` | `ModelManager` / `LayeredKB` / `local_search` |
| 18 | actions | `actions.py` | `parse_actions` / `_sanitize_response_actions` |
| 19 | verify / judge | `verify.py` | `run_tester` / `verify_gate` |
| 20 | contracts | `contracts.py` | `Action` / `Observation` / `Subtask` |
| 21 | lsp | `lsp.py` | `LSPClient` / `hint` |
| 22 | 入口/调试 | `cli.py` / `__main__.py` / `dbg.py` / `log.py` | 参数解析 / 调试 CLI / 日志 |

---

## 2. 块详述

### 2.1 L1/L2/L3 loops（三层嵌套 loop）

**文件**：`supervisor.py:1281` `build_executor_agent`；`agent.py:196` `_run_loop`。

- **装配**（`supervisor.py:1304-1310`）：三层 `LoopConfig` 嵌套
  - `l3 = LoopConfig(max_iter=MAX_STEPS, hook_point=L3_LOOP_START)`（单步 toolcall）
  - `l2 = LoopConfig(max_iter=MAX_ROUNDS, child=l3, hook_point=L2_LOOP_START, on_iter_start=_l2_start, on_iter_end=_l2_gate)`
  - `l1 = LoopConfig(max_iter=MAX_ATTEMPTS, child=l2, hook_point=L1_LOOP_START, on_iter_start=_l1_start, on_iter_end=_l1_gate)`
- **阈值**（BUILD 层，`config.py:293-295`）：`MAX_ATTEMPTS=3` / `MAX_ROUNDS=3` / `MAX_STEPS=8`，unattend 模式由 `--max-attempts/--max-rounds/--max-steps` 覆盖。
- **驱动**（`agent.py:_run_loop`）：`for it in range(1, loop.max_iter+1)` 内
  1. `guard.reset_at(loop.hook_point)` —— 按本层锚点清零对应限制；
  2. `reason = _run_loop(child) if child else _step()`；
  3. **post-step emit** `HOOK_HUB.emit(loop.hook_point, ctx=self.ctx)`，消费 `first_block_decision`（见 §2.3/§2.4）。
- **stall 传播语义（关键，2026-09-13 收口）**：REJECT 时
  - gate-less 内层 loop（`on_iter_end is None`）→ 立即 `return "stuck"`（阈值步即停，不空转 `max_iter`）；
  - 有 `on_iter_end` 的 loop → 置 `reason="stuck"` 落入 `loop.on_iter_end(self.ctx, reason)` 重校验（见 §2.6 闸门），**绝不 early-return 跳过父闸门**，否则外层 gate 永不触发、重试预算被误判终止。
- **emit 必须 post-step**：让本步 `tick` 的 stall 计数同一轮被检测到，保证「连续 N 次 → stuck」在精确第 N 步触发。

### 2.2 agent class（统一 Agent）

**文件**：`agent.py:96-179`（`RunState`/`LoopConfig`/`RoleConfig`/`AgentMode`）、`agent.py:163` `Agent`。

- **`Agent`**：六角色 + 三层 loop 都是它的实例。核心约束：run/`_run_loop`/`_step` 不直接感知 `ModelManager`（模型生命周期走回调/hook）；对话 buffer 与循环防护由 `RunState.cm` 统一持有。
- **`RoleConfig`**：角色差异（prompt / tools / mode / stop_actions / model_override）显式落在此，根治「副驾静默换模型」「tester 无模型」等 bug（`roles_config.py:34` `make_role_config`）。
- **`RunState`**（`agent.py:96`）：单次 run 跨迭代共享可变态的**单一真相源**——循环控制 + `metadata`（结果/标志，不放计数器）+ `cm`（buffer）+ `guard`（具名限制）。计数器一律进 `guard`，不散落。
- **`LoopConfig`**（`agent.py:120`）：`max_iter` / `child` / `on_iter_start` / `on_iter_end` / `hook_point` / `repeat_threshold`/`no_tool_threshold`（可逐角色放宽，如 analyzer）。
- **`AgentMode`**：`TEXT`(chat_text) / `JSON`(response_format) / `TOOLCALL`(chat_toolcalls) —— 与 `models.FCCapability` 配合决定「怎么调」。

### 2.3 hooks（统一 hook 总线）

**文件**：`hooks.py`。**单例 `HOOK_HUB`**（line 245）是全局总线；`HookHub` 是类。引用单例一律用 `HOOK_HUB`。

- **`HookPoint`**（line 45，枚举，取代魔法字符串 phase）：
  `RUN_START`(轮次边界全量重置) / `L1_LOOP_START` `L2_LOOP_START` `L3_LOOP_START`(各层迭代起始重置) / `BEFORE_TOOL_CALL` `AFTER_TOOL_CALL` `ERROR_TOOL_CALL` `FINALLY_TOOL_CALL`(单次工具调用前后) / `VALIDATION_FAIL`(supervisor 闸门 tick 后 emit)。
- **`GateAction`**（line 69）：`ALLOW` / `REJECT`(回灌 reason+loop 继续) / `BREAK_LOOP`(回灌 reason+终止 loop) / `OVERRIDE`(AFTER 阶段覆盖结果)。
- **`GateDecision`**（line 78）：`allow()/reject(reason)/break_loop(reason)/override(new_result)`。
- **`HookHub`**（line 126）：`on(point, fn)` 订阅、`emit(point, **fields)` 触发、`register(guard)` 按 `guard.hook_points` 挂自描述 guard；裁决辅助 `first_block`(返 reason str) / `first_block_decision`(返 `GateDecision` 本体，区分 REJECT vs BREAK_LOOP) / `override_of`。
- **fail-open**：单订阅者异常不影响主流程（`emit` line 173 try/except 记日志）。
- 旧式 hook 经 `_legacy_adapter`（line 227）零改动接入（`_PHASE_TO_POINT` 映射在 `registry.py:38`）。

### 2.4 guard / limits（自描述护栏范式）

**文件**：`guard.py`（具名限制）/ `guards.py`（行为护栏）。

- **`Limit`**（`guard.py:37`）：`name` / `reset_at:HookPoint`(重置时机自声明) / `threshold`。`Guard`（`guard.py:59`）是注册表+计数：`tick/value/set/reset_at/reset_all`。
- **`reset_at` 锚点约定**（`guard.py:136` `default_limits`）：
  - `consec_repeat` / `no_tool_streak` / `empty_streak` → `L2_LOOP_START`（每 round 清零；单层角色无 L2 → 整个 run 累计）
  - `bar_consec_fail` / `lint_consec_fail` / `drift_injections` → `RUN_START`（跨 attempt 累计，仅在 gate 判定成功时显式清零）
  - `tool_call_count` → `L3_LOOP_START`（每步清零）
- **自描述 `Guard` 子类**（`guards.py:29`）：`hook_points` / `applies_to` / `check(payload)→GateDecision|None`。注册只需 `HOOK_HUB.register(instance)` 或 `guards.register_all_builtin()`（幂等，先定向 clear 再注册，line 216）。
- **内置护栏**（全部走同一总线，无特例）：
  - `WriteSizeGuard`(`guards.py:52`，`BEFORE_TOOL_CALL`，仅 `write_file`)：超 `WEAK/STRONG_MAX_WRITE_LINES` → `REJECT`
  - `ReadSizeGuard`(`guards.py:83`，`BEFORE_TOOL_CALL`，仅 `read_file`)：超 `MAX_READ_LINES` → `REJECT`
  - `StallGuard`(`guards.py:124`，`L1/L2/L3_LOOP_START`)：连续重复/未调工具 → `REJECT`(stuck)
  - `UnsolvableGuard`(`guards.py:163`，`VALIDATION_FAIL`)：单杠/lint 连续失败达 `VAL_DOOMED_THRESHOLD` → 置 `unsolvable` + `BREAK_LOOP`
- **🔴 陷阱**：`Guard.value()` 对未知键静默返 0（`guard.py:87` `.get(name,0)`），而 `tick` 抛 `KeyError`；重命名 `Limit` 键后测试/调用点用旧名不会报错只会得 0。凡重命名键必须全量 grep `ctx.guard.value/tick/reset/set` + `tests/test_harness_contracts.py`。

### 2.5 context（对话 buffer + 分级记忆）

**文件**：`management.py:318` `ContextManager`（harness 接入层）+ 根级 `contextmgr/` 包（L0–L3 分级记忆引擎，深设计见 `contextmgr_design.md`）。

- **`ContextManager`**：对话 buffer 的**有状态拥有者**——持有 `_msgs`(工作集，发往模型) 与 `_truth`(真相日志，永远记原文，压缩不丢) ；`append` 同时写两者（`management.py:380`）；压缩是内部不变量（`compress_if_needed` 超阈值才调 backend）。
- **contextmgr 包**：`store/layered/index/persist/manager/prepare/compress/retrieve/inject/distill/symbols/embedder/tokenize/llm_backends/types`。L0 原始→L1 结构化→L2 可视化→L3 索引+回 L0 指针；RAG 引擎 `LayeredKB`（code/proj/global 三独立实例 + 磁盘增量缓存）。
- **会话隔离硬约束**：禁止 contextmgr 引入模块级可变单例；`RunState.cm` per-instance；配置调用期注入，不 import `config`。
- `Agent` 在 `RunState.cm is None` 时自建 `ContextManager()`（`agent.py:177`）。

### 2.6 harness（校验流水线 + 传感器 + 闸门）

**文件**：`harness.py`。

- **校验流水线**：`detect_stack`(workspace) → `_build_validation` → `_run_test_bar`(pytest 单杠) / `run_lint` → `SensorRegistry.run_pipeline`（`FileDriftSensor`/`ModuleImportSensor`/`UnitTestSensor`/`LspDiagnosticSensor`/`LanguageCliSensor`）。
- **单杠三件套**（缺任一不算完成）：① lint（`_l2_gate` 每 round 出 round 前）② 模型自生成 unittest（pytest，`_l1_gate`）③ tester 自然语言验收（`verify_gate`）。
- **L1 闸门 `_l1_gate`**（`supervisor.py:1184`）：pytest `no_tests`/`fail` → `tick("bar_consec_fail")` → 达 `VAL_DOOMED_THRESHOLD` 即 `emit(VALIDATION_FAIL)` 由 `UnsolvableGuard` 判 `BREAK_LOOP`；pass → `verify_gate` → `done`(pass/no_points/skipped) / `continue`(fail/error)。
- **L2 闸门 `_l2_gate`**（`supervisor.py:1121`）：`lint fail` → `tick("lint_consec_fail")` → 达阈 emit `VALIDATION_FAIL`；pass → `reset("lint_consec_fail")` 出 round 进 pytest。`model_error` → `early_stop` + `break`。
- **`VAL_DOOMED_THRESHOLD=2`**（`config.py:331`）：连续失败熔断，避免弱模型烧满预算在「当前模型无法逾越的墙」上。
- **缺失文件语义**：failed(执行了但失败) vs not-executed(没跑) 必须区分以定位 Agent 行为。

### 2.7 mcp（Model Context Protocol）

**文件**：`mcp.py`。**本模块是扩展层 MCP 真实后端**：supervisor 把 `mcp_tool`/`list_mcp_resources`/`read_mcp_resource` 路由到这里。

- `MCPClient`(stdio JSON-RPC) / `MCPHttpClient`(Streamable-HTTP) / `MCPManager`；读 `REPO_ROOT/mcp.json`，支持 stdio + Streamable-HTTP + `${VAR}` 展开。
- 取实例走 `get_mcp()`（懒连接，line 见 `mcp.py`）；HTTP 依赖 `requests`（缺则仅 stdio 可用）。
- 与 hooks 关系：`BEFORE_TOOL_CALL`/`AFTER_TOOL_CALL` 的 OVERRIDE/REJECT 语义同样适用 MCP 工具调用。

### 2.8 plugins（插件扩展层装配）

**文件**：`plugins.py`（对齐 claude-code plugin 机制）。

- `load_plugins(plugins_root, enable_mcp)` 把插件四路汇入真实后端：
  - `skills/` `commands/` → `skills.PLUGIN_SKILLS`（经 `reload_skills` 进 `ALL_SKILLS`）
  - `.mcp.json` → `mcp.PLUGIN_MCP_SERVERS`（仅 `enable_mcp=True` 连接）
  - `.lsp.json` → `lsp.PLUGIN_LSP_SERVERS`（按 languageId 索引，写文件回灌诊断）
  - `agents/*.md` → `supervisor.SUBAGENT_PROMPTS` / `SUBAGENT_ALLOWED`
- 为避免循环依赖，`supervisor/mcp/lsp` 仅在 `load_plugins()` 内懒导入（line 223-224）。

### 2.9 skills（技能系统）

**文件**：`skills.py`。

- `Skill` 数据类（name/description/content/when_to_use/allowed_tools/model/context:inline|fork/source）。
- 来源：磁盘 `REPO_ROOT/skills/<name>/SKILL.md`(claude-code 格式) + 内置(bundled) + 插件。
- `run_skill(name, args)`：inline 技能把指令注入对话上下文（模型接着执行）；fork 技能在子智能体独立运行。
- `skills_prompt_section()` 生成系统提示片段；`ALL_SKILLS` 为全集（含插件）。

### 2.10 tools / registry（自描述工具 + 派发 + 传感器）

**文件**：`tools.py` + `registry.py`。

- **`ToolDef`**(`registry.py:81`)：name/category/description/schema/run/dangerous/examples/when_to_use + 可拼装 hooks(before/after/on_success/on_fail)。`ToolRegistry.prompt_fragment()` 据声明动态生成工具清单提示词（实现与说明单点定义、永不同步）。
- **`tool` 装饰器**：工具函数在 `tools.py` 经 `@tool(...)` 注册（8 个内置：`read_file`@161 / `edit_file`@224 / `write_file`@335 / `exec_shell`@395 / `grep_files`@499 / `glob_files`@548 / `ask_user`@584 / `reload_plugins`@624）。
- **`dispatch`**(`registry.py:157`)：before 钩子(全局+工具级)→run→after 钩子→on_success/on_fail。**全局钩子收口到 `HOOK_HUB`**（按 `HookPoint` 锚点），工具级挂在 `ToolDef` 上。
- **路径安全**：`_safe_rel` 防 `../` 逃逸；`_sanitize` 把绝对 WORKSPACE 路径替换成相对 `./`（防弱模型抄绝对路径导致 cwd thrash）；`check_dangerous` 危险命令确认；`_hidden_test_guard` 隐藏测试路径保护。
- **传感器**(`registry.py`)：`BaseSensor`/`SensorRegistry`/`SensorFact`/`sensor()` 装饰器——客观校验层永远是代码，调度只消费结构化 Fact。

### 2.11 models / LLM 传输层

**文件**：`models.py` + `llm_glm.py` + `llm_lmstudio.py` + `model_swap.py`。

- **分层**：① `PROVIDERS`(zhipu/GLM 远程, lmstudio 本机, 仅 OpenAI 兼容) ② `MODELS` 注册表(每模型登记 `provider`/`fc`/`load_unload`/`context_length`，`fc` 全 `NATIVE_TOOLS`) ③ 角色只声明能力，运行时挑模型。
- **换模型 = 改 env**：`PLANNER_MODEL`/`ANALYZER_MODEL`/`EXECUTOR_MODEL`/`TESTER_MODEL`/`SIDECAR_COMPRESS_MODEL`（`ROLE_ENV` 在 `models.py:106`），不碰角色代码。
- **弱模型**：`WEAK_EXECUTOR_MODELS`(`models.py:37`) → `is_weak_executor()` 选精简 prompt + 更紧写入上限。
- `role_model_id`/`role_context_length`/`role_load_unload`/`judge`(含 `JUDGE_MODEL`/`ANALYZER_JUDGE_ROUNDS`) 归一化接入。
- **`model_swap.py`**：LM Studio 模型 load/unload（`ModelManager` 见 §2.17）+ `SidecarCompressSession`(LFM 压缩副驾，先压后自卸)。

### 2.12 roles（角色配置）

**文件**：`roles.py`(提示词/动作 schema 常量) + `roles_config.py`(`make_role_config`)。

- 六角色 = `Agent` 实例，差异仅在 `RoleConfig`(prompt+tools+mode+stop_actions+model_override)。
- planner(JSON, 无工具) / analyzer(TOOLCALL, 只读) / tester(TOOLCALL, 只读+`finish_verify`) / executor(TOOLCALL, 全量工具) / compact(TEXT, `model_override=SIDECAR_COMPRESS_MODEL`) / reviewer/researcher(见 `ROLE_TOOLS`)。
- **🔴 坑**：analyzer/tester 的 `model_override` 必须透传（否则升级路径静默失效 / tester 永 error）。

### 2.13 compact / 上下文压缩

**文件**：`compact.py`。

- `estimate_tokens`(粗略估算) / `get_auto_compact_threshold`(`CONTEXT_WINDOW - AUTOCOMPACT_BUFFER`) / `compact_conversation` / `maybe_auto_compact`。
- 摘要走标准 OpenAI 文本补全（`models.chat_text_messages`），两级压缩：model-free 一级(`contextmgr.compress`) + LFM 语义二级兜底(`compress_backend`)。
- `start_stdin_monitor`：stdin 手动触发压缩（`MANUAL_COMPACT_REQUESTED`）。

### 2.14 config（BUILD 层）

**文件**：`config.py`。**约束集中地**：阈值/路径/模型 env 默认值。

- loop 上限：`MAX_ATTEMPTS=3`/`MAX_ROUNDS=3`/`MAX_STEPS=8`(`config.py:293-295`)；`LOOP_REPEAT_THRESHOLD=3`(:317)；`MAX_ACTIONS_PER_RESPONSE=5`(:322)。
- 熔断：`VAL_DOOMED_THRESHOLD=2`(:331)；`DRIFT_MAX_INJECTIONS=3`(:336)。
- 写入上限：`WEAK_MAX_WRITE_LINES=500`/`STRONG=1000`/`WEAK_TEST=1000`/`STRONG_TEST=2000`(:357-360)；`MAX_READ_LINES=100`(:365)。
- 上下文：`CONTEXT_WINDOW=32768`/`AUTOCOMPACT_BUFFER=13000`(:403-404)；`MODEL_CONTEXT_LENGTH=96000`(:409)；`OUTPUT_BUDGET=6000`(:368)。
- 端点：`GLM_BASE_URL`/`LMSTUDIO_BASE_URL="http://localhost:1234/v1"`/`TESTER_MODEL=lfm2.5-2.6b`/`SIDECAR_COMPRESS_MODEL=lfm2.5-2.6b`。

### 2.15 state（全局运行态 + 遥测）

**文件**：`state.py`。

- `GLOBAL_STATE`(`state.py:63`)：任务/计划/完成列表等全局态；**唯一默认值来源** `_default_state()`（模块加载即初始化，REPL 路径不调 `reset_state`，故键须恒定存在避免 KeyError）；`reset_state()` 必须**原地**修改（clear+update）以保引用一致。
- 停滞/循环防护指纹（`_loop_nudge`/`_loop_replan`/`_turn_fingerprint`）、跨 run 记忆（`load_agent_memory`/`build_memory_note`）、遥测（`_telemetry`/`stats_summary`）。
- 路径/命令归一化小工具（`_safe_rel`/`_norm_cmd`）供 tools 与 supervisor 共用。

### 2.16 layers / 观测

**文件**：`layers.py`。

- 仅保留**跨层遥测** helper：`emit_event(layer, kind, detail)` / `layer_events()` / `reset_layer_events()`（供 supervisor 在嵌套循环边界打点）。循环防护已迁出（现由 `RunState.cm`/guard 持有）。

### 2.17 management（模型管理 + 分层 KB + 检索）

**文件**：`management.py`。

- `ModelManager`(`management.py:56`)：LM Studio 模型 load/unload（显存时序：先压后 load，出 round 即 unload）。
- `LayeredKB`(`management.py:211`，继承 `contextmgr.layered.LayeredKB`)：code/proj/global 三独立实例 + 磁盘增量缓存，首次构造增量构建、跨 run 复用。
- `ContextManager`(`management.py:318`，见 §2.5)；`local_search`(`management.py:292`)：跨 scope RAG 检索。
- `set_compress_backend`(`management.py:175`)：注入压缩后端（factory 模式，供本地 lmstudio 模型）。

### 2.18 actions（动作解析与净化）

**文件**：`actions.py`。

- `parse_actions(model_out)`：原生 toolcall 返回的 JSON 数组字符串 → 动作对象列表（最小容错，去 ```json 围栏）。
- `_sanitize_response_actions`：① 同轮去重（防 3B 模型吐数百相同 read_file）② 突发封顶至 `MAX_ACTIONS_PER_RESPONSE` ③ 只读主导预警（提示动手写码）。

### 2.19 verify / judge（独立验收 + 判定）

**文件**：`verify.py`。

- `run_tester`：独立只读 agent（tester 角色，仅 read_file/glob/grep/shell+finish_verify），利益隔离、证据约束防幻觉。
- `verify_gate`：消费 planner 生成的 `verify_points`，逐条核对引用证据；fail 才令 executor 重跑，skipped（读不到/工具缺失）不计入失败、不触发重跑。
- `models.judge`：独立 judge 模型（`JUDGE_MODEL`），空/占位→no，notsure 不升温 temp=0 重试×3 降级 no。

### 2.20 contracts（层间契约）

**文件**：`contracts.py`。

- `Action`(L1→L2 模型产出) / `Observation`(L2→L1/L3 执行观察) / `Subtask`(L3→L1 supervisor 派发)。各层只消费契约、不读另一层内部状态。

### 2.21 lsp（Language Server Protocol）

**文件**：`lsp.py`。

- `LSPClient`：stdio + JSON-RPC（Content-Length 分帧，区别于 MCP 逐行 JSON）。服务器由插件 `.lsp.json` 提供（同 MCP 由 `.mcp.json` 提供）。
- `hint(path)`：write_file/edit_file 的 after 钩子调用，把 LSP 诊断作为**非阻塞提示**回灌模型。

### 2.22 入口/调试

**文件**：`cli.py`(argparse 入口) / `__main__.py` / `dbg.py`(调试 CLI：`cmd_executor`/`cmd_planner`/`cmd_tester`/`cmd_mock_loop`/`cmd_e2e` + `install_logging`) / `log.py`(统一 logging，默认 `logs/`)。

---

## 3. 关键不变量与坑（落地前必读）

1. **stall 禁止 early-return 跳过父闸门**：REJECT → gate-less `return "stuck"`，有 `on_iter_end` 置 `reason="stuck"` 流入父闸门重校验（§2.1）。
2. **emit 必须 post-step**：让本步 tick 同轮被检测，保证「连续 N 次 → stuck」在精确第 N 步触发。
3. **`Guard.value()` 未知键静默返 0**：重命名 `Limit` 键须全量 grep（§2.4）。
4. **`reset_at` 自声明**：限制重置时机写在 `Limit` 声明处，由 `_run_loop` 按 `loop.hook_point` 统一执行，不靠人记。
5. **显存时序铁律**：先压缩 context（LFM 副驾，压完即自卸）→ 再 load 工作模型；绝不两模型同时驻留（`_l2_start`/`_l2_gate`）。
6. **Workspace 路径不泄漏**：prompt 与回灌一律相对路径；绝对路径经 `_sanitize` 转 `./`（§2.10）。
7. **fail-open**：hook 订阅者 / on_message 回调异常不阻断主链路。
8. **单例引用一致性**：`GLOBAL_STATE`/`HOOK_HUB` 必须原地修改或引用单例，禁止重新赋值掩盖旧对象。
9. **harness bug 走 GLOBAL 一次根治**；完成标准 = lint + `_run_test_bar` + tester `verify_gate` 三件套。
10. **工具注册/执行一致性**：whitelist `ROLE_TOOLS` 与 `roles.py` 执行路径须一致（analyzer 的 shell 工具 prepend `cd /testbed` 曾触发 pytest 失败，已修）。

---

## 4. 文件清单速查（职责）

- `agent.py`：统一 Agent + 三层 loop 驱动（`_run_loop`/`_step`）+ `RunState`/`LoopConfig`/`RoleConfig`
- `supervisor.py`：executor 三层 loop 装配 + `_l1_gate`/`_l2_gate` 校验闸门 + subtask 上下文
- `hooks.py` / `guard.py` / `guards.py`：hook 总线 + 具名限制 + 自描述护栏
- `harness.py`：校验流水线 + 传感器 + 单杠/lint
- `models.py` / `llm_glm.py` / `llm_lmstudio.py` / `model_swap.py`：模型目录 + 传输 + load/unload
- `roles.py` / `roles_config.py`：角色提示词 + 配置中心化
- `tools.py` / `registry.py`：内置工具 + 注册/派发/传感器
- `contextmgr/`（根级包）+ `management.py`：分级记忆引擎 + ContextManager + LayeredKB + ModelManager
- `compact.py` / `actions.py` / `verify.py` / `contracts.py`：压缩 / 动作净化 / 验收 / 契约
- `mcp.py` / `plugins.py` / `skills.py` / `lsp.py`：扩展层（MCP / 插件 / 技能 / LSP）
- `state.py` / `layers.py` / `config.py`：全局态 / 遥测 / BUILD 常量
- `cli.py` / `__main__.py` / `dbg.py` / `log.py`：入口 / 调试 / 日志
