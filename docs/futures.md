# futures.md — 统一生命周期总线（lifecycle-stage-bus）设计方向

> **状态**：设计方向文档，**未实施**。所有结论基于当前代码事实（`swe_agent/hooks.py`、`guards.py`、`supervisor.py`、`agent.py`）。代码是 source of truth；本文件若与代码冲突，以代码为准并回补本文件。
> **决策记录（2026-09-13 讨论定稿）**：
> 1. 下一步 = 落本设计方向文档（不动代码）。
> 2. planner/analyzer 建模 = **保留 Agent 实例，由 `PRE_*` stage handler 触发**（非 collapse 成 handler）。
> 3. handler 分类 = **分 `Guard` / `Stage` / `Gate` 多类**（控制 / 计算 / 强制判定，语义分离）。
> 4. `on_iter_end` = **彻底移除**，所有闸门走 hook bus。

---

## 1. 动机：当前 supervisor 是"硬塞"拓扑

`supervisor.py:1307-1311` 在组装 `LoopConfig` 时命令式注入 `on_iter_end=_l1_gate/_l2_gate`；`build_executor_agent` 之外还串起 planner / analyzer 独立 Agent。

问题在于：**verify（pytest/lint 单杠）、planner、analyzer 本质都是独立关切，跟 loop 拓扑没有天然关系**——它们只是"借" loop 的生命周期锚点附着。换 hook 化，仍是"挂在某处"，但：

- 从「supervisor 内命令式单槽回调」→「开放注册面 + 自描述 handler」，**新增验证/护栏/阶段零改 supervisor、零改 `agent._run_loop`**。
- 单一消费点（`agent._run_loop` 一处 `first_block_decision`）符合「GLOBAL 一次根治 / 常量进 BUILD」原则，根治散落回调 bug。

代价（已确认需接受）：正向推动（plan→analyze→execute→verify）从过程式显式顺序，打散为「注册集合 + `hook_points` + `order`」的隐式分布；可读性靠**具名 handler 类 + 显式 manifest** 恢复（见 §7）。

---

## 2. 核心模型

一条生命周期总线（hook bus），所有"在生命周期某点做事"的单元都建模为 **handler**，挂在 `HookPoint` 上，由 `HOOK_HUB` 分发，`agent._run_loop` 统一消费结果。

当前已具备：`HookPoint` 枚举、`HOOK_HUB` 单例、`GateDecision`/`GateAction`、`first_block_decision`（fail-open，无决策返 `None` → 放行，`hooks.py:194/:211`）。

handler 三分类（决策 3）：

| 类 | 角色 | 返回语义 |
|---|---|---|
| **Guard** | 控制/拦截 | `REJECT` / `BREAK_LOOP` / 静默 `None`（放行） |
| **Stage** | 计算/产出 | 跑 LLM 或副作用，写 `ctx`，返回 `ALLOW`（放行） |
| **Gate** | 强制判定 | `mandatory=True`，返回 `COMPLETE`(成功) / `BREAK_LOOP`(失败) / `None`；漏注册 = fail-closed 报错 |

`Guard`/`Gate` 可持久持有状态、跨多 `hook_point` 观察（现有 `StallGuard` `guards.py:133` 已挂 `(L1/L2/L3_LOOP_START)` 跨层累加，证明此能力默认可得）。

---

## 3. hook bus 扩展（决策 2/4 的依赖）

```python
# hooks.py — GateAction 增补
class GateAction(Enum):
    ALLOW="allow"
    REJECT="reject"
    BREAK_LOOP="break_loop"
    OVERRIDE="override"
    COMPLETE="complete"          # ← 新增：正向成功判定

@dataclass
class GateDecision:
    action: GateAction = ALLOW
    reason: str = ""
    scope: Optional[int] = None  # ← 新增：terminate/complete 的目标层级（1=最外 L1 … 3=最内 L3）

# 类方法（仿 hooks.py:94-99 的 reject/break_loop）
@classmethod
def complete(cls, reason="", scope=None): ...
@classmethod
def terminate(cls, level: int, reason=""): ...   # BREAK_LOOP + scope
```

- `COMPLETE` 解决当前 hook bus 只含负向动作、无法表达 `_l1_gate` 返回 `"done"` 的缺口（`supervisor.py:1184`）。
- `scope` 解决跨层 takeover（见 §5）。
- `mandatory` 标志（在 handler 类上声明）让强制 gate fail-closed，避免开放注册面漏注册静默错终止。

---

## 4. hook points 增补（planner/analyzer 挂钩）

```python
class HookPoint(Enum):
    RUN_START="run_start"
    PRE_PLANNER="pre_planner"            # 新增
    PRE_ANALYZER="pre_analyzer"          # 新增
    PRE_EXECUTOR_START="pre_executor_start"  # 新增
    L1_LOOP_START="l1_loop_start"
    L2_LOOP_START="l2_loop_start"
    L3_LOOP_START="l3_loop_start"
    BEFORE_TOOL_CALL="before_tool_call"
    AFTER_TOOL_CALL="after_tool_call"
    ERROR_TOOL_CALL="error_tool_call"
    FINALLY_TOOL_CALL="finally_tool_call"
    VALIDATION_FAIL="validation_fail"
```

层级约定（`agent.py:10-12` `executor=L1(L2(L3))`）：**L1 最外、L3 最内**。

---

## 5. 冒泡与终止语义（已确认：冒泡 + 父 gate 复检）

- `BREAK_LOOP` / `COMPLETE` 带 `scope=N` → 消费时一层层 `return "break_loop"` 直到当前 loop 层级 == N。
- **途经的父 gate 仍逐层重校验**（不短路）——保留"冒泡 + 安全网"语义；只是"停到哪层"由 guard 用 `scope` 显式声明，而非父 gate 隐式决定。
- 例：attempt 耗尽直接 `terminate(level=1)` 顶到 L1 全停，但经过 L2/L1 时仍跑对应 Gate 复检，不丢安全网。
- `REJECT`（stall/体积护栏）→ 注入 reason、跳过本步、loop 继续（默认）；gate-less 内层 loop 须 `return "stuck"` 立即停（`agent.py:226-236` 已固化此语义，迁移时不可破坏）。

---

## 6. planner/analyzer 建模（决策 2）

保留为 `Agent(RoleConfig)` 实例（OOP 角色模型不推翻）。`PRE_PLANNER` / `PRE_ANALYZER` 上挂 **Stage handler**，其作用只是**触发**对应 Agent 并写产物进 `ctx`：

```python
class PlanBeforeExecutor(Stage):
    hook_points = (HookPoint.PRE_PLANNER,); order = 10
    def run(self, payload):
        plan = run_planner_agent(payload.ctx)   # 现有 planner Agent 实例
        payload.ctx.cm.append("system", plan)
        return GateDecision.allow()

class AnalyzeBeforeExecutor(Stage):
    hook_points = (HookPoint.PRE_ANALYZER,); order = 20
    def run(self, payload):                     # 读 ctx 里的 plan
        analysis = run_analyzer_agent(payload.ctx)
        payload.ctx.cm.append("system", analysis)
        return GateDecision.allow()
```

> 命名 = 顺序声明：**名字编码真实顺序**（plan→analyze→executor），读类定义即知管线位置——这是隐式分布后恢复可读性的关键。

---

## 7. 注册方式 + 可读性恢复（决策 1 的落地形态）

- **guard / gate**：保留模块 import 副作用自注册（稳定、轻量，现有 `guards.py` 即此风格）。
- **stage（pipeline 阶段）**：用**显式 manifest** 注册，避免"实际跑了啥"依赖 import 副作用（比过程式更隐晦）。

```python
# supervisor.build_executor_agent 收敛为：
register_pipeline([
    PlanBeforeExecutor, AnalyzeBeforeExecutor,   # 正向推动 Stage
    ExecutorLoop,                                 # 三层 loop 骨架（仍是 Agent._run_loop）
    PytestBarGate, LintGate,                      # 正向判定 Gate（mandatory）
    StallGuard, UnsolvableGuard, WriteSizeGuard,  # 负向护栏 Guard
])
```

薄 supervisor（只定义 hook 骨架 + 注册）+ 一处可读清单 = 两全。另可加 `HOOK_HUB.handlers_at(point)` 自省视图，保证与真实注册同步、不随注释过期。

---

## 8. 代价与缓解（诚实记录）

| 代价 | 缓解 |
|---|---|
| 全局顺序隐式（散在 N 个 handler） | 具名类 + manifest 一处清单 + 自省视图 |
| 开放注册面 fail-open（漏注册静默不拦） | 关键 Gate 标 `mandatory` → fail-closed |
| hook bus 契约变宽（控制+计算+排序+角色身份） | handler 分三类（Guard/Stage/Gate），语义边界在类型内而非外部 |
| verify 不属于 loop 拓扑的 smell 仍在 | hooks 化后从"supervisor 装配函数里"挪到"handler 的 `hook_points` 声明里"，耦合更局部、更自描述（概念别扭本身未消失，仅被隔离） |

---

## 9. 与当前代码 file:line 对照（实施时回填）

| 当前 | 目标 |
|---|---|
| `supervisor.py:1307-1311` `on_iter_end=_l1_gate/_l2_gate` | 删除；`_l1_gate`/`_l2_gate` 逻辑搬进 `PytestBarGate`/`LintGate` 的 `check()` |
| `supervisor.py:1184` `_l1_gate` 返回 `"done"/"continue"/"break"` | → `PytestBarGate` 返回 `COMPLETE` / `None` / `BREAK_LOOP` |
| `agent.py:226-236` `on_iter_end` 消费 | 删除；统一消费 `COMPLETE`/`BREAK_LOOP+scope` 冒泡 |
| `hooks.py:69-75` `GateAction` 仅 4 动作 | + `COMPLETE`；`GateDecision` + `scope` |
| planner/analyzer 独立 Agent 串接 | → `PRE_PLANNER`/`PRE_ANALYZER` Stage handler 触发 |

## 10. 未来实施步骤（分 PR，未开工）

1. **PR-A（最小 gate 合并）**：扩 `hooks.py`（`COMPLETE`+`scope`+`mandatory`）、新增 `PytestBarGate`/`LintGate`、删 `on_iter_end` 消费、迁移 `_l1_gate`/`_l2_gate`。验证：3 个 stall 测试 + `test_hooks.py` 加 `COMPLETE`/`scope`/`mandatory` 用例全绿。
2. **PR-B（handler 分三类）**：`Guard`/`Stage`/`Gate` 基类拆分 + `order` 字段 + `handlers_at` 自省。
3. **PR-C（pipeline stage 化）**：`PRE_PLANNER`/`PRE_ANALYZER` Stage + manifest 注册；planner/analyzer 改为被触发。
4. **文档回填**：本文件结论回写 `ARCHITECTURE.md` §hooks/§loops，并刷新 `harness_design_map.md`。

> 每 PR 实施前先出 diff 走 checkpoint（用户红线：改盘前确认方向）。

---

## 11. 额外的架构清晰度杠杆（设计模式映射）

> 讨论延伸（2026-09-13）：除主线 hook bus 统一外，下列模式可进一步澄清架构。**原则**：已存在的模式"命名即清晰"（零成本）；只有 FSM / Composite 两条值得新采用；勿叠加新框架（见 §11.3）。

### 11.1 已存在、命名即清晰（不新增代码，只在文档/注释点名）

| 模式 | 当前落点 | 清晰度收益 |
|---|---|---|
| **Registry + Adapter** | `registry.py:158 dispatch` + `:299 glm_tools`；`ROLE_TOOLS` 含 `mcp_tool`/`skill`/`lsp` | 工具/插件/mcp/skill 经 Adapter 统一进一个 `Tool` 接口——"4 条 seam"里最稳的一条 |
| **Mediator + Chain of Responsibility** | `HOOK_HUB`（`hooks.py`）+ `first_block_decision` 取首个拦截（`hooks.py:194/:211` fail-open） | hook bus = Mediator（handler 互不认识）+ CoR（同点 handler 链式、首个 block 胜） |
| **Strategy** | `RoleConfig`（`agent.py`）+ `roles_config.make_role_config` | 每个角色 = 同一 Agent 的不同 Strategy（prompt/tools/model） |
| **Specification** | `Guard.check`（`guards.py:52/133/163`） | Guard = 谓词+决策 的 Specification，自描述字段即规格 |
| **Template Method** | `_run_loop`（`agent.py`） | L1/L2/L3 共用同一驱动骨架，差异只在 `hook_point`+gate 配置 |
| **Ports & Adapters** | `contextmgr/` 包 vs `management.ContextManager` | harness 只依赖 `ContextManager` 端口，实现（store/index/compress）在 adapter 里 |

### 11.2 值得采用（主线未覆盖，建议纳入未来 PR）

**① 强类型 `LoopOutcome` 状态机**
当前 `_run_loop` 靠返回字符串传状态（`agent.py:226-236` 的 `"stuck"`/`"break_loop"`；`design_map §5` 终止契约词汇 `"done"/"continue"/"all_done"/"break"/"limit"/"model_error"`）。改枚举 + 转移表：

```python
class LoopOutcome(Enum):
    CONTINUE="continue"; DONE="done"; BREAK_LOOP="break_loop"
    STUCK="stuck"; MODEL_ERROR="model_error"

# 不变量可断言（编译期/测试期可抓，替代靠记忆的铁律）：
#   STUCK 只能来自 gate-less 内层 loop
#   DONE  只能来自 Gate.COMPLETE
#   BREAK_LOOP 必带 scope（terminate 目标层级）
```

收益：把"stall 禁止 early-return 跳过父闸门"等隐式铁律**变成可测试的转移约束**。

**② Composite loop tree**
`executor=L1(L2(L3))`（`agent.py:10-12`）现在是隐式嵌套调用。显式 `LoopNode` 树后：

```python
class LoopNode:
    def run(self, ctx) -> LoopOutcome: ...
    def terminate_to_depth(self, n, ctx) -> LoopOutcome:  # 冒泡 + 逐层父 gate 复检
        ...
```

`terminate(level=1)` = `node.terminate_to_depth(1)`，冒泡/父 gate 复检 = 树的遍历语义；takeover 不再是"返回值里藏编排意图"，而是树结构的自然结果。

### 11.3 红线：勿叠加新框架

hook bus 已是 **CoR + Observer + Mediator 三合一**。主线方向（扩 `COMPLETE`/`scope`/`mandatory`）是在其内部加动作，不是加新框架。**勿再套 Saga / Workflow 引擎**——会与你"单一机制、GLOBAL 根治"的偏好直接冲突。`Visitor` 收集 manifest、`Saga` 管补偿在此体量属过度工程，跳过。

---

## 12. 模式映射速查（给未来实施者）

- 想加新工具/插件/mcp → 写 Adapter 注册进 `ToolRegistry`（§11.1 Registry+Adapter）。
- 想加新护栏/判定 → 写 `Guard`/`Gate` 自描述类挂 `hook_point`（§2/§3）。
- 想加新生命周期阶段（如新 PRE_*）→ 写 `Stage` + manifest 登记（§6/§7）。
- 改 loop 终止语义 → 改 `LoopOutcome` 转移表 + `LoopNode.terminate_to_depth`（§11.2）。
- 任何"再加一层调度框架"的冲动 → 见 §11.3 红线。
