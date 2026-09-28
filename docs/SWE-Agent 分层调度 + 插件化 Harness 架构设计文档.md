# SWE\-Agent 分层调度 \+ 插件化 Harness 架构设计文档

## 1\. 文档核心定位

本文档定义一套**工业级稳定、可扩展、低Token消耗、防模型幻觉**的 AI 软件开发 Agent 架构。

核心解决传统「全程大模型单打独斗」的三大顽疾：

- 模型自我欺骗（口头完成、实际架构漂移、残留垃圾文件）

- 自测自审幻觉（自己写实现、自己写错误测试用例）

- 无限重试烂地基（在错误规划上反复修补，Token爆炸、越改越乱）

整体范式：**轻量模型做总调度、强模型做高推理、小模型做体力编码、代码工具做客观兜底**。

## 2\. 整体架构总览（四层解耦）

架构严格分层、职责完全隔离，无交叉权责，彻底规避模型幻觉与能力错配问题。

### 2\.1 调度层（Supervisor）

**核心角色：LFM\-1\.2B / LFM\-2\.5 轻量调度模型**

定位：AI 项目 PM / 总指挥官，只负责流程调度，不做编码、不做业务推理、不碰物理文件。

核心职责：

- 状态管理：维护任务阶段（未规划/编码中/校验中/评审中/完成）

- 工具路由：调用三大 Worker 工具（规划/编码/评审）

- 故障决策：基于标准化 Fact 判定「重试编码」或「回退重规划」

### 2\.2 工作层（Worker 工具集群）

所有核心能力**封装为标准 Function\-Call 工具**，统一交由调度层调用，无硬编码流转。

- **tool\_planner（高阶强模型）**：输出结构化 JSON 架构契约（模块、文件、接口、依赖），负责动脑做设计

- **tool\_executor（轻量编码模型）**：严格按照 Plan 编码落地，只做体力搬砖，禁止私自修改架构、新增文件

- **tool\_reviewer（高阶强模型）**：基于标准化 Fact 做业务评审，只判逻辑对错，不做工程语法校验

### 2\.3 客观校验层（Harness 传感器流水线，核心底座）

**核心铁律：纯代码实现，零 LLM 参与，绝对客观，不可被调度模型跳过**

Executor 编码落盘后，**强制自动执行**，不接受模型“口头完成”的汇报，只认磁盘真实状态、工具返回码、静态检查结果。

核心能力：将原始工具日志 → 结构化机器可读 Fact，彻底解决 LLM 解读日志的幻觉问题。

### 2\.4 业务扩展层（插件化自定义校验）

支持用户自定义业务校验插件，适配不同技术栈、业务场景，不改动内核代码。

## 3\. 核心设计思想（全文最关键）

### 3\.1 彻底解耦：主观 LLM 决策 \& 客观机器事实

所有工程对错的判决权，**从模型主观自省，转移到机器客观事实**：

- 模型说“已删除旧文件”没用，磁盘扫描说了算

- 模型说“代码无问题”没用，静态检查、导入校验说了算

- 模型写的测试用例不算数，Golden 基线/不变量属性测试说了算

### 3\.2 能力分层，杜绝大模型滥用

- 贵的强模型：只做规划、评审、业务推理（低频调用，省Token）

- 便宜小模型：只做编码体力活（高频执行，降成本）

- 轻量调度模型：只做流程流转（极简推理，稳定高效）

- 代码工具：只做客观校验（零成本、零幻觉、百分百可靠）

### 3\.3 错误分级路由，杜绝无效重试

传统架构：所有报错全部丢给大模型重试，越修越崩、Token 爆炸。

新架构分级处理：

- 语法/单变量/简单 Bug → 重试 Executor（改实现即可）

- 架构漂移/导入崩溃/规划冲突 → 强制回退 Planner（推翻地基重规划，不烂修）

## 4\. 插件化 Harness 传感器详细设计

### 4\.1 核心抽象

统一抽象 `BaseSensor` 检测器接口，所有内置校验、用户插件统一规范输出 `SensorFact` 结构化数据。

上层调度模型 **完全无需感知底层技术栈差异**，统一消费 Fact 做决策。

### 4\.2 预制基础通用传感器（全项目通用）

- **文件漂移检测器**：比对磁盘文件与 Plan 契约，拦截新增/冗余/幽灵文件，根治架构漂移

- **模块导入检测器**：批量校验模块可导入性，拦截落地即崩的代码

- **单元测试执行器**：获取测试退出码、通过率，不采信模型自测结果

### 4\.3 技术栈专属传感器（按需加载）

根据项目类型自动装配，适配多语言场景：

- Python：Pyright/Mypy 静态类型检查

- Golang：Go Vet/Go Build 编译校验

- Web/前端：ESLint 语法规范校验

### 4\.4 用户自定义插件（业务层扩展）

支持用户以插件形式注入业务专属校验，不改动内核：

- Golden Case 基线比对：用人工标准答案校验模型输出

- 属性/不变量测试：如生命游戏「细胞数≥0、网格尺寸不变」

- 项目专属规范：目录规范、命名规范、接口契约校验

### 4\.5 执行强约束（不可突破的红线）

- 传感器流水线**代码强制自动执行**，LFM 调度模型无权跳过、无权关闭

- 所有校验结果结构化存入 State，作为唯一调度依据

- 禁止原始日志、原始 stderr 输入调度模型，彻底杜绝解读幻觉

## 5\. 完整标准执行流水线

1. **调度层触发规划**：LFM 调用 tool\_planner，强模型输出结构化架构 Plan

2. **规划评审准入**：强模型 Reviewer 校验 Plan 合理性，不合格直接重规划

3. **调度层触发编码**：LFM 调用 tool\_executor，小模型严格按 Plan 编码落盘

4. **强制客观校验**：代码自动运行全量传感器流水线，生成标准化 Fact 集合

5. **智能故障路由**：LFM 基于 Fact 判断：重试编码 / 回退重规划 / 进入评审

6. **业务最终评审**：强模型 Reviewer 基于 Fact\+Plan 判业务逻辑对错

7. **任务完结**：全量校验通过，标记任务完成

## 6\. 新旧架构对比（核心优势）

### 6\.1 旧架构：全程单一大模型无 Harness

- 所有工作全靠高价大模型，Token 消耗极高

- 模型自策自审，存在天然幻觉，架构漂移、假成功频发

- 报错无脑重试，容易在错误规划上无限迭代

- 无统一校验标准，不可观测、不可统计、不可迭代优化

### 6\.2 新架构：LFM调度\+分层Worker\+插件化Harness

- 高低能力分层，大幅降低 Token 成本

- 机器客观事实兜底，彻底解决模型自我欺骗

- 错误分级路由，杜绝无效迭代，稳定性质变

- 插件化适配多技术栈、多业务场景，可无限扩展

- 全流程结构化、可观测、可量化、可迭代

## 7\. 核心红线规范（必须严格遵守）

1. **感知层永远是代码，绝不交给 LLM**：文件扫描、静态检查、测试执行、Fact 解析，全部代码实现

2. **调度模型只消费结构化 Fact，不读原始日志**

3. **Worker 各司其职，禁止越权**：Executor 不许改架构，Planner 不许写业务代码

4. **校验永不跳过**：编码完成必跑全量传感器，无例外

5. **测试不自产自销**：优先 Golden 基线/属性测试，杜绝模型自测自审

## 8\. 落地价值总结（一句话核心）

该架构不依赖模型无限变聪明、不依赖海量 Token 堆砌，**通过「流程分工\+客观机器校验\+插件化扩展」解决 AI 编码的固有幻觉天花板**，同时实现低成本、高稳定、可落地的工业化 AI 软件开发流水线。

> （注：部分内容可能由 AI 生成）



# SWE‑Agent 全景架构图（文本版全景 + 数据流 + 边界）
> 基于前面设计文档，包含：LFM‑1.2B调度、worker工具、Harness传感器流水线、LSP、插件、state状态、全部数据流、红线约束

## 分层全景（从上到下）
```
┌─────────────────────────────────────────────────────────────┐
│ 调度层 Supervisor                                            │
│ LFM‑1.2B / LFM‑2.5 轻量调度模型（PM角色）                     │
│ 输入：state[plan, facts, review_result]                       │
│ 输出：function‑call 调用哪个worker工具                        │
│ ❌禁止：读原始stderr、原始LSP报文、直接读写磁盘               │
└───────────────────────┬─────────────────────────────────────┘
                        │ function‑call工具调用
┌───────────────────────▼─────────────────────────────────────┐
│ Worker工具层（全部封装为Function‑call工具）                   │
│ ├─ tool_planner    【高阶强模型】输出结构化Plan契约          │
│ ├─ tool_executor   【编码小模型】按Plan写代码落盘到workspace │
│ └─ tool_reviewer   【高阶强模型】业务逻辑评审，输入plan+facts│
└───────────────────────┬─────────────────────────────────────┘
                        │ tool_executor执行完毕，代码强制触发，不经过LFM
┌───────────────────────▼─────────────────────────────────────┐
│ 客观校验层 Harness 传感器流水线（纯Python代码，0LLM）         │
│ 【内置基础传感器】                                            │
│  ├─ FileDriftSensor        磁盘vsPlan文件漂移检测             │
│  ├─ ModuleImportSensor    模块导入测试                       │
│  ├─ UnitTestSensor         运行单元测试，读取进程退出码       │
│  ├─ LspDiagnosticSensor    ← LSP(gopls/pyright‑langserver)   │
│  │     ▶内部：启动LSP服务→拿diagnostic→解析→输出SensorFact   │
│  └─ LanguageCliSensor      pyright/go vet/eslint cli备选     │
│                                                              │
│ 【用户插件扩展点(BaseSensor)】                                │
│  ├─ GoldenCaseOracleSensor  Golden用例基线比对                │
│  └─ CustomRuleSensor       自定义业务/目录/命名规范校验       │
│                                                              │
│ 统一输出：List[SensorFact] 标准化事实集合                     │
│ 所有原始日志/LSP原始报文 在本层消化，不向上透传                │
└───────────────────────┬─────────────────────────────────────┘
                        │ facts写入全局state
┌───────────────────────▼─────────────────────────────────────┐
│ State 状态存储器（LibSQL /内存）                              │
│ { task, plan:Plan, facts:List[SensorFact], review_result, retry_cnt }
│ 只存结构化对象；**不存原始stderr、原始LSP诊断报文**           │
└───────────────────────┬─────────────────────────────────────┘
                        │ state回传给调度层LFM‑1.2B，进入下一轮循环
┌───────────────────────▼─────────────────────────────────────┐
│ Workspace 沙箱磁盘目录                                        │
│ 代码文件、测试文件、golden参考文件                            │
└─────────────────────────────────────────────────────────────┘
```

## 📝完整闭环数据流（以康威生命游戏举例）
1. 用户输入需求 → 初始化state，送入**LFM‑1.2B调度器**
2. LFM决策：调用 `tool_planner`
3. tool_planner内部调用高阶大模型，产出`Plan`，存入state
4. LFM读取state，调用 `tool_reviewer` 校验plan是否合理；不合理回到tool_planner
5. LFM决策：调用 `tool_executor`
6. tool_executor调用编码模型，把代码写入沙箱workspace磁盘
7. ✅**代码强制触发Harness传感器流水线，LFM无权跳过**
    - FileDriftSensor扫描磁盘比对plan
    - LspDiagnosticSensor启动LSP服务，获取诊断，解析成SensorFact
    - ModuleImportSensor尝试导入模块
    - UnitTestSensor跑单元测试
    - 加载用户插件GoldenCaseOracleSensor做业务基线校验
8. 全部传感器输出转为`List[SensorFact]`，写入state
9. state回传给LFM‑1.2B，LFM基于facts做调度判断三选一：
    - 普通编码bug：重试tool_executor（有限次数）
    - 严重漂移/导入失败：回退tool_planner，重新做方案
    - facts全部ok：调用tool_reviewer做业务评审
10. tool_reviewer拿到plan+facts，输出业务评审结果写入state
11. 评审通过 → state标记done，任务结束；不通过则回退规划

## ⚠️全局红线（全景里必须守住）
1. **LSP永远在Harness传感器内部**
   - LSP服务生命周期由Sensor管理；原始LSP诊断报文**禁止向上透传到Worker/Supervisor**
   - LFM完全不知道底层是LSP还是cli，只消费SensorFact
2. **传感器流水线触发权在代码，不在LFM调度模型**
   executor写完代码，无条件跑全部注册的sensor；LFM不能选择关闭/跳过校验
3. **分层权责不能越界**
   - LFM‑1.2B：只派活，**不写代码、不做业务推理、不读原始日志**
   - executor：只按plan实现，禁止私自新增文件、修改接口契约
   - planner：只输出plan契约，不写实现代码
   - reviewer：只做业务语义评审，不做语法/文件检查（交给harness）
4. **state只存结构化对象，不存原始大文本日志，规避模型解读幻觉**

## 关键组件对照表
|组件|所属层|是否LLM|核心作用|
|---|---|---|---|
|LFM‑1.2B|调度层|✅轻量LLM|任务分发、状态流转决策|
|tool_planner|Worker层|✅高阶LLM|输出结构化Plan|
|tool_executor|Worker层|✅编码模型|按Plan生成代码落盘|
|tool_reviewer|Worker层|✅高阶LLM|业务逻辑评审|
|LspDiagnosticSensor|Harness传感器层|❌纯代码|调用LSP，转换诊断为Fact|
|FileDriftSensor|Harness传感器层|❌纯代码|检测文件架构漂移|
|GoldenCase插件|Harness‑插件扩展|❌纯代码|业务基线校验|
|State存储|状态层|❌纯存储|保存plan/facts，循环上下文|
|Workspace磁盘|沙箱环境|❌文件系统|存放生成代码|

## 两种禁止的错误架构（全景避坑）
❌错误1：把LSP封装成Function‑call工具交给LFM去调用；LFM拿到原始LSP json自己解读
> 后果：回到模型读原始报文，产生幻觉，harness客观保障失效

❌错误2：LFM调度模型可以选择是否运行传感器流水线
> 后果：调度模型幻觉跳过校验，直接判定任务完成，出现大量“假成功”

# SWE‑Agent 全景架构 — 详细数据流说明
> 整体架构分层：调度层(Supervisor‑LFM‑1.2B) → Worker工具层 → Harness传感器流水线(纯代码) → State存储 → Workspace沙箱磁盘。
> 核心原则：**原始底层诊断报文（LSP、stderr）仅存在Harness内部，向上只输出标准化`SensorFact`；传感器流水线由代码强制触发，调度模型不能跳过**。

## 核心数据对象定义（所有流转的数据结构）
1. **Task**：用户原始需求字符串
2. **Plan**：结构化规划契约，`modules[] / interfaces / description`，由`tool_planner`输出
3. **SensorFact**：单条校验结果
```json
{
  "sensor_name": "string",
  "ok": boolean,
  "message": "人类可读简短描述",
  "payload": {} // 结构化负载：错误数量、漂移文件列表、测试通过率等
}
```
4. **State**：全局任务状态，存放于LibSQL/内存，是调度层唯一输入源
```json
{
  "task": "用户需求",
  "plan": Plan|null,
  "facts": [SensorFact],
  "review_result": {ok:bool,comment:string}|null,
  "retry_cnt": {executor:0,planner:0},
  "done": false
}
```
> ⚠️State**禁止存储**：原始stderr、原始LSP diagnostic报文、完整工具输出日志。

5. **FunctionCall**：调度层输出的工具调用指令，用于调用worker工具。

---

# 完整数据流分步拆解（以康威生命游戏多模块任务为例）
## 阶段1：任务初始化
1. 用户提交任务：`实现多模块康威生命游戏，带单元测试`
2. 系统初始化`State`对象：task赋值，其余字段置空，retry计数器归零，done=false。
3. 将完整`State`送入**调度层LFM‑1.2B**。
> LFM输入只有结构化state，没有原始工具日志。

## 阶段2：调度层 → Worker层：执行规划 tool_planner
1. LFM‑1.2B读取state，判断：还没有plan，输出FunctionCall：调用`tool_planner(task)`。
2. 系统路由执行`tool_planner`工具：
   - 工具内部调用**高阶强模型**，输出结构化`Plan`对象。
3. 将`Plan`写入`state.plan`。
4. state更新完毕，再次回传给LFM调度器。

## 阶段3：调度层 → Worker层：plan评审 tool_reviewer
1. LFM读取state：已有plan，调用`tool_reviewer(plan=null, facts=null)`做方案评审。
2. `tool_reviewer`内部使用高阶强模型，只评审业务方案合理性，**不做语法、文件检查**。
3. 评审结果写入`state.review_result`。
    - 如果评审不通过：state重置plan，retry_cnt.planner +=1；回到阶段2重新生成plan。
    - 如果评审通过：继续向下流转。

## 阶段4：调度层 → Worker层：编码执行 tool_executor
1. LFM读取state，决策调用`tool_executor(plan)`。
2. `tool_executor`内部调用编码小模型，严格遵守plan契约，生成代码文本。
3. 工具把代码**写入沙箱Workspace磁盘**。
> ⚠️重点：**executor执行结束，不返回给LFM；直接由代码强制触发Harness传感器流水线，这个步骤绕过LFM调度，模型无权干预**。

## 阶段5：Workspace磁盘 → Harness传感器流水线（纯代码，无LLM）
输入：workspace沙箱目录路径 + 当前`Plan`对象。
流水线顺序执行全部注册的`BaseSensor`检测器：
1. `FileDriftSensor`：扫描磁盘目录，对比plan.modules，找出额外生成的漂移文件，输出`SensorFact`。
2. `ModuleImportSensor`：尝试import所有模块，捕获导入异常，输出`SensorFact`。
3. `LspDiagnosticSensor`
    - 内部启动对应语言LSP服务(gopls/pyright‑langserver)；
    - 推送workspace文件变更，向LSP请求diagnostics诊断列表；
    - **在Sensor内部完成过滤、收敛、统计**；原始LSP JSON报文不对外暴露；
    - 转换为统一格式`SensorFact(payload={error_count,warning_count})`。
4. `UnitTestSensor`：执行单元测试命令，读取进程退出码、测试统计，输出`SensorFact`。
5. 依次执行用户自定义插件，例如`GoldenCaseOracleSensor`，完成业务基线比对，输出`SensorFact`。

> 流水线输出：`List[SensorFact]`集合。
> 所有原始诊断、stderr全部留在传感器内部，向上只输出标准化fact。

## 阶段6：Harness输出回写到State
1. 将传感器输出的fact列表赋值给`state.facts`。
2. 重置state.review_result。
3. 更新executor重试计数器。
4. 完整state回传给调度层LFM‑1.2B。

## 阶段7：调度层基于facts做故障路由决策
LFM‑1.2B接收完整state，只读取结构化facts，执行三分支决策：
1. **普通编码bug（语法、简单静态报错，无架构漂移，无模块导入崩溃）**
    - 条件：facts中存在失败，但无严重级别失败项。
    - 动作：有限次数内重试`tool_executor`，回到阶段4。
2. **严重故障（文件漂移、核心模块导入失败）**
    - 条件：对应sensor的fact.ok=false，属于规划地基损坏。
    - 动作：清空state.plan，planner计数器+1；回到阶段2重新生成plan，**不在烂代码上反复修补**。
3. **全部facts.ok=true，工程校验全部通过**
    - 动作：调用`tool_reviewer(plan, facts)`进入业务语义评审。

## 阶段8：业务最终评审 tool_reviewer
1. `tool_reviewer`接收：plan + facts集合。
> reviewer同样只接收结构化fact，不接收原始LSP、原始stderr。
2. 高阶强模型做业务逻辑校验：算法逻辑、业务规则是否正确。
3. 评审结果写入state.review_result。
    - 评审不通过：回退到规划阶段；
    - 评审通过：设置`state.done=true`，任务结束。

## 阶段9：任务结束
输出workspace产物，任务闭环。

---

# 数据流关键边界与禁止流向（非常重要）
## ✅允许流向
1. Workspace磁盘 → Harness传感器内部读取文件；
2. Harness内部解析原始LSP/stderr → 转为`SensorFact` → 写入state → 送入LFM调度层；
3. LFM只消费state中结构化对象，输出FunctionCall调用worker。

## ❌严格禁止流向
1. ❌Workspace磁盘 / LSP原始诊断报文 → 直接流向LFM调度层；
> 不允许模型阅读原始诊断JSON，防止解读幻觉。
2. ❌LFM调度层发出调用指令，触发传感器流水线；
> 传感器必须executor完成后代码强制运行，模型不能决定跑不跑校验。
3. ❌tool_executor直接修改state.plan；
> executor只能落盘代码，不允许修改架构契约。
4. ❌State存储保存大段原始日志、原始LSP报文，造成上下文膨胀、引入未受控信息。

---

# 异常分支数据流举例
### 案例A：executor偷偷生成额外`game_logic.py`，文件漂移
1. executor写完代码落盘；强制跑Harness流水线。
2. `FileDriftSensor`检测到额外文件，产出fact：`ok:false,extra_files:["game_logic.py"]`。
3. facts写入state，交给LFM。
4. LFM读取fact识别严重故障，决策回退planner；清空plan，重新生成方案。
> 不会反复调用executor去修补已经漂移的代码。

### 案例B：仅仅是代码少一个逗号语法错误
1. LSP诊断sensor输出fact ok=false，但是无漂移、无导入崩溃。
2. LFM判定普通编码错误，有限次数重试executor，plan保持不变。

### 案例C：facts全部通过，但业务逻辑错误（生命游戏迭代规则写错）
1. Harness全部fact全部ok，工程层面没问题。
2. 交给`tool_reviewer`强模型做业务评审识别逻辑错误。
3. 评审不通过，回退planner重新规划。
> Harness传感器只解决工程问题，**不能解决业务算法正确性问题**。

---

# 数据流简明总结
> 1. 用户需求 → state → LFM调度；
> 2. LFM只输出工具调用，驱动planner/executor/reviewer；
> 3. executor落盘代码后，**绕过模型，代码强制跑Harness流水线**；
> 4. Harness吃掉所有底层原始诊断，向上只吐标准化fact；
> 5. fact写入state再交还给LFM做调度决策；
> 6. 业务正确性交给reviewer强模型；
>
> 本质：**把计算机客观现实，先做一层收敛过滤，再交给概率模型做决策，避免模型直接面对原始杂乱底层输出而产生幻觉。**
