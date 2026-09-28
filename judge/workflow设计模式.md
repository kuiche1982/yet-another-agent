# Agent‑Harness Judge‑Mode 同类校验模式清单
> 场景：产出物后置校验，结构化输出判决，失败回灌/重试，最多N轮；你当前 analyzer / tester 属于**事后二元评审Judge模式**。
> 不写代码，只做模式定义、触发条件、典型用法、和你的场景对比。

## 1. Judge‑Mode（你现在这套，事后评审校验）
- 逻辑：模块输出产物 → 交给独立Judge LLM → Structured Output(yes/no/notsure + reason) → 分支：
  - no：reason+原始上下文回灌重跑模块
  - notsure：调温度/top‑p，重试判定，上限N次
  - yes：放行
- 你的实例：analyzer校验(概念解释+需求说明)；tester校验(有效单元测试+有效实现，禁止空文件)
- 适用：单阶段产出物合规性检查；调研文档、测试代码、Skill输出。
- 弱点：Judge本身会幻觉；notsure重试会放大开销；只校验输出，不干涉生成过程。

## 2. Guard‑Rail 前置守卫模式（Pre‑Judge）
- 逻辑：**进入模块之前**做校验，不是产出之后。输入参数、上下文、约束条件先过Guard Judge。
- 流程：输入 → Guard‑LLM结构化判断输入是否合法完备 → 非法直接拒绝/补全输入，不进入analyzer/tester。
- 对比Judge‑Mode：Judge看输出；Guard看输入。
- 例子：调用代码生成前，判断“需求上下文是否足够，缺少则直接返回需要补充信息，不启动编码”。

## 3. Reflexion 反思修正模式
- 逻辑：生成 → Judge打分+给出具体缺陷列表 → 把**缺陷清单**回传给原Agent，Agent自主修改，而不是简单重跑。
- 和你的Judge区别：你只回灌reason+上下文；Reflexion要求Agent理解critique，针对性修复，不是无脑重跑。
- 典型：SWE‑Agent常用；打分不是yes/no，是结构化缺陷数组。

## 4. Self‑Consistency 多投票模式
- 逻辑：同一个Judge prompt跑N次，多份structured output投票。
- yes>半数 → 通过；no>半数 → 回灌；分散 → 提升参数再投票。
- 适配你的notsure场景替代方案，替代单模型重试。开销更高，降低Judge幻觉。

## 5. Critique‑Then‑Resolve 批判‑解决双Agent模式
- 两个独立角色：Critic（只挑错，不修复）、Resolver（拿到Critic全部问题去修复）。
- 流程：Generator产出 → Critic输出全部问题点 → Resolver基于原始产物+问题点修复。
- 和Judge‑Mode差异：Judge是yes/no三元判决；Critic输出完整问题清单，不做简单布尔结论。

## 6. Unit‑of‑Work Boundary Check（工作单元边界校验）
- Harness工程常用。按Skill/Goal切分工作单元，每个单元进入/退出都有校验契约(Schema/规则文本)。
- 进入：输入满足契约；退出：输出满足契约。不满足则该单元标记失败，抛出给上层Loop处理。
- 你的analyzer/tester可以包装成两个独立Work Unit，各自绑定一套退出Judge规则。

## 7. Plan‑and‑Verify 规划‑验证模式
- 先生成显式Plan结构化输出；Verify Judge校验Plan是否满足约束；Plan通过再执行；Plan不通过迭代修改Plan。
- 特点：校验发生在**执行之前**，校验Plan，不是最终产出。
- 例：Agent先输出编码计划，Judge校验计划是否覆盖需求，计划不合格不写代码。

## 8. Meta‑Judge 元校验模式
- Judge本身的输出也交给另一层Meta‑Judge校验，防止Judge幻觉错判。
- 流程：产物 → Judge输出判决+reason → Meta‑Judge校验：Judge的reason是否真的可以支撑result结论。
- 适用场景：对正确性要求极高，防止Judge乱给yes。成本翻倍。

## 9. ReAct‑like Loop‑Assert 循环断言模式
- 在Agent Loop每一步内部嵌入硬规则+LLM断言混合。一部分规则硬编码，一部分交给LLM Judge。
- 不是完整跑完整个模块再校验；每一步中间状态就做断言，失败立刻中断当前step。
- 区别Judge‑Mode：Judge是模块结束后；Loop‑Assert是step粒度校验。

## 10. Golden‑Case Reference Judge（基准案例比对校验）
- Judge不仅看规则文本，同时输入Golden Case基准样本做参照。
- Judge结构化输出：对比当前产出和Golden case是否满足同等质量标准。
- 适合你的机票Skill、调研文档场景；有标准样例库的时候效果优于纯文本规则Judge。

# 模式对比简表
|模式|校验时机|输出形式|失败处理|
|---|---|---|---|
|Judge‑Mode(你的方案)|模块输出后|yes/no/notsure + reason|回灌重跑，有限重试|
|Guard‑Rail|模块执行前|合法/非法|拒绝执行、补输入|
|Reflexion|输出后|缺陷列表|Agent针对性修复|
|Critique‑Then‑Resolve|输出后|完整问题清单|独立Resolver修复|
|Plan‑Verify|执行前(校验Plan)|Plan合格/不合格|迭代Plan，不执行|
|Self‑Consistency|输出后|多份Judge投票|投票聚合，减少幻觉|
|Meta‑Judge|Judge输出之后|Judge判决是否可信|推翻错误Judge结果|
|Loop‑Assert|每一步中间状态|step断言通过/失败|中断当前Step|
|Work‑Unit Check|WorkUnit进出|契约满足/不满足|单元失败向上抛异常|
|Golden‑Case Judge|输出后|和基准对比结果|回灌重跑|

# 对你当前架构的简短建议
1. 你现在 analyzer / tester 的实现，是标准 **Judge‑Mode**。notsure最多重试3次是工程上很合理的收敛限制，避免无限循环。
2. 风险点：Judge本身会幻觉。可选增强方向：
   - 可选1：叠加Meta‑Judge防止Judge乱判yes；成本上升
   - 可选2：换成Self‑Consistency投票替代notsure重试；
   - 可选3：把yes/no改成输出结构化缺陷数组，走Reflexion/Critique模式，而不是只回灌reason无脑重跑。
3. 不要把Judge做的太重，harness实践经验：Judge不要承担创造内容的责任，只做判决+指出哪里不对。

# 硬 vs 软的选择

# 硬Guard（代码层确定性校验） vs 软Judge‑Mode（LLM裁判）
> 业界全部模式都有落地；**标准工程范式：硬规则做底线，LLM‑Judge做上层质量评审，二者组合，不二选一**。

## 定义区分
- **硬Guard / Harness硬判断**：Python代码、Schema、状态机、正则、退出码、计数器、权限白名单。**不调用大模型，输入固定则结果固定，模型无法绕过**。
- **软Judge‑Mode**：调用LLM做评审（就是你analyzer/tester这套）。擅长语义、内容完整性；**会幻觉、结果非确定，有token/延迟开销**。

## 什么时候用【硬Guard】（必须优先写）
✅ 适合：**可精确、可形式化表达的条件，红线、资源、格式、流程状态**
1. 格式、文件存在性：不能是空文件、不能空测试文件、JSON Schema校验。
> 你的tester场景：`if len(output.strip()) ==0 → 直接失败`，这个**必须硬写，不要交给LLM judge去判断空文件**。
2. 流程状态机：阶段跳转锁，没跑完analyzer，不允许进入tester；最大重试次数上限（最多3次），循环熔断，防止无限轮次。
3. 资源&安全：最大token、最大迭代轮数、工具白黑名单、沙箱权限、成本上限。
4. 可量化客观指标：文件行数、必填字段是否存在、退出码、IATA码格式、时间范围。

> 关键点：**凡是能写成代码判断的，绝不丢给LLM**；硬校验作为第一道拦截，失败直接抛异常，不走LLM调用。

## 什么时候用【软Judge‑Mode】（LLM裁判）
✅ 适合：**语义层面，无法用代码精确描述的质量标准**
1. 内容语义完备性：调研文档「是否包含概念解释、是否包含需求说明」——语义判断，正则写不全，适合Judge。
2. 逻辑质量：单元测试是否**有效**（不是有没有写test函数，而是断言是否真的覆盖业务逻辑）；判断“测试只是空壳”很难纯代码搞定。
3. 主观/业务质量：论述是否完整、推理是否合理、方案是否匹配原始需求。

⚠️ 软Judge短板：会错判，会幻觉；同一个输入两次调用，结果可能不一样；增加延迟与token消耗；**不能用来做安全红线**。

## 你的 analyzer / tester 正确分层（工程落地）
1. **第一层：硬Guard（Harness）**
    - analyzer输出：文件非空、基础schema合法；不满足直接失败，不调用Judge。
    - tester输出：不能是空文件；有测试函数占位不等于有效测试，硬校验只能筛“完全空白”。
2. **第二层：LLM Judge‑Mode（你现在设计）**
    - 硬校验通过后，才调用Judge，判断语义：是否有概念解释、需求说明；单元测试是否真正有效。
    - 输出 `yes/no/notsure+reason`，最多3次重试（**最大重试次数本身是硬计数器，写在Harness，不能交给模型自己控制**）。
3. 失败分支：
    - `no`：reason+原始上下文回灌重跑子模块；
    - `notsure`：修改温度重试，硬计数器递减；计数耗尽直接标记任务失败，**不要无限重试**。

## 业界组合模式（漏斗式）
1. **硬规则前置过滤 → LLM‑Judge做质量评审 → Meta‑Judge（可选）复核Judge的reason是否站得住脚**。
2. 硬管“能不能跑、格式对不对、不许无限循环”；LLM管“写得好不好、语义齐不齐”。

### 简明对照表
|维度|硬Guard|LLM Judge‑Mode|
|---|---|---|
|是否调用LLM|否|是|
|确定性|100%确定|概率性，存在幻觉|
|模型能否绕过|不能|有可能（prompt逃逸）|
|成本|几乎0|token+延迟开销|
|适合|格式、空文件、计数器、状态流转、权限、必填字段|语义完整性、逻辑质量、业务内容完备度|
|不适合|复杂语义、主观质量判断|安全红线、空文件检测、循环次数上限|

## 避坑要点（来自SWE‑Agent / Harness工程实践）
1. ❌ 不要把“判断文件不为空”交给LLM judge；这是典型硬guard职责。
2. ❌ 不要把最大重试轮次交给模型自己决定；轮次计数器是harness硬变量。
3. ✅ Judge只负责**语义质量**；硬校验先把低级错误拦截干净，减少无效LLM调用。
4. ✅ `notsure`重试有上限，耗尽直接标记任务失败，不要指望模型总能得出明确结论。
5. ✅ 高风险动作，**不能仅依赖LLM judge放行**，必须硬规则兜底。

# 软guard要不要开thinking模式

# Judge软判断是否开thinking（DeepSeek‑V4‑Flash / 带思考输出类模型）
> 这里的thinking模式 = 模型输出内部推理链（`...`）

## 核心结论
**Judge 模块：默认关闭 thinking；只在遇到 notsure 重试那一轮，可以选择性打开。**
不要所有judge调用都开thinking，有明确取舍。

### 1）关闭 thinking（推荐，绝大多数Judge调用）
适用：analyzer / tester 的常规judge判定
- 优点
  1. 输出稳定，更容易严格遵守 structured‑output JSON，减少模型把reason写到think块里、JSON输出残缺。
  2. token消耗更低、latency更小，Judge是高频旁路调用，成本很敏感。
  3. 你只需要最终判决 result + reason，不需要看模型内心推演；reason字段本身就是对外输出的理由。
- 风险：复杂语义边界案例，判错概率轻微上升；靠「notsure重试机制」补偿。

> 关键点：**reason字段就是你的对外可回灌理由，不要依赖``作为回灌上下文。** 如果开thinking，很容易出现：真正有效理由藏在think块，JSON内reason敷衍。

### 2）开启 thinking，仅用于：notsure重试轮次（第2、3次判定）
当上一轮返回 `notsure`，说明边界模糊、语义模棱两可。
- 打开thinking，提升模型深度分析能力，让它充分拆解产出物对照你的标准。
- 拿到结果后：**丢弃``内容，只取structured output里的result/reason做业务逻辑**。
- 即便开thinking，依然强制要求输出JSON schema；思考只是内部辅助，不能把业务逻辑放到think标签。

### 3）绝对不要做的两件事
1. ❌ 不要把 `` 片段直接回灌到 analyzer/tester 重跑上下文。think是模型内部草稿，经常幻觉、发散，不是可信critique。只用JSON的`reason`。
2. ❌ 不要永久打开thinking做Judge，会拉高token，并且结构化输出故障率上升。

## 和你的现有架构结合
```
第一轮judge：thinking=off
→ result=no：用reason回灌，重跑analyzer
→ result=yes：放行
→ result=notsure：计数器-1；打开thinking模式，发起第二轮judge
    → 还notsure：再-1；如果还有配额，维持thinking打开再跑一轮
    → 计数器耗尽 → 任务标记失败，不继续
```

## 对比参考
|场景|thinking模式|原因|
|---|---|---|
|普通Judge首轮|关闭|追求JSON稳定性、低开销|
|notsure重试轮次|开启|边界模糊，需要深度推理辅助判决|
|Reflexion / Critic批判Agent|可以常开|Critic本身就要输出详细批判论据|
|Guard硬校验|不涉及LLM|纯代码|

## 补充坑点（工程踩过）
部分思考型模型开启thinking之后，会优先填充think块，挤压JSON输出，出现截断、JSON不完整。
> 应对：即使开启thinking，依然强制约束 `response_format=json`；并且给prompt明确约束：**所有对外批判理由必须写在JSON.reason字段，不要放在思考内部**。
