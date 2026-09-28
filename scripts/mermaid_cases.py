#!/usr/bin/env python3
"""mermaid DSL 能力评估数据集（固化，与执行器解耦）。

新增/修改 case 只动本文件，执行器 scripts/mermaid_probe.py 不改。
后续评估任何模型都复用同一批 case，保证横向可比。

每个 case 字段：
    id      唯一标识
    kind    read=理解（喂图问问题） / write=生成（喂需求出图） / edit=改图
    title   人类可读标题
    src     mermaid 源码（write 类为 None）
    ask     提问/指令
    checks  判分项，三种形态之一：
              {"name":..., "any":[...]}   任一子串命中（大小写不敏感、空白归一）
              {"name":..., "all":[...]}   全部子串命中
              {"name":..., "re":"..."}    正则命中（对归一化后的全文）
    gt      标准答案，供人工复核
"""

DATAFLOW = """flowchart LR
    RAW[原始日志] --> PARSER[解析器]
    PARSER --> QUEUE[(消息队列)]
    QUEUE --> WORKER[Worker]
    WORKER --> DB[(结果库)]
    WORKER --> CACHE[缓存]"""

FLOW_DECISION = """flowchart TD
    A[收到请求] --> B{已登录?}
    B -- 是 --> C[查权限]
    B -- 否 --> D[返回 401]
    C --> E{有权限?}
    E -- 是 --> F[处理请求]
    E -- 否 --> G[返回 403]
    F --> H[返回结果]"""

MODULES = """flowchart TB
    subgraph API[接口层]
        A1[Router]
        A2[Middleware]
    end
    subgraph CORE[核心层]
        C1[Planner]
        C2[Executor]
    end
    subgraph INFRA[基础设施层]
        I1[LLM Client]
        I2[Tools]
    end
    A1 --> A2 --> C1
    C1 --> C2
    C2 --> I1
    C2 --> I2"""

SEQUENCE = """sequenceDiagram
    participant U as 用户
    participant W as Web
    participant S as Server
    participant D as DB
    U->>W: 提交表单
    W->>S: POST /api/submit
    S->>D: INSERT
    D-->>S: OK
    S-->>W: 201 Created
    W-->>U: 显示成功
    Note over S,D: 事务在此提交"""

CLASSES = """classDiagram
    class Animal {
        +String name
        +eat()
    }
    class Dog {
        +bark()
    }
    Animal <|-- Dog
    class Owner {
        +String name
    }
    Owner "1" --> "*" Dog : owns"""

STATES = """stateDiagram-v2
    [*] --> Idle
    Idle --> Running : start
    Running --> Paused : pause
    Paused --> Running : resume
    Running --> Done : finish
    Running --> Failed : error
    Done --> [*]"""

TRAP = """flowchart LR
    A[start] --> B[process]
    B --> end
    end --> C[finish]"""

# 箭头写法在模型输出里很不稳定（--> / -> / → / =>），统一用这个字符类
_ARW = r"(?:-->|->>|->|→|=>|-\s*->)"

CASES = [
    # ==================== 理解（读图） ====================
    {
        "id": "r1_dataflow",
        "kind": "read",
        "title": "数据流图·节点与存储形状识别",
        "src": DATAFLOW,
        "ask": (
            "回答四个问题：\n"
            "(1) 图中共有几个节点？分别列出 id 和名称。\n"
            "(2) 从 RAW 到 DB 的完整路径是什么？\n"
            "(3) 哪些节点是数据库/存储形状（圆柱体 [( )]）？\n"
            "(4) 图中共有几条边？"
        ),
        "checks": [
            {"name": "含 CACHE 节点（易漏判别点）", "re": r"cache|缓存"},
            {"name": "路径 RAW→PARSER→QUEUE→WORKER→DB",
             "re": rf"raw\s*{_ARW}\s*parser\s*{_ARW}\s*queue\s*{_ARW}\s*worker\s*{_ARW}\s*db"},
            {"name": "存储节点=QUEUE 与 DB", "all": ["queue", "db"]},
            {"name": "边数=5", "re": r"(5|五)\s*[^\n]{0,6}条边|边[^\n]{0,8}(5|五)\s*条|共\s*(5|五)\s*条"},
        ],
        "gt": "节点6个(RAW/PARSER/QUEUE/WORKER/DB/CACHE)；路径 RAW→PARSER→QUEUE→WORKER→DB；"
              "圆柱存储=QUEUE、DB；边5条",
    },
    {
        "id": "r2_decision",
        "kind": "read",
        "title": "流程图·分支推理",
        "src": FLOW_DECISION,
        "ask": (
            "(1) 一个未登录用户的请求会依次经过哪些节点（用 id 表示）？\n"
            "(2) 哪些节点是判断节点（菱形 {}）？\n"
            "(3) 哪些是终止节点（没有任何出边）？"
        ),
        "checks": [
            {"name": "未登录路径 A→B→D",
             "re": rf"a\s*(?:{_ARW}|[,，、])\s*b\s*(?:{_ARW}|[,，、])\s*d\b"},
            {"name": "判断节点=B 与 E", "re": r"b\s*(?:和|与|、|,|，)\s*e\b|e\s*(?:和|与|、|,|，)\s*b\b"},
            {"name": "终止节点=D、G、H", "re": r"d\s*(?:、|,|，|和|与)\s*g\s*(?:、|,|，|和|与)\s*h\b"},
        ],
        "gt": "未登录 A→B→D；判断节点 B、E；终止 D、G、H",
    },
    {
        "id": "r3_modules",
        "kind": "read",
        "title": "模块图·层次划分与跨层依赖",
        "src": MODULES,
        "ask": (
            "(1) 图中有几个 subgraph（模块）？各叫什么，分别包含哪些节点 id？\n"
            "(2) 跨模块的边有哪几条？"
        ),
        "checks": [
            {"name": "模块数=3", "re": r"(3|三)[^\n]{0,8}个|共\s*(3|三)"},
            {"name": "识别 API / CORE / INFRA", "all": ["api", "core", "infra"]},
            {"name": "跨层边 A2→C1", "re": rf"`?a2`?[^\n]{{0,24}}{_ARW}[^\n]{{0,24}}`?c1\b"},
            {"name": "跨层边 C2→I1、C2→I2",
             "re": rf"`?c2`?[^\n]{{0,24}}{_ARW}[^\n]{{0,24}}`?i1\b"
                   rf"|`?c2`?[^\n]{{0,24}}{_ARW}[^\n]{{0,24}}`?i2\b"},
        ],
        "gt": "3 个模块：API={A1,A2}、CORE={C1,C2}、INFRA={I1,I2}；跨层边 A2→C1、C2→I1、C2→I2",
    },
    {
        "id": "r4_sequence",
        "kind": "read",
        "title": "时序图·参与者、消息序与箭头语义",
        "src": SEQUENCE,
        "ask": (
            "(1) 有几个参与者？\n(2) 一共有几条消息？\n"
            "(3) 第 3 条消息是谁发给谁的，内容是什么？\n"
            "(4) 虚线箭头 -->> 与实线箭头 ->> 在语义上有什么区别？"
        ),
        "checks": [
            {"name": "参与者=4", "re": r"(4|四)\s*[^\n]{0,6}参与者|参与者[^\n]{0,8}(4|四)"},
            {"name": "消息=6", "re": r"(6|六)\s*[^\n]{0,6}条消息|消息[^\n]{0,8}(6|六)"},
            {"name": "第3条是 Server→DB", "re": rf"\b(s|server)\s*({_ARW}|发给|发送给)\s*\b(d|db)\b"},
            {"name": "内容含 INSERT", "re": r"insert"},
            {"name": "-->> 表示响应/返回", "any": ["响应", "返回", "回复", "response", "reply", "回送"]},
        ],
        "gt": "4 参与者；6 条消息；第 3 条 S→D: INSERT；-->> 为响应/返回消息，->> 为请求/调用",
    },
    {
        "id": "r5_class",
        "kind": "read",
        "title": "类图·继承方向与多重性",
        "src": CLASSES,
        "ask": (
            "(1) 继承关系是谁继承谁？\n(2) Animal 有哪些成员？\n"
            "(3) Owner 与 Dog 之间是什么关系，多重性如何？"
        ),
        "checks": [
            {"name": "Dog 继承 Animal",
             "re": r"(dog|狗)[^\n]{0,20}(继承|子类|extends|inherits)|"
                   r"(继承|子类)[^\n]{0,20}(dog|狗)|animal[^\n]{0,15}(父类|基类|父)"},
            {"name": "Animal 成员 name / eat", "all": ["name", "eat"]},
            {"name": "Owner 1 对多 Dog",
             "any": ["一对多", "one to many", "1..*", "多重性", "multiplicity", "1 对多", "1对多"]},
        ],
        "gt": 'Dog 继承 Animal（<|--）；Animal={+String name, +eat()}；Owner "1" → "*" Dog 一对多关联 owns',
    },
    {
        "id": "r6_state",
        "kind": "read",
        "title": "状态图·初态/终态/死状态识别",
        "src": STATES,
        "ask": (
            "(1) 初始状态是哪个？\n(2) 哪些是终止状态？\n"
            "(3) Failed 状态有出边吗？这意味着什么？\n(4) 从 Paused 如何回到 Running？"
        ),
        "checks": [
            {"name": "初态=Idle", "re": r"idle"},
            {"name": "终态=Done", "re": r"done"},
            {"name": "Failed 无出边（死状态）",
             "re": r"failed[^\n]{0,40}(没有出边|无出边|死状态|无法继续|无法恢复|无法转移|不再转移|没有后续)"},
            {"name": "Paused --resume--> Running", "re": r"resume"},
        ],
        "gt": "初态 Idle；终态 Done；Failed 无出边（死状态，无法恢复）；Paused --resume--> Running",
    },
    {
        "id": "r7_trap",
        "kind": "read",
        "title": "语法陷阱·end 保留字",
        "src": TRAP,
        "ask": (
            "这段 mermaid 能正确渲染吗？如果有问题，指出具体是什么问题，"
            "并给出修改后的完整代码。"
        ),
        "checks": [
            {"name": "判定为有问题（渲染会失败）",
             "re": r"不能(正确)?渲染|无法渲染|不可渲染|不(能|可|会)(正确|渲染)|渲染(失败|错误|报错)|有(问题|毛病)|不正确|不合法|无效|冲突"},
            {"name": "定位到 end", "re": r"\bend\b"},
            {"name": "解释为保留字/关键字冲突",
             "any": ["保留字", "关键字", "保留词", "key word", "keyword", "闭合", "冲突", "解析"]},
            {"name": "给出修法（改名或加引号）",
             "any": ["改名", "重命名", "引号", "替换", "rename", "escape", '"end"', "'end'", "改为", "改成", "换成"]},
        ],
        "gt": "end 是 mermaid 保留字（用于闭合 subgraph），小写独立出现的 end 会破坏解析；"
              "修法：节点 id 改名为 E/finish_node，或加引号 \"end\"",
    },
    # ==================== 生成（写图） ====================
    {
        "id": "w1_flow",
        "kind": "write",
        "title": "生成·带双判断的注册流程",
        "src": None,
        "ask": (
            "用 mermaid flowchart 画出这个用户注册流程：用户提交表单 → 校验参数；"
            "校验失败则返回错误；校验通过 → 查重；已存在则提示用户已注册；"
            "不存在 → 写入数据库 → 发送欢迎邮件 → 结束。"
        ),
        "checks": [
            {"name": "首行 flowchart 声明", "re": r"^\s*flowchart\b"},
            {"name": "含菱形判断节点", "re": r"\{[^}]*\}"},
            {"name": "含分支标签（-- 是/否 --> 或 -->| |）", "re": r"--[^\n]*-->|-->\|"},
            {"name": "节点数≥6", "re": r"(?s)(\w+\s*[\[\{(\(]).*(\w+\s*[\[\{(\(]).*(\w+\s*[\[\{(\(]).*(\w+\s*[\[\{(\(]).*(\w+\s*[\[\{(\(]).*(\w+\s*[\[\{(\(])"},
        ],
        "gt": "首行 flowchart；需 2 个判断节点（参数校验、查重）+ 6 个以上节点 + 分支标签",
    },
    {
        "id": "w2_sequence",
        "kind": "write",
        "title": "生成·认证时序图（含分支）",
        "src": None,
        "ask": (
            "用 mermaid sequenceDiagram 描述：客户端请求网关，网关转发到认证服务，"
            "认证服务查询 Redis，命中则返回 token，未命中返回 401。"
        ),
        "checks": [
            {"name": "首行 sequenceDiagram 声明", "re": r"^\s*sequencediagram\b"},
            {"name": "含 participant", "re": r"participant"},
            {"name": "用 alt/else 表达分支", "re": r"\balt\b|\belse\b"},
            {"name": "出现 401", "re": r"401"},
        ],
        "gt": "首行 sequenceDiagram；4 参与者；用 alt/else 表达命中/未命中分支",
    },
    # ==================== 改图 ====================
    {
        "id": "e1_edit",
        "kind": "edit",
        "title": "改图·插入审计日志节点并保持原结构",
        "src": FLOW_DECISION,
        "ask": (
            "在上面这个流程图里插入一个新节点：在 F[处理请求] 之后、H[返回结果] 之前，"
            "插入 AUDIT[写审计日志]，并把边接好。输出修改后的完整 mermaid 代码，"
            "其余部分保持原样。"
        ),
        "checks": [
            {"name": "保留原有 8 个节点结构", "all": ["a[收到请求]", "b{已登录", "e{有权限"]},
            {"name": "新节点 AUDIT 存在", "re": r"audit|审计"},
            {"name": "F 的出边指向审计节点", "re": rf"f\s*{_ARW}\s*[^\n]{{0,30}}(audit|审计)"},
            {"name": "审计节点接回 H", "re": rf"(audit|审计)[^\n]{{0,40}}{_ARW}\s*h\b"},
        ],
        "gt": "F→AUDIT→H，其余 8 个节点与所有分支保持不变",
    },
]
