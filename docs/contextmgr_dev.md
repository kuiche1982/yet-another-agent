# contextmgr —— 开发文档（模块划分 / 扩展点 / 接入 / 运行）

> 对应 `contextmgr/` 包。本包 model-free、可单测、不依赖 LLM。

---

## 1. 模块划分

| 文件 | 职责 | 关键导出 |
|---|---|---|
| `types.py` | L0–L3 数据结构、`Source` 枚举、源优先级 | `Fragment, L1Structured, L2Visual, L3Index, Source` |
| `tokenize.py` | token 估算 / 消息组估算 / 中英混合分词（确定性；可选 tiktoken） | `estimate_tokens, estimate_messages, tokenize_words` |
| `embedder.py` | 检索打分接口；默认 `BM25Embedder`（model-free）；`BGEEmbedder` 占位 | `Embedder, BM25Embedder, BGEEmbedder` |
| `distill.py` | L0→L1/L2/L3 蒸馏；`KeywordDistiller`(通用) + `CodeDistiller`(ast) | `Distiller, KeywordDistiller, CodeDistiller, split_code_fragments` |
| `store.py` | L0 片段存储 + L1/L2/L3 派生缓存 | `FragmentStore` |
| `index.py` | L3 索引构建 + `search`（短索引匹配、会话 boost） | `Index` |
| `retrieve.py` | 预算感知 descend 到 L0 | `Retriever` |
| `compress.py` | 三压缩策略 + 两级收敛（纯函数） | `CompressionStrategy, compress, compress_two_tier` |
| `prepare.py` | **对话出向组装**：RAG 决策 → 预算裁剪 → 组装 → opt-in 召回 → 出向清理（依赖注入，不读 config、model-free） | `prepare_messages, recall_from_truth` |
| `manager.py` | `ContextManager` 编排（三源接入 + build_context + compress_if_needed + recall） | `ContextManager` |
| `llm_backends.py` | **可选** LLM 蒸馏器插件（默认不启用，包仍可 model-free 单测）；接入真模型走 L1/L2/L3 | `LLMDistiller` |
| `persist.py` | **落盘层**：FragmentStore 序列化为 markdown；L3 不落盘（加载时由 L1 重建）；按源文件 mtime 增量重建 | `MarkdownStore` 风格 API：`fragment_to_md/md_to_fragment/save_fragment/load_store/rebuild/write_l2_overview/write_l3_index` |

依赖方向：manager → {index, retrieve, compress, distill, store, embedder, tokenize, types}；
无循环依赖。

---

## 2. 扩展点（插件接口）

### 2.1 Distiller（蒸馏）
```python
class Distiller(ABC):
    def distill(self, frag: Fragment) -> tuple[L1Structured, L2Visual, L3Index]: ...
```
- 默认 `KeywordDistiller`（确定性启发式，单测/降级用）。
- `LLMDistiller`（已实现，位于 `llm_backends.py`，**可选插件**）：用弱模型（Ling/qwen/lfm）
  产出结构化 L1 + mermaid L2 + L3 标签。接口严格对齐 `Distiller`，可注入
  `ContextManager(library_distiller=LLMDistiller(model="lfm2.5-2.6b"))`。
  - 代码源仍走 `CodeDistiller`（AST，不浪费 LLM）；文本/会话源走 LLM。
  - **失败兜底**：LLM 不可用/超时/解析失败 → 静默回退 `KeywordDistiller`/`CodeDistiller`，绝不炸链路。
  - 关闭思考：`enable_thinking:False` 顶层字段（lfm 实测有效；Ling 需改 `chat_template_kwargs`）。
  - **注意**：L2 mermaid 须语法合法（避免 `end` 保留字、嵌套方括号等陷阱，见 mermaid 评估结论），
    渲染失败不丢信息。
- 代码源固定 `CodeDistiller`（stdlib ast，model-free，L0==L1 合并）。

### 2.2 Embedder（检索打分）
```python
class Embedder(ABC):
    def similarity(self, query: str, doc: str) -> float: ...
```
- 默认 `BM25Embedder`（对 L3.label + L1.key_points 打分，短文本高信噪）。
- 生产 `BGEEmbedder`：加载 BGE-M3，**只对 L3 一行标签 / L1 关键词做 embedding**（不要对整个 L0 碎片做向量），
  命中后由 `Retriever` 沿 `l0_pointer` 取精确片段。未接入时抛 `NotImplementedError`，避免静默走错路径。

### 2.3 压缩策略
`compress(messages, budget, strategy)` 纯函数；新增策略在 `CompressionStrategy` 枚举加值并实现分支即可。

---

## 3. 接入 harness（swe_agent/management.py）

> **状态（2026-09-12）：已接入主链路。** 早期本节列的是「A 包裹 / B 替换」二选一草案，
> 实际落地方案为**薄适配层 + 调用期注入**（见下），比 A/B 都更贴合「contextmgr 承担全部内容管理」。

### 3.1 落地方案：harness `ContextManager` = contextmgr 算法 + harness 配置

`swe_agent/management.ContextManager` 仍是 `RunState.cm`（会话 buffer 拥有者），但**只剩接线**：

| 归 contextmgr（内容算法） | 归 harness（配置 / 状态 / 协议） |
|---|---|
| `prepare.prepare_messages` 组装（RAG→压缩→组装→召回） | 会话状态：`_msgs` / `_truth`（真相日志） |
| `prepare.recall_from_truth` 召回打分 | **循环防护计数 `guard`**（`consec_repeat` / `no_tool_streak` / `empty_streak`）——「模型/工具调用次数」策略，阈值来自 `LoopConfig` |
| `compress_two_tier` 预算收敛（COMPRESS→SLIDING→语义后端） | 配置注入：`C.RAG_INJECT_*` / `kb_budget` / `compress_backend`（副驾） |
| `inject.*` 出处/切片/注入块组装 | OpenAI 协议出向清理 `_strip_reasoning`（剥 `reasoning_content`、tool 轮 content 归零） |
| `layered.LayeredKB` 检索 + `persist` 落盘 | REPL 流式钩子 `on_message`（增量渲染）+ `ModelManager`（load/unload） |

**关键纪律**：contextmgr **不得 import `swe_agent.config`**（import 期冻结值 → monkeypatch / e2e env 失效）。
harness 在**调用期**把后端作为可调用对象注入，缺省即恒等行为（无 RAG、不压缩、不清洗），故 contextmgr 单测天然 model-free：

```python
# swe_agent/management.py（薄层，示意）
return _prepare.prepare_messages(
    self._msgs, model_context_length=..., user_input=..., recall_query=...,
    truth=self._truth,                 # 真相日志 → opt-in 召回
    rag_refs=self._rag_refs,           # LayeredKB 检索
    rag_block=self._rag_block,         # 注入块（含出处行号 + 可选副驾摘取）
    compress_fn=self._compress_two_tier,
    recall_fn=self.recall,
    sanitize=self._strip_reasoning,    # 协议出向清理
)
```

数据流（每轮出向）：`agent._step` → `cm.prepare_messages(model_context_length, user_input)`
→（ctxmgr）RAG 精选 → 预算裁剪 → 组装 → `sanitize` → 发模型。

🔴 **不变量（勿简化）**：`tool_call` 循环中**绝不**在末尾重复注入 user input + RAG；仅当「当前用户轮
恰好是 buffer 末尾那条 user」或「`user_input` 与末条 user 不一致」时才拼回。守卫测试见
`tests/test_contextmgr_prepare_messages.py::test_prepare_messages_no_user_duplicate_after_tool_call`。

资料库/代码预 ingest：走 `layered.LayeredKB` 的磁盘增量重建（`persist.rebuild`），
落盘目录由 harness 侧 `C.*_KB_DIR` 决定（见 `docs/contextmgr_design.md` 的四层作用域）。


---

## 4. 知识库（Library）接入 —— 以 `KnowledgeBase/` 为例

本包对「文档进资料库」已提供现成路径。约定：把文档放进一个目录（例 `KnowledgeBase/`），
按扩展名路由注入，无需逐个文件手写。

### 4.1 目录约定（扩展名路由）
| 扩展名 | 注入为 | 调用 |
|---|---|---|
| `.md` `.markdown` `.txt` `.rst` | **Library**（资料库，文档化语义） | `ingest_library` |
| `.py` `.pyi` | **Code**（代码，AST 切分，函数/类级 L0） | `ingest_code` |

> `KnowledgeBase/` 当前含：`howto.md`、`workflow设计模式.md`（→Library）、`mcp_test_server.py`（→Code）。
> 代码文件走 Code 线而非 Library——开发 agent 语境下 Code 是与 Session/Library **并列的第三条线**（见设计文档）。

### 4.2 一键注入脚本（推荐）
`scripts/ingest_knowledge_base.py` 扫描目录、按扩展名路由、注入后可跑样例 query 验证回落：
```bash
# 默认 KeywordDistiller（model-free，可复现）
PYTHONPATH=. python scripts/ingest_knowledge_base.py

# 用 lfm 走真实 LLM 蒸馏（需模型在线，量大时较慢）
PYTHONPATH=. python scripts/ingest_knowledge_base.py --llm

# 只注入不跑 demo / 指定目录 / 自定义单条 query
PYTHONPATH=. python scripts/ingest_knowledge_base.py --no-demo
PYTHONPATH=. python scripts/ingest_knowledge_base.py --root path/to/KB --query "KV 量化怎么选"
```
脚本核心是可复用函数 `build_knowledge_base(cm, root)`，在你的 agent 启动处调用一次即可：
```python
from contextmgr import ContextManager, BM25Embedder, KeywordDistiller
from scripts.ingest_knowledge_base import build_knowledge_base   # 或复制该函数
cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(), library_distiller=KeywordDistiller())
build_knowledge_base(cm, root="KnowledgeBase")   # 注入 Library + Code 两线
```

### 4.3 单文件 / 单字符串 API
不想要目录扫描时，直接调：
```python
cm.ingest_library(open("doc.md", encoding="utf-8").read(), doc_id="kb:doc")  # 资料库
cm.ingest_code(open("mod.py", encoding="utf-8").read(), name="swe_agent/mod.py")  # 代码
```
- `doc_id` 仅作来源标签（`origin`），会进入 `fid = f"{doc_id}#p{i}"`，便于 `l0_pointer` 回落与审计。
- 切块规则：`ingest_library` 按**空行**（`\n\n`）切 L0 片段，每块一次蒸馏。

### 4.4 外部资料库适配器（WorkBuddy 资料库 / 腾讯文档 / 远端 wiki）
架构已留好扩展点：写一层薄 adapter，从外部系统**拉取纯文本**，逐篇 `ingest_library` 即可。
伪代码：
```python
def ingest_from_workbuddy_kb(cm, kb_client):
    for doc in kb_client.list_docs():
        text = kb_client.get_text(doc.id)          # 取纯文本
        cm.ingest_library(text, doc_id=f"wbkb:{doc.id}")
# 腾讯文档同理：tencent_docs 拉 markdown -> ingest_library(..., doc_id="tdoc:xxx")
```
要点：
- adapter **只负责取文本**，蒸馏/RAG/回落全交给 `contextmgr`，勿在 adapter 里做语义切分。
- 外部库变更时需**重注入**（当前 `FragmentStore` 内存态，无自动 watch/同步，见 §6）。

### 4.5 验证命令（确认注入与回落正确）
```bash
# 注入后看分源统计
PYTHONPATH=. python -c "
from contextmgr import ContextManager, BM25Embedder
from scripts.ingest_knowledge_base import build_knowledge_base
cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder())
build_knowledge_base(cm, 'KnowledgeBase')
print(cm.stats())                       # by_source: library=N, code=M
print([f.origin for f in cm.retrieve('judge mode 怎么用')])  # 应回落到 workflow设计模式.md
"
```

### 4.6 落盘持久化（markdown，增量重建）

`FragmentStore` 默认内存态、进程退出即丢。加 `--persist` 落盘为 markdown，重启/换进程无需重蒸馏（除非源变了）。

**落盘介质约定（呼应最初「mermaid DSL 作可视化介质」前提）：**
- **mermaid 是 L2 介质**：每个片段的 `## L2 Visual` 写 ```` ```mermaid ```` 详图；语料级 `L2_overview.md` 是聚合 mermaid `mindmap` 脑图。
- **L3 不落盘**：L3 是纯派生视图（label 来自 L1、指针来自 L0），加载时由 `_regen_l3` 从 L1 重建，每次启动自动重建。落盘只存 **L0(Truth) / L1(Structured) / L2(mermaid)** 三级，不把低级别塞进 L3。
- 每片段一个 `.md`，frontmatter 带 `updated_level`（算到哪一级：0/1/2）、`src_mtime`（源文件 mtime）、
  `src_size` + `src_sha256`（**源文件内容指纹**，增量判定兜底用；Session 无源文件改存片段级 `l0_sha256`）。

**增量重建语义（`persist.rebuild`，`_cache_fresh` 判定）：**

判定「缓存是否仍新」走**双路**，任一命中即磁盘 load，都不命中才重派生：

| 路径 | 条件 | 覆盖场景 |
|---|---|---|
| **快路径** | `frontmatter.src_mtime == 当前文件 mtime`（逐位相等） | 常规：文件没被动过，免掉一次 hash |
| **慢路径（兜底）** | `src_sha256 == hashlib.sha256(file_bytes)` **且** `src_size == 文件大小` | ① mtime **回拨**（`git checkout` 旧版、`rsync -t` 保 mtime）② mtime 被**推到未来**（`touch`、编辑器「保存但无改动」） |

外加前提：所有缓存片段 `updated_level >= 2`；`src_sha256` 缺失（旧 cache / 读文件失败）一律判 stale 走重派生。

> 设计要点：**内容指纹才是真相，mtime 只是廉价前置筛子**。原来写的是
> `cached.mtime >= file_mtime` 单向比较，会导致「touch 一下就整份语料重蒸馏」；
> 且 mtime 回拨时会漏检变更。现在两个方向都被 hash 兜住。

其它语义不变：源文件已删 → 清理其落盘片段；变了/新增 → 从 L0 重派生该文件全部片段，
末尾 `_regen_l3` 自动 resync L1/L2/L3。

```bash
# 落盘注入（首次：全量蒸馏并写盘；之后：未变文件磁盘 load，变了才重派生）
PYTHONPATH=. python scripts/ingest_knowledge_base.py --persist

# 落盘目录默认 <repo>/.contextmgr_store ；可指定
PYTHONPATH=. python scripts/ingest_knowledge_base.py --persist --cache-dir /path/to/cache

# 程序内等价调用（agent 启动时跑一次即可）
from contextmgr import persist as P
P.rebuild(cm, 'KnowledgeBase', '.contextmgr_store', 'keyword', {'.md','.txt','.rst'}, {'.py'}, base='.')

# 只看落盘产物（不依赖模型）
ls .contextmgr_store/            # 每片段一个 .md + L2_overview.md + L3_index.md
cat .contextmgr_store/L2_overview.md   # 语料级 mermaid 脑图（L2 介质）
cat .contextmgr_store/L3_index.md     # 纯文本索引 + 回 L0 指针（L3 由 L1 重建，非 mermaid）
```

单测：`tests/test_contextmgr_persist.py`（roundtrip / 增量跳过未变 / 变更重派生 / **mtime 回拨重派生** /
**只 touch 不重派生** / frontmatter 含源指纹 / 聚合产物 / Code 增量）。

### 4.7 三源（Library / Code / Session）持久化与加载对照

contextmgr 管理三条**并列**记忆线，开发 agent 语境下三者缺一不可。下表说清每条线的
「L0–L2 何时生成、L3 何时重建、是否落盘、重启时如何加载」。

| 源 | L0 写入时机 | L1/L2 生成 | L3 重建（由 L1，不读 L2、不落盘） | 落盘 | 重启加载 |
|---|---|---|---|---|---|
| **Library** | `ingest_library`（文档导入） | 同上，落盘 | 加载时 `_regen_l3` | ✅ 文件 `mtime` 增量 | ✅ 读 cache + L3 |
| **Code** | 代码更改（`rebuild` 判 `mtime` 变） | 同上，落盘 | 加载时 + 代码改时 `_regen_l3` | ✅ 文件 `mtime` 增量 | ✅ 读 cache + L3 |
| **Session** | `append`（会话消息） | 同上，落盘 | `append` 时（内存）+ 加载时 `_regen_l3` | ✅ `session_buffer.json` + 片段 `.md` | ✅ `load_session` 还原 buffer + 片段 |

**统一不变量**（贯穿三源）：
> L3 永远由 L1(+L0 指针) 派生、永不在 L2 之后、永不落盘、每次 store 被（重）填充就重建。
> Library/Code 的 store 填充来自磁盘 → 加载时重建；Session 的 store 填充来自 `append`(内存) 与 `load_session`(磁盘)。

**Session 线为什么必须落盘 + 加载**（「火车票缺口」）：
- 场景：agent 这轮对话里用户说「我订了火车票去北京」，进程退出；下次启动若 Session 不落盘，
  agent 不知道有这回事。
- 修正：Session 片段本就与 Library/Code 片段一样躺在 `store` 内，只需补 `save_session` / `load_session`：
  - `save_session(cache_dir)`：写 `session_buffer.json`（原始消息 role+content 序列）+ 每个 Session 片段的 `.md`（L0/L1/L2；`src_mtime` 用落盘时刻，会话无源文件）。
  - `load_session(cache_dir)`：读 `session_buffer.json` 还原 `cm.buffer` + 读 Session 片段进 store，并 `_regen_l3`。
- 推荐启动顺序：`cm.load_session(cache)` **先**于 `persist.rebuild(...)`，使 `L2_overview.md` / `L3_index.md` 也含会话标签；二者都跑后 store 含三线、L3 完整。
- `rebuild` 的 prune 分支**跳过 Session 源**（Session 由 save/load 管理，不在 `root` 文件扫描内），不会误删。

**代码入口**：
```python
# 进程退出前（或每次重要会话后）
cm.save_session('.contextmgr_store')
# 重启后、注入知识库前
cm.load_session('.contextmgr_store')
P.rebuild(cm, 'KnowledgeBase', '.contextmgr_store', 'keyword', {'.md'}, {'.py'}, base='.')
```

**三线连调验证**：`tests/test_contextmgr_integration.py` —— 建三线 → 落盘 → 新 ContextManager 重启加载
→ 断言三源片段俱在、会话 buffer 含「火车票」、且 `retrieve` 可分别按 Session/Library/Code 三源跨重启命中、
`build_context` 组装不超预算。model-free（KeywordDistiller），证明架构连接正确。

### 4.8 Code 线 L0–L3 分层实现对照（与需求定义核对）

Code 源四层的**权威定义**（对齐「源代码脑图最顶层抽象」诉求）：

| 层 | 定义 | 实现位置 | 实际产出 | 状态 |
|---|---|---|---|---|
| **L0** | 源代码本身 | `distill.split_code_fragments` + `manager.ingest_code` | `ast.parse` → 每顶层定义（FunctionDef / AsyncFunctionDef / ClassDef）一个 `Fragment`：`fid={file}#{name}`、`text=完整定义源码`、`offset/length=字符位移`、`lineno/end_lineno` | ✅ |
| **L1** | 函数/类/接口的**详细结构化描述，带所在文件与行号** | `distill.CodeDistiller.distill` | `key_points=["def foo(a, b) — docstring 一行 @ m.py:1-3", ...]`、`entities=[名字]`、`calls=[被本定义调用的同模块名字]`、`merged_with_l0=True`（代码即结构，L0==L1） | ✅ 行号已补 |
| **L2** | mermaid **SDL 缩减描述 + 调用指引**（脑图最顶层抽象） | `distill.CodeDistiller` + `persist.write_l2_callgraphs` | 片段级：`graph TD`，模块 → 各定义；调用边 `D1 -->|"calls"| D0`。语料级 `L2_callgraphs.md`：**跨片段同模块 def→def 调用图**，外部调用画孤立灰节点降噪 | ✅ 调用边已补 |
| **L3** | 一行索引 + 回 L0 精确指针 | `distill.CodeDistiller` + `persist._regen_l3` | `label="m.py#foo: 1 定义 @ m.py:1-3"`（截断 60）、`l0_pointer=(fid, offset, length)`；`L3_index.md` 一行一条 `- [category] label => fid(off,len)` | ✅ |
| **descend** | 逐级递进搜索 | `index.Index.search` + `retrieve.Retriever.retrieve` | BM25 只在 `L3.label + L1.key_points` 打分 → 沿 `l0_pointer` 回落 L0。`include_levels=True` 返回 `[(L3, L1, L0)]` 三段式，供模型「命中标签 → 函数签名 → 完整代码」导航 | ✅ L1 已暴露 |

**为什么 L1 对 Code 是「L0==L1 合并」**：代码本身已结构化，再做文本蒸馏是冗余。
但 L1 仍要承载**导航信息**（行号、调用关系），否则模型拿到 L0 后无法回答「这函数在文件哪、被谁调用」——
这部分通过 `key_points` 的 `@ file:N-M` 与 `calls` 字段补上，不额外占 L0 空间。

**聚合产物清单**（`cache_dir/`）：
- `L2_overview.md` —— 语料级 mermaid `mindmap`（L2 介质）
- `L2_callgraphs.md` —— 语料级 def→def 调用图（L2 介质，仅 Code 线有内容）
- `L3_index.md` —— 纯文本索引 + 指针（**非 mermaid**，L3 不落盘、加载时重建）
- `session_buffer.json` —— Session 线原始消息序列

三者都被 `load_store` / `load_session` / `_scan_cache` 的聚合文件白名单排除，不会被误当片段解析。

**入口**：
```python
from contextmgr import persist as P
# 每段都拿 L3/L1/L0 三段式（给模型做导航）
for l3, l1, l0 in cm.retrieve('foo 的实现', include_levels=True):
    print(l3.label, '->', l1.key_points, '->', l0.text[:80])
```

单测：`tests/test_contextmgr_code_levels.py`（行号 / calls / L2 调用边 / 语料级调用图 / 三段式 descend /
跨重启持久化，8 条）。

---

## 5. 运行与验证

```bash
# 单测（model-free / tool-free）—— 用项目 venv，别用裸 python
.venv/bin/python -m pytest tests/test_contextmgr_*.py -q

# venv 重建后（uv sync 会按 [dependency-groups].dev 装 pytest）
uv sync --all-groups   # 或 uv pip install pytest
```

# 极简冒烟（不写文件，直接 import 跑）
python -c "
from contextmgr import ContextManager, CompressionStrategy, BM25Embedder
cm = ContextManager(budget_tokens=1024, embedder=BM25Embedder())
cm.ingest_library('Redis 用作缓存层，需处理击穿。', doc_id='redis')
cm.ingest_code('def f():\n    return 1\n', name='m.py')
cm.append('user', 'Redis 击穿怎么处理')
print([m['role'] for m in cm.build_context('Redis 击穿', budget_tokens=512)])
print(cm.stats())
"
```

### 端到端流程验证（可选，需真模型）
```bash
# 用 lfm2.5-2.6b(nothinking) 走真蒸馏，验证 Library→L1/L2/L3→检索→回落 L0→组装
PYTHONPATH=. python scripts/contextmgr_flow_demo.py
```
已验证（lfm2.5-2.6b nothinking）：Library 经 LLM 蒸馏出 L3 标签（如 `Redis 缓存三大问题及解决方案`）
+ 合法 mermaid L2；Code 经 AST（model-free）；Session 高优先级；query 命中后 `Retriever` 沿
`l0_pointer` 精确回落 L0 原文，组装进 `build_context` 的 RAG 块。证明架构在真实模型下链路通。
（该脚本非单测套件一部分；单测全 model-free。）

---

## 6. 已知限制 / 后续

- 持久化已做（markdown 落盘 + 按 mtime 增量重建，见 §4.6）。当前为「每进程启动 rebuild 一次」模型；
  同进程内重复调用 rebuild 仅测试场景，磁盘陈旧风险已在重派生分支按 origin 强制重存规避。
  L3 不落盘、每次启动由 L1 重建（`_regen_l3`）。
- **Session 线持久化已补**（§4.7）：`save_session`/`load_session` 闭环「火车票缺口」——会话 buffer + 片段
  落盘，重启后还原。三线（Library/Code/Session）落盘与加载对照见 §4.7。
- **增量判定已加内容指纹兜底**（§4.6）：`src_sha256` + `src_size` 与 mtime 双路判定，
  覆盖 mtime 回拨（`git checkout` / `rsync -t`）与只 touch 未改内容两类场景。
  残留边界：同一秒内「改内容后立即把 mtime 精确写回原值 *且* 内容恰好同 sha」—— 由 sha 保证不可能误判；
  真正未覆盖的是**并发写**（hash 算完后文件被改），概率极低，需要时用文件锁解决。
- **Code 线 L1/L2 已补全**（§4.8）：L1 带 `file:行号` 与 `calls`，L2 含 def→def 调用边，
  `retrieve(include_levels=True)` 可拿 L3/L1/L0 三段式。
  已知限制：调用边只解析**同模块内** `ast.Name`（`self.method()` / 跨模块 import 未解析），
  跨模块调用图待后续补 import 解析。
- 跨片段聚合（原 `memory/` 的摘要树）未纳入；当前检索是「L3→精确 L0 片段」，不做多片段合并摘要。
- 检索默认为 BM25；大规模语料应换 BGE-M3（仅对短索引 embed）。
- **已接入 `swe_agent/management.py` 主链路**（2026-09-12）：harness `ContextManager` 为薄适配层，
  内容算法全在 contextmgr（见 §3）。设计文档旧版此处写的「未接入、保持主链路稳定」已过期。
