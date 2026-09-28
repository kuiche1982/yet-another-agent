# contextmgr —— 分级记忆 + ContextManager 设计文档

> 目的：把 mermaid/压缩从「给人看的展示层」换成「给 agent 自用的上下文压缩与索引层」，
> 优化 `ContextManager`（对话历史按 token 预算自动 RAG + 剪裁）。
> 本文档对应 `contextmgr/` 包。model-free、可单测、不依赖 LLM。

---

## 0. 命名边界（必读，避免与脑图混淆）

本项目存在**两套 L 级**，语义不同，切勿混用：

| 体系 | L1 | L2 | L3 | 真相源 |
|---|---|---|---|---|
| **认知三层**（`docs/harness_design_map.md`） | 设计原则 / 架构契约 | 脑图（现状索引） | 代码（事实源） | **L3 代码** |
| **记忆分级**（本包 `contextmgr`） | 结构化蒸馏 | mermaid 可视化 | 一行索引 + 回 L0 指针 | **L0 原始片段** |

本包一律用 `Source / L0Source / L1Structured / L2Visual / L3Index` 显式命名，杜绝歧义。

---

## 1. 动机与要解决的痛点

1. **原 `memory/` 包（BGE-M3 递归摘要树）的硬伤**：
   - 它是「向上有损聚合」：L0 碎片 → L1 事件摘要 → L2 项目总摘要，**聚合后丢失回 L0 的精确指针**，无法「沿线索向下搜索到某段原文」。
   - embedding 对**整块 700-token 碎片**做 dense+sparse，检索粒度被 chunk 锁死，拿不回精确片段。
   - 本包已**取代** `memory/`（原代码保留但不集成；如确认无用可后续清理）。
2. **现有 `swe_agent/management.py` 的 `ContextManager`** 只做「长度感知对话 buffer 压缩」，没有资料库/代码 RAG 回落、没有 L3 索引分级。本包在其之上补齐。

---

## 2. 架构总览

```
                 ┌─────────────── 三源 L0（Truth of Source）───────────────┐
                 │ Library(文档)   Session(会话,高优先级)   Code(代码)      │
                 └───────┬───────────────┬───────────────┬────────────────┘
                         │ distill        │ distill        │ AST 切分+distill
                         ▼                ▼                ▼
        L1 结构化 ──┐   L1(结构化)      L1(结构化)      L0==L1(代码即结构)
        L2 可视化 ──┼─→ L2(mermaid)     L2(mermaid)     L2(代码脑图,AST直出)
        L3 索引 ───┘   L3(一行标签+指针) L3(一行标签+指针) L3(索引+指针)
                         │                │                │
                         └────────────────┴────────────────┘
                                  │ 检索：query 匹配 L3/L1（短索引）
                                  ▼
                        Retriever：按 (分数+会话boost, 源优先级) 排序
                                  │ 沿 l0_pointer 精确 descend
                                  ▼
                            L0 原文片段（组装进 context，预算贪婪填充）
                                  │
                          ContextManager.build_context
                          + compress(SlidingWindow/Compress/Aggressive)
                                  ▼
                       下一轮对话的 messages（system + RAG参考 + 压缩后buffer + 最新query）
```

**检索方向**：L3（最简略）→ 沿指针 → L0（真相）。L1/L2/L3 全是从 L0 派生的视图，
可丢失、可重建、渲染失败不丢信息。**只有 L0 是 Truth**（需求 #4）。

---

## 3. L0–L3 语义

| 层 | 是什么 | 可否丢失 | 代码类型 |
|---|---|---|---|
| **L0** | 原始片段（某段文本/代码），唯一真相 | 否 | `Fragment` |
| **L1** | 结构化蒸馏（summary/key_points/entities），JSON 友好、可检索 | 可 | `L1Structured` |
| **L2** | mermaid 可视化（agent 自导航；图里禁写绝对路径） | 可 | `L2Visual` |
| **L3** | 一行索引标签 + `(fid, offset, length)` 回 L0 指针，检索入口 | 可 | `L3Index` |

**代码源特例（澄清 #2）**：代码本身即结构，**L0 与 L1 合并**（`L1Structured.merged_with_l0=True`），
不再做冗余文本蒸馏；**L2 脑图由 AST 直接生成**（不经 L1），L3 索引照常。
Library/Session 仍走完整 L0→L1→L2→L3。

---

## 4. 三源与优先级

- `Source.LIBRARY`：文档导入资料库（需求 #1），`ingest_library(text, doc_id)`。
- `Source.SESSION`：当前会话（需求 #2），`append(role, content)` 自动建高优先级 L0 片段。
- `Source.CODE`：代码（需求 #7），`ingest_code(text, name)` → AST 切每定义一个 L0 片段，Code.L3 脑图参与检索。
- 优先级：Session(3) > Library(2) > Code(1)。检索命中项内，会话源享 `SESSION_BOOST` 保证排前（需求 #2）。

---

## 5. 检索（需求 #3/#4/#5）

- `Index.search(query)`：只在 **L3.label + L1.key_points（短、高信噪）** 上用 `Embedder` 打分；
  **绝不**对整块 L0 做向量（修正 embedding 误区）。
- 仅召回 `score > 0` 的片段；会话源在命中项内排前。
- `Retriever.retrieve(query, budget)`：沿 `l0_pointer` 精确 descend 到 L0 原文，贪婪按 token 预算填充。
- `Session.L3 + Library.L3` 即最简略资料分级，可向下搜索（需求 #3）。

---

## 6. 压缩三策略（需求 #6，全部 model-free、确定性）

| 策略 | 行为 | 适用 |
|---|---|---|
| `SLIDING_WINDOW` | 保留最后 `window` 条消息（真·滑动窗口），超预算从头砍 | 长对话保近因 |
| `COMPRESS` | 保留 system + 首部 + 尾部，中间折叠为一行占位 | 保首尾上下文 |
| `AGGRESSIVE` | 砍光历史，仅留 system + 最近一条 user；RAG 参考由 `build_context` 外拼 | 极小上下文、省 token |

---

## 7. 与 harness 的接入（已落地，2026-09-12）

- `swe_agent/management.py` 的 `ContextManager` 作 `RunState.cm`，现为**薄适配层**：
  内容算法（出向组装 / 召回 / 两级压缩 / 注入层 / 分层 KB）全在本包；
  harness 只保留**会话状态 + 循环防护计数（模型/工具调用次数）+ config 注入 + 协议出向清理**。
  边界表与代码示意见 `docs/contextmgr_dev.md` §3。
- 接入点：`agent._step` 每轮调 `cm.prepare_messages(model_context_length=..., user_input=...)`
  取出向 messages（对 buffer 只读，不缩历史；循环防护计数仍按全量历史算）。
- 生产蒸馏/检索可换弱模型（LLM Distiller / BGE-M3 Embedder），接口已留占位，默认 model-free。

---

## 8. 测试策略

- 全部 `model-free` / `tool-free`：用 `KeywordDistiller` + `CodeDistiller`(ast) + `BM25Embedder`，
  不触 LLM、不触向量模型、不触网络。
- 单测覆盖：蒸馏（L0 指针保持、L2 合法 mermaid、代码 L0==L1）、检索（descend/L3 指针/预算/会话优先/源过滤）、
  压缩（三策略保真+预算）、集成（三源接入 + build_context + AGGRESSIVE + Code.L3 参与）。
- 运行：`python -m pytest tests/test_contextmgr_*.py -q`
