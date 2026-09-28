# Agent 记忆系统

基于 **BGE-M3** + **增量递归摘要树** 的分层记忆系统，为 general purpose agent 提供强大的记忆能力。

## 📋 特性

- **6类记忆分类**：会话瞬时、个人事实、静态公理、碎片化事件、高层聚合摘要、目标计划
- **混合检索**：dense + sparse 向量检索，准确率高
- **自动分类**：规则 + LLM 混合分类，智能识别记忆类型
- **增量摘要**：递归摘要树，支持碎片化输入的分层聚合
- **持久化存储**：SQLite 元数据 + 文件系统存储
- **批量导入**：支持从 session 文件批量导入数据

## 📁 目录结构

```
memory/
├── config.yaml              # 系统配置
├── models/
│   └── bge-m3-mlx-6bit/     # BGE-M3 模型
├── raw/                     # 原始碎片存储
│   ├── session/             # 会话对话
│   ├── knowledge/           # 项目规范
│   ├── facts/               # 个人事实
│   ├── code/                # 代码片段
│   ├── config/              # 配置项
│   └── task/                # 待办事项
├── embeddings/              # BGE-M3 向量
│   ├── dense/               # dense向量
│   └── sparse/              # sparse向量
├── summaries/               # 递归摘要树
│   ├── L0/                  # 原始碎片摘要
│   ├── L1/                  # 事件级摘要
│   └── L2/                  # 项目级总摘要
├── metadata.db              # SQLite 元数据
├── incremental_tree.py      # 增量递归摘要树
├── mem_classifier.py        # 记忆分类器
├── mem_manager.py           # 记忆管理器
├── retriever.py             # BGE-M3 检索器
├── pipeline.py              # 数据导入管道
├── example.py               # 使用示例
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 mlx-embeddings（用于 BGE-M3）
pip install mlx-embeddings tiktoken numpy

# 安装 litert-lm（用于分类）
# 已安装在项目中
```

### 2. 基础使用

```python
from memory import MemoryManager, BGERetriever

# 初始化
memory_manager = MemoryManager()

# 添加记忆碎片（自动分类）
memory_manager.add_fragment(
    text="def process_data(data): return data * 2",
    project_id="demo"
)

memory_manager.add_fragment(
    text="项目使用 Redis 作为缓存层",
    project_id="demo"
)

# 获取全局摘要
global_summary = memory_manager.get_global_summary()
print(f"全局摘要: {global_summary}")

# 列出类型
print(memory_manager.list_types())
```

### 3. 批量导入 Session 文件

```python
from memory import import_sessions_batch

# 初始化
memory_manager = MemoryManager()

# 导入 session 目录
result = import_sessions_batch(
    sessions_dir="/path/to/sessions",
    memory_manager=memory_manager,
    project_id="my-project",
    parse_mode="conversation"
)

print(f"导入完成: {result['total_fragments']} 个碎片")
print(f"类型分布: {result['memory_types']}")
```

### 4. BGE-M3 检索

```python
from memory import BGERetriever

# 初始化检索器
retriever = BGERetriever()

# 编码文档
documents = ["文档1...", "文档2...", "文档3..."]
retriever.encode_documents(documents)

# 执行检索
results = retriever.hybrid_retrieve("查询文本", top_k=3)
for i, text in enumerate(results, 1):
    print(f"{i}. {text}")
```

## 📊 记忆分类

| 类型 | 说明 | 识别特征 | 示例 |
|------|------|----------|------|
| **code** | 代码片段 | `def`, `class`, `import`, 代码块 | `def process_data(data): ...` |
| **knowledge** | 规范/文档 | `API文档`, `规范`, `协议`, `标准` | `REST API规范要求...` |
| **facts** | 个人事实 | `我叫`, `邮箱`, `偏好`, `习惯` | `我不喜欢红色` |
| **config** | 配置项 | `env.`, `配置文件`, `URL=`, `PORT=` | `database=production` |
| **session** | 对话/讨论 | `好的`, `收到`, `讨论`, `意见` | `收到，我会尽快处理` |
| **task** | 待办/计划 | `待办`, `计划`, `目标`, `TODO` | `TODO: 完成数据库迁移` |

### 自动分类

分类器使用规则 + LLM 混合策略：

1. **规则分类**：快速、零成本
   - 匹配预设模式（正则表达式）
   - 返回匹配类型和分数

2. **LLM 分类**：准确、灵活
   - 如果规则无匹配，使用 gemma4-e4b 模型
   - 分析文本语义，返回准确类型

3. **多候选**：返回多个候选类型
   - 如果规则有多个匹配，LLM 验证并排序
   - 调用者可选择确认或指定

## 🔍 检索机制

### 混合检索

```
查询 → BGE-M3 encode_queries → dense + sparse 向量
                                          ↓
    文档 → BGE-M3 encode_documents → dense + sparse 向量
                                          ↓
    混合打分 = dense_weight * dense_score + sparse_weight * sparse_score
                                          ↓
    Top-K 召回 → reranker 筛选（可选） → 返回结果
```

### 参数配置

```yaml
# config.yaml
dense_weight: 0.7      # dense 向量权重
sparse_weight: 0.3     # sparse 向量权重
top_k_recall: 8       # 召回数量
top_k_rerank: 3       # 重筛数量
chunk_size: 700       # 分块大小
chunk_overlap: 150    # 重叠大小
```

## 🧠 递归摘要树

### 树结构

```
L2 (项目级总摘要)
└── L1 (事件级摘要)
    ├── L0 (原始碎片)
    ├── L0 (原始碎片)
    └── L0 (原始碎片)
```

### 工作原理

1. **添加碎片**：插入到 L0（叶子）
2. **向上合并**：每 N 个碎片合并为摘要
3. **分层压缩**：继续向上聚合，直到根节点
4. **全局摘要**：根节点提供项目级全局视图

### 使用示例

```python
# 添加碎片（自动向上合并）
memory_manager.add_fragment("任务1...", project_id="proj-a")
memory_manager.add_fragment("任务2...", project_id="proj-a")
memory_manager.add_fragment("任务3...", project_id="proj-a")

# 获取全局摘要（根节点摘要）
global_summary = memory_manager.get_global_summary()
# 输出: "项目A包含任务1、任务2、任务3，重点关注..."

# 获取摘要树深度
depth = memory_manager.get_summary_depth()  # 返回 2
```

## 📖 API 参考

### MemoryManager

```python
# 初始化
memory_manager = MemoryManager(config_path="memory/config.yaml")

# 添加碎片
result = memory_manager.add_fragment(
    text: str,
    type: Optional[str] = None,  # 可选，默认自动分类
    project_id: Optional[str] = None
)

# 获取全局摘要
summary = memory_manager.get_global_summary()

# 列出类型
types = memory_manager.list_types()
```

### BGERetriever

```python
# 初始化
retriever = BGERetriever(
    model_path="memory/bge-m3-mlx-6bit",
    config_path="memory/config.yaml"
)

# 编码文档
retriever.encode_documents(texts: List[str])

# 混合检索
results = retriever.hybrid_retrieve(query: str, top_k: int)

# 检索并返回详细信息
details = retriever.retrieve_with_details(query: str, top_k: int)
```

### SessionPipeline

```python
# 初始化
pipeline = SessionPipeline(memory_manager)

# 导入 session 文件
result = pipeline.import_session(
    session_path: str,
    project_id: str,
    parse_mode: str = "conversation"
)

# 批量导入
result = import_sessions_batch(
    sessions_dir: str,
    memory_manager: MemoryManager,
    project_id: str,
    parse_mode: str = "conversation"
)
```

## 🧪 测试

运行示例：

```bash
cd ~/kuiwork/workdir2/litertlm
python memory/example.py
```

## 📝 配置说明

### config.yaml

```yaml
# 模型配置
model_path: "memory/bge-m3-mlx-6bit"
model_max_length: 8192

# 分块参数
chunk_size: 700
chunk_overlap: 150

# 混合检索参数
dense_weight: 0.7
sparse_weight: 0.3
top_k_recall: 8
top_k_rerank: 3

# 递归摘要参数
summary_group_size: 2
summary_max_tokens: 700

# 记忆分类配置
use_llm_classifier: true
llm_model_path: "~/.litert-lm/models/gemma4-e4b/model.litertlm"

# 持久化路径
raw_dir: "memory/raw"
embeddings_dir: "memory/embeddings"
summaries_dir: "memory/summaries"
metadata_db: "memory/metadata.db"
```

## 🔧 高级功能

### 自定义分类器

```python
from memory import MemoryClassifier

classifier = MemoryClassifier(use_llm=False)  # 禁用 LLM

# 使用规则分类
candidates = classifier.classify("text...")
print(candidates)  # [('facts', 2), ('session', 1)]
```

### 自定义 Reranker

```python
class CustomReranker:
    def rerank(self, query: str, results: List[Dict], top_k: int):
        # 自定义重排逻辑
        for res in results:
            res['score'] = self._compute_score(query, res['text'])
        results.sort(key=lambda x: -x['score'])
        return results[:top_k]

# 设置 reranker
retriever.set_reranker(CustomReranker())
```

## 🚧 TODO

- [ ] 集成 Reranker
- [ ] 支持向量数据库（Milvus）
- [ ] 实现记忆遗忘机制
- [ ] 支持记忆更新
- [ ] 添加 UI 界面
- [ ] 优化大规模数据性能

## 📄 License

MIT
