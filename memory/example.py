"""
记忆系统使用示例
"""
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import (
    MemoryManager,
    BGERetriever,
    SessionPipeline,
    import_sessions_batch
)


def example_1_basic_usage():
    """示例1：基础使用"""
    print("=" * 50)
    print("示例1: 基础使用")
    print("=" * 50)

    # 初始化记忆管理器
    memory_manager = MemoryManager()

    # 添加不同类型的记忆
    examples = [
        ("def process_data(data): return data * 2", "code", "数据项目"),
        ("项目使用 Redis 作为缓存层", "knowledge", "项目规范"),
        ("我不喜欢红色，偏好蓝色", "facts", "个人偏好"),
        ("REST API规范要求所有接口返回200状态码", "knowledge", "API文档"),
        ("TODO: 完成数据库迁移脚本", "task", "待办事项"),
        ("收到，我会尽快处理", "session", "项目讨论"),
    ]

    for text, mem_type, project_id in examples:
        result = memory_manager.add_fragment(
            text=text,
            type=mem_type,
            project_id=project_id
        )
        print(f"✓ 添加 {mem_type}: {text[:30]}...")
        print(f"  方法: {result['method']}, 路径: {result['path']}\n")

    # 获取全局摘要
    print(f"\n摘要树深度: {memory_manager.get_summary_depth()}")
    print(f"全局摘要: {memory_manager.get_global_summary()[:100]}...")

    # 列出类型
    print(f"\n可用类型: {memory_manager.list_types()}")


def example_2_session_import():
    """示例2：导入 session 文件"""
    print("\n" + "=" * 50)
    print("示例2: 导入 session 文件")
    print("=" * 50)

    # 初始化
    memory_manager = MemoryManager()

    # 导入 session 目录
    sessions_dir = "~/.codex/sessions/2026/07/"
    if Path(sessions_dir).exists():
        result = import_sessions_batch(
            sessions_dir=sessions_dir,
            memory_manager=memory_manager,
            project_id="codex-sessions",
            parse_mode="conversation"
        )

        print(f"\n导入完成:")
        print(f"  文件数: {result['total_files']}")
        print(f"  碎片数: {result['total_fragments']}")
        print(f"  类型分布: {result['memory_types']}")
    else:
        print(f"目录不存在: {sessions_dir}")


def example_3_retrieval():
    """示例3：BGE-M3 检索"""
    print("\n" + "=" * 50)
    print("示例3: BGE-M3 混合检索")
    print("=" * 50)

    # 初始化
    memory_manager = MemoryManager()

    # 添加测试数据
    test_fragments = [
        "项目使用 Redis 作为缓存层，减少数据库压力",
        "REST API规范要求所有接口返回200状态码",
        "def process_data(data): return data * 2",
        "收到，我会尽快处理",
    ]

    for frag in test_fragments:
        memory_manager.add_fragment(frag, project_id="test")

    # 编码文档（需要安装 mlx-embeddings）
    retriever = BGERetriever()

    print("\n准备编码文档...")
    retriever.encode_documents(test_fragments)

    # 执行检索
    queries = [
        "如何使用Redis缓存？",
        "项目API规范",
        "数据处理函数"
    ]

    for query in queries:
        print(f"\n查询: {query}")
        results = retriever.hybrid_retrieve(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result}")


def example_4_auto_classify():
    """示例4：自动分类演示"""
    print("\n" + "=" * 50)
    print("示例4: 自动分类")
    print("=" * 50)

    memory_manager = MemoryManager()

    test_texts = [
        "def hello(): return 'world'",  # 代码
        "我使用 Mac，喜欢深色主题",  # 事实
        "请按照 RFC 7231 规范实现接口",  # 知识
        "配置文件中设置 database=production",  # 配置
        "好的，没问题",  # 会话
    ]

    for text in test_texts:
        result = memory_manager.add_fragment(text, project_id="test-auto")

        print(f"\n文本: {text[:40]}...")
        print(f"  主要类型: {result['type']}")
        print(f"  方法: {result['method']}")

        # 显示分类详情
        classifier = memory_manager.classifier
        classify_result = classifier.classify_auto(text)
        print(f"  详细: {classify_result}")


def example_5_session_to_memory():
    """示例5：完整流程 - session -> 记忆系统 -> 检索"""
    print("\n" + "=" * 50)
    print("示例5: 完整流程演示")
    print("=" * 50)

    # 1. 初始化
    memory_manager = MemoryManager()
    retriever = BGERetriever()

    # 2. 添加测试数据
    test_data = [
        "Agent 的核心是工具调用",
        "使用 read_auto 命令读取文档",
        "在 plan 阶段禁止 write 和 delete",
        "每轮只生成1个文件",
        "项目使用 FastAPI + MySQL 架构"
    ]

    print("\n1. 添加记忆碎片...")
    for i, text in enumerate(test_data, 1):
        result = memory_manager.add_fragment(
            text=text,
            project_id="demo-project"
        )
        print(f"   [{i}/{len(test_data)}] ✓ {text[:30]}...")

    # 3. 编码
    print("\n2. 编码文档...")
    retriever.encode_documents(test_data)

    # 4. 检索
    print("\n3. 执行检索...")
    queries = [
        "如何使用工具？",
        "项目架构是什么？",
        "哪些操作被禁止？"
    ]

    for query in queries:
        print(f"\n   查询: {query}")
        results = retriever.hybrid_retrieve(query, top_k=3)

        for i, text in enumerate(results, 1):
            print(f"     {i}. {text[:50]}...")

        # 5. 构建Prompt
        print(f"\n   📝 构建Prompt...")
        fragments = retriever.hybrid_retrieve(query, top_k=2)
        global_summary = memory_manager.get_global_summary()

        prompt = f"""【项目全局摘要】
{global_summary[:200]}...

【检索到的细节】
{chr(10).join(f"- {f}" for f in fragments)}

任务：{query}

请基于上述信息回答问题。"""

        print(f"   Prompt 长度: {len(prompt)} 字符")
        print(f"   全局摘要长度: {len(global_summary)} 字符")


def main():
    """运行所有示例"""
    print("🧠 记忆系统使用示例\n")

    # 取消注释要运行的示例

    # example_1_basic_usage()
    # example_2_session_import()
    # example_3_retrieval()
    # example_4_auto_classify()
    example_5_session_to_memory()

    print("\n" + "=" * 50)
    print("示例运行完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
