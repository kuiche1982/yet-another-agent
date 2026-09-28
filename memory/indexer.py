"""
知识库索引工具
智能分类 + 长文本摘要 + embeddings 编码
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import litert_lm

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.mem_classifier import MemoryClassifier
from memory.mem_manager import MemoryManager
from memory.incremental_tree import FragmentNode


class KnowledgeIndexer:
    """知识库索引器"""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        output_embeddings_dir: str = "data/embeddings",
        classify_threshold: int = 500,  # 超过此长度先摘要再分类
        use_llm: bool = True
    ):
        self.raw_dir = Path(raw_dir)
        self.output_embeddings_dir = Path(output_embeddings_dir)
        self.classify_threshold = classify_threshold

        # 初始化分类器
        print("[索引器] 初始化分类器...")
        self.classifier = MemoryClassifier(
            use_llm=use_llm,
            summarize_threshold=classify_threshold
        )

        # 初始化 MemoryManager（用于保存原始数据）
        # 创建临时配置文件
        import tempfile
        config = {"use_llm_classifier": False}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config, f)
            config_path = f.name

        self.mem_manager = MemoryManager(
            config_path=config_path,
            enable_auto_classify=False  # 索引器负责分类
        )

        # 统计信息
        self.stats = {
            "total_processed": 0,
            "classified": {},
            "summarized": 0,
            "skipped": 0
        }

    def index_file(self, file_path) -> Dict[str, Any]:
        """
        索引单个文件

        Args:
            file_path: JSONL 文件路径（str 或 Path）

        Returns:
            索引统计信息
        """
        file_path = Path(file_path)
        print(f"\n{'='*60}")
        print(f"索引文件: {file_path.name}")
        print(f"{'='*60}")

        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return {"error": f"文件不存在: {file_path}"}

        # 统计当前文件
        file_stats = {
            "file": file_path.name,
            "total_lines": 0,
            "processed": 0,
            "classified": {},
            "summarized": 0,
            "errors": 0
        }

        # 读取并索引文件
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                file_stats["total_lines"] += 1

                try:
                    entry = json.loads(line)
                    text = entry.get("text", "").strip()
                    if not text:
                        file_stats["errors"] += 1
                        continue

                    # 智能分类 + 摘要
                    result = self.classifier.classify_auto(text)

                    mem_type = result["primary"]
                    is_summarized = result.get("summarized", False)
                    summary = result.get("summarized_text", "")

                    # 使用关键信息（如果被摘要过）
                    index_text = summary if is_summarized and summary else text

                    # 保存到 MemoryManager（会自动分类并保存到 raw_dir）
                    self.mem_manager.add_fragment(
                        text=index_text,
                        type=mem_type,
                        project_id=entry.get("project_id", "default"),
                        add_to_summary=False  # 索引时不构建摘要树
                    )

                    # 更新统计
                    file_stats["processed"] += 1
                    file_stats["classified"][mem_type] = file_stats["classified"].get(mem_type, 0) + 1
                    if is_summarized:
                        file_stats["summarized"] += 1

                    # 进度输出
                    if file_stats["processed"] % 100 == 0:
                        print(f"  进度: {file_stats['processed']}/{file_stats['total_lines']} "
                              f"(摘要: {file_stats['summarized']})")

                except json.JSONDecodeError as e:
                    print(f"  ⚠️ JSON 解析错误: {e}")
                    file_stats["errors"] += 1
                except Exception as e:
                    print(f"  ⚠️ 处理错误: {e}")
                    file_stats["errors"] += 1

        print(f"\n✅ 文件处理完成: {file_stats['processed']}/{file_stats['total_lines']}")
        print(f"   分类分布: {file_stats['classified']}")
        if file_stats["summarized"] > 0:
            print(f"   长文本摘要: {file_stats['summarized']} 条")

        return file_stats

    def index_directory(self, pattern: str = "*.jsonl") -> List[Dict[str, Any]]:
        """
        索引目录下的所有文件

        Args:
            pattern: 文件匹配模式（默认 *.jsonl）

        Returns:
            所有文件的统计信息列表
        """
        print(f"\n{'='*60}")
        print(f"开始索引目录: {self.raw_dir}")
        print(f"匹配模式: {pattern}")
        print(f"{'='*60}\n")

        if not self.raw_dir.exists():
            print(f"❌ 目录不存在: {self.raw_dir}")
            return []

        # 查找所有 JSONL 文件
        jsonl_files = list(self.raw_dir.glob(pattern))
        if not jsonl_files:
            print(f"❌ 没有找到匹配的文件")
            return []

        print(f"找到 {len(jsonl_files)} 个文件\n")

        # 索引所有文件
        all_stats = []
        for file_path in sorted(jsonl_files):
            file_stat = self.index_file(file_path)
            all_stats.append(file_stat)
            self.stats["total_processed"] += file_stat["processed"]

        # 汇总统计
        self._print_summary(all_stats)

        return all_stats

    def _print_summary(self, all_stats: List[Dict[str, Any]]):
        """打印汇总统计"""
        print(f"\n{'='*60}")
        print(f"索引汇总")
        print(f"{'='*60}")

        total_processed = sum(s.get("processed", 0) for s in all_stats)
        total_summarized = sum(s.get("summarized", 0) for s in all_stats)
        total_errors = sum(s.get("errors", 0) for s in all_stats)

        print(f"总处理文件: {len(all_stats)}")
        print(f"总处理条目: {total_processed}")
        if total_summarized > 0:
            print(f"长文本摘要: {total_summarized} 条")
        if total_errors > 0:
            print(f"错误条目: {total_errors}")

        # 按类型统计
        type_stats = {}
        for stat in all_stats:
            for mem_type, count in stat.get("classified", {}).items():
                type_stats[mem_type] = type_stats.get(mem_type, 0) + count

        if type_stats:
            print(f"\n分类分布:")
            for mem_type, count in sorted(type_stats.items(), key=lambda x: -x[1]):
                print(f"  {mem_type}: {count}")

        print(f"\n{'='*60}\n")

    def rebuild_embeddings(self, force: bool = False):
        """
        重新构建 embeddings

        Args:
            force: 是否强制重新编码（即使已有 embeddings）
        """
        print(f"\n{'='*60}")
        print(f"重新构建 Embeddings")
        print(f"{'='*60}\n")

        # 清空现有 embeddings
        if force:
            print("⚠️ 强制模式，将覆盖现有 embeddings")
            for mem_type in ["code", "knowledge", "facts", "config", "session", "task"]:
                type_dir = self.output_embeddings_dir / mem_type
                if type_dir.exists():
                    import shutil
                    shutil.rmtree(type_dir)
                    print(f"  清空: {type_dir}")

        # 重新编码所有文本
        from .retriever import BGERetriever

        retriever = BGERetriever()

        # 编码所有原始数据
        print("\n编码所有文本...")
        retriever.encode_documents([])  # 先清空
        retriever.encode_documents(self.mem_manager.get_all_texts())
        print("✅ 编码完成\n")

        print(f"{'='*60}\n")

    def run_full_index(self):
        """运行完整索引流程"""
        print("\n" + "="*60)
        print("知识库索引流程")
        print("="*60)

        # 1. 索引原始数据
        self.index_directory()

        # 2. 重新构建 embeddings
        self.rebuild_embeddings(force=False)

        print("✅ 索引完成！")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="知识库索引工具")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="原始数据目录（默认：data/raw）"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="data/embeddings",
        help="embeddings 输出目录（默认：data/embeddings）"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="文件匹配模式（默认：*.jsonl）"
    )
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="重新构建 embeddings（覆盖现有）"
    )
    parser.add_argument(
        "--classify-threshold",
        type=int,
        default=500,
        help="分类摘要阈值（默认：500 tokens）"
    )

    args = parser.parse_args()

    # 初始化索引器
    indexer = KnowledgeIndexer(
        raw_dir=args.raw_dir,
        output_embeddings_dir=args.embeddings_dir,
        classify_threshold=args.classify_threshold,
        use_llm=True
    )

    # 运行索引
    indexer.run_full_index()


if __name__ == "__main__":
    main()
