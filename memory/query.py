"""
知识库查询工具
支持向量检索和关键词检索
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

from .retriever import BGERetriever


class KnowledgeQuery:
    """知识库查询器"""

    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # 初始化检索器
        print(f"[查询器] 初始化检索器...")
        print(f"[查询器] embeddings 目录: {self.embeddings_dir}")
        self.retriever = BGERetriever(
            model_path="bge-m3-mlx-6bit",
            config_path=None
        )
        print(f"[查询器] 检索器初始化完成")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        搜索知识库

        Args:
            query: 查询文本
            top_k: 返回结果数量（默认 self.top_k）
            min_score: 最低分数阈值

        Returns:
            搜索结果列表
        """
        if top_k is None:
            top_k = self.top_k

        # 检查是否有数据
        if not self.retriever.raw_texts:
            print("[查询器] 警告: embeddings 目录为空")
            return []

        print(f"\n[查询] 查询: {query}")
        print(f"[查询] Top-K: {top_k}")

        # 执行检索
        results = self.retriever.retrieve_with_details(query, top_k=top_k)

        # 过滤低分结果
        if min_score > 0:
            results = [r for r in results if r["score"] >= min_score]
            print(f"[查询] 过滤后剩余: {len(results)} 条")

        if not results:
            print("[查询] 没有找到匹配的结果")
            return []

        # 打印结果
        self._print_results(results)

        return results

    def _print_results(self, results: List[Dict[str, Any]]):
        """打印搜索结果"""
        print(f"\n找到 {len(results)} 条结果:")
        print("-" * 80)

        for i, item in enumerate(results, 1):
            print(f"\n{i}. [{item['type']}] 分数: {item['score']:.4f}")
            print(f"   {item['text']}")

    def keyword_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        关键词搜索（仅使用 sparse embedding）

        Args:
            query: 查询文本
            top_k: 返回结果数量（默认 self.top_k）
            min_score: 最低分数阈值

        Returns:
            搜索结果列表
        """
        if top_k is None:
            top_k = self.top_k

        print(f"\n[关键词搜索] 查询: {query}")
        print(f"[关键词搜索] Top-K: {top_k}")

        # 使用较小的 dense_weight，增加 sparse_weight
        results = self.retriever.retrieve_with_details(
            query,
            top_k=top_k,
            dense_weight=0.2,
            sparse_weight=0.8
        )

        # 过滤低分结果
        if min_score > 0:
            results = [r for r in results if r["score"] >= min_score]
            print(f"[关键词搜索] 过滤后剩余: {len(results)} 条")

        if not results:
            print("[关键词搜索] 没有找到匹配的结果")
            return []

        # 打印结果
        self._print_results(results)

        return results

    def search_by_type(
        self,
        query: str,
        type: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        按类型搜索

        Args:
            query: 查询文本
            type: 类型（code, knowledge, facts, config, session, task）
            top_k: 返回结果数量（默认 self.top_k）

        Returns:
            搜索结果列表
        """
        print(f"\n[类型搜索] 类型: {type}")
        print(f"[类型搜索] 查询: {query}")

        # 检查是否有该类型的数据
        type_dir = self.embeddings_dir / type
        if not type_dir.exists():
            print(f"[类型搜索] 警告: 类型 '{type}' 没有数据")
            return []

        # 执行检索
        results = self.retriever.retrieve_with_details(query, top_k=top_k)

        # 过滤指定类型
        filtered_results = [r for r in results if r["type"] == type]
        print(f"[类型搜索] 找到 {len(filtered_results)} 条 '{type}' 类型结果")

        # 打印结果
        if filtered_results:
            self._print_results(filtered_results)
        else:
            print("[类型搜索] 没有找到匹配的结果")

        return filtered_results

    def search_multi_type(
        self,
        query: str,
        types: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        多类型搜索

        Args:
            query: 查询文本
            types: 类型列表
            top_k: 返回结果数量（默认 self.top_k）

        Returns:
            搜索结果列表
        """
        print(f"\n[多类型搜索] 类型: {', '.join(types)}")
        print(f"[多类型搜索] 查询: {query}")

        # 执行检索
        results = self.retriever.retrieve_with_details(query, top_k=top_k)

        # 过滤指定类型
        filtered_results = [r for r in results if r["type"] in types]
        print(f"[多类型搜索] 找到 {len(filtered_results)} 条结果")

        # 打印结果
        if filtered_results:
            self._print_results(filtered_results)
        else:
            print("[多类型搜索] 没有找到匹配的结果")

        return filtered_results

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "total_documents": len(self.retriever.raw_texts),
            "embeddings_dir": str(self.embeddings_dir),
            "document_types": {}
        }

        # 统计各类型数量
        if self.retriever.text_types:
            for type_ in self.retriever.text_types:
                stats["document_types"][type_] = stats["document_types"].get(type_, 0) + 1

        return stats


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="知识库查询工具")
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="data/embeddings",
        help="embeddings 目录（默认：data/embeddings）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回结果数量（默认：5）"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="查询文本"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="最低分数阈值（默认：0.0）"
    )
    parser.add_argument(
        "--keyword",
        action="store_true",
        help="仅使用关键词检索（sparse embedding）"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["code", "knowledge", "facts", "config", "session", "task"],
        help="按类型搜索"
    )
    parser.add_argument(
        "--types",
        type=str,
        nargs="+",
        choices=["code", "knowledge", "facts", "config", "session", "task"],
        help="多类型搜索（用空格分隔）"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="显示知识库统计信息"
    )

    args = parser.parse_args()

    # 初始化查询器
    query = KnowledgeQuery(
        embeddings_dir=args.embeddings_dir,
        top_k=args.top_k
    )

    # 显示统计信息
    if args.stats:
        print("\n知识库统计信息:")
        stats = query.get_stats()
        print(f"  总文档数: {stats['total_documents']}")
        print(f"  分类分布:")
        for type_, count in stats['document_types'].items():
            print(f"    {type_}: {count}")

    # 执行搜索
    if args.type:
        query.search_by_type(args.query, args.type, args.top_k)
    elif args.types:
        query.search_multi_type(args.query, args.types, args.top_k)
    elif args.keyword:
        query.keyword_search(args.query, args.top_k, args.min_score)
    else:
        query.search(args.query, args.top_k, args.min_score)


if __name__ == "__main__":
    main()
