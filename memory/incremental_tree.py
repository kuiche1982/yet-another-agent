"""
增量递归摘要树
支持碎片化输入的分层摘要聚合
"""
import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class FragmentNode:
    """叶子节点：原始碎片"""
    text: str
    timestamp: str
    project_id: str
    memory_type: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SummaryNode:
    """摘要节点：压缩后的摘要"""
    is_summary: bool
    text: str
    children: List[Any]  # 可以是FragmentNode或SummaryNode
    timestamp: str
    level: int  # 0=叶子, 1=事件摘要, 2=项目摘要

    def to_dict(self):
        if self.is_summary:
            return {
                "is_summary": True,
                "text": self.text,
                "children_count": len(self.children),
                "timestamp": self.timestamp,
                "level": self.level
            }
        else:
            return {
                "is_summary": False,
                "text": self.text[:100] + "...",
                "timestamp": self.timestamp,
                "memory_type": self.memory_type
            }


class IncrementalRecursiveTree:
    """增量递归摘要树"""

    def __init__(
        self,
        root_path: str = "memory/summaries",
        group_size: int = 2,
        summary_max_tokens: int = 700
    ):
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.group_size = group_size
        self.summary_max_tokens = summary_max_tokens

        # 初始化树结构
        self.root: Optional[SummaryNode] = None
        self.leaf_nodes: List[FragmentNode] = []
        self._load_from_disk()

    def add_fragment(self, fragment: FragmentNode):
        """增量添加碎片，自动向上合并摘要"""
        # 添加到叶子节点列表
        self.leaf_nodes.append(fragment)

        # 创建新的叶子节点
        new_leaf = SummaryNode(
            is_summary=False,
            text=fragment.text,
            children=[fragment],
            timestamp=fragment.timestamp,
            level=0
        )

        # 向上合并
        current_node = new_leaf
        while True:
            parent = self._find_parent_for_new_child(current_node)

            if parent:
                # 收集需要合并的叶子节点（跳过已经是摘要的节点）
                leaf_nodes_to_merge = [c for c in parent.children if not c.is_summary]

                # 检查是否需要生成父摘要
                if len(leaf_nodes_to_merge) >= self.group_size:
                    # 生成摘要（只合并叶子节点，不包含已有的摘要）
                    parent_text = self._merge_leaf_nodes(leaf_nodes_to_merge)
                    if len(parent_text) > self.summary_max_tokens:
                        parent_text = parent_text[:self.summary_max_tokens]

                    parent.is_summary = True
                    parent.text = parent_text
                    parent.level += 1

                    # 清空children，只保留合并后的摘要
                    parent.children = [parent]

                    # 如果父节点就是根，需要新建根
                    if parent is self.root:
                        new_root = SummaryNode(
                            is_summary=True,
                            text=parent_text,
                            children=[parent],
                            timestamp=fragment.timestamp,
                            level=1
                        )
                        self.root = new_root
                        break

                    current_node = parent
                else:
                    # 还没凑够组数，停止
                    break
            else:
                # 没有父节点，创建新根（当前节点就是根）
                current_node.is_summary = True
                current_node.text = fragment.text
                current_node.level = 1
                self.root = current_node
                break

    def _find_parent_for_new_child(self, child: SummaryNode) -> Optional[SummaryNode]:
        """查找新节点的父节点（简化实现：全部挂载到根）"""
        if self.root is None:
            # 创建新的根节点，并将新节点作为第一个子节点
            self.root = SummaryNode(
                is_summary=False,
                text="",
                children=[child],
                timestamp=child.timestamp,
                level=0
            )
        else:
            # 将新节点添加到根节点的children中
            self.root.children.append(child)
        return self.root

    def _merge_leaf_nodes(self, leaf_nodes: List[FragmentNode]) -> str:
        """合并叶子节点为摘要文本（不带[摘要]前缀）"""
        joined = "\n\n----分片分隔----\n\n".join(
            node.text for node in leaf_nodes
        )
        return joined

    def _merge_children_text(self, children: List[Any]) -> str:
        """合并多个子节点为摘要文本（包含摘要节点的标识）"""
        joined = "\n\n----分片分隔----\n\n".join(
            c.text if not c.is_summary else f"[摘要] {c.text}"
            for c in children
        )
        return joined

    def get_global_summary(self) -> str:
        """获取当前全局高层摘要"""
        if self.root is None or (self.root.is_summary and not self.root.children):
            return ""
        return self.root.text if self.root.is_summary else ""

    def get_all_leaf_text(self) -> List[str]:
        """获取所有原始碎片文本（用于BGE-M3编码）"""
        return [node.text for node in self.leaf_nodes]

    def save_to_disk(self):
        """持久化到磁盘"""
        if self.root is None:
            return

        # 保存叶子节点
        leaf_file = self.root_path / "leaves.jsonl"
        with leaf_file.open("w", encoding="utf-8") as f:
            for leaf in self.leaf_nodes:
                f.write(json.dumps(asdict(leaf), ensure_ascii=False) + "\n")

        # 保存根摘要
        root_file = self.root_path / "root_summary.json"
        root_data = {
            "text": self.root.text,
            "timestamp": self.root.timestamp,
            "level": self.root.level
        }
        with root_file.open("w", encoding="utf-8") as f:
            json.dump(root_data, f, ensure_ascii=False, indent=2)

    def _load_from_disk(self):
        """从磁盘加载"""
        leaf_file = self.root_path / "leaves.jsonl"
        root_file = self.root_path / "root_summary.json"

        if leaf_file.exists():
            self.leaf_nodes = []
            for line in leaf_file.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    self.leaf_nodes.append(FragmentNode(**json.loads(line)))

        if root_file.exists():
            root_data = json.loads(root_file.read_text(encoding="utf-8"))
            self.root = SummaryNode(
                is_summary=True,
                text=root_data["text"],
                children=[],
                timestamp=root_data["timestamp"],
                level=root_data["level"]
            )

    def get_summary_depth(self) -> int:
        """获取摘要树的深度"""
        if self.root is None:
            return 0
        return self.root.level

    def clear(self):
        """清空树结构"""
        self.root = None
        self.leaf_nodes = []
        if self.root_path.exists():
            shutil.rmtree(self.root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
