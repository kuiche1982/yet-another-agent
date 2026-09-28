"""
记忆系统 - Agent的分层记忆核心
基于 BGE-M3 + 递归摘要树
"""
from .incremental_tree import IncrementalRecursiveTree, FragmentNode, SummaryNode
from .mem_classifier import MemoryClassifier
from .mem_manager import MemoryManager
from .retriever import BGERetriever
from .pipeline import SessionPipeline, import_sessions_batch

__all__ = [
    "IncrementalRecursiveTree",
    "FragmentNode",
    "SummaryNode",
    "MemoryClassifier",
    "MemoryManager",
    "BGERetriever",
    "SessionPipeline",
    "import_sessions_batch",
]

__version__ = "1.0.0"
__author__ = "litertlm"
