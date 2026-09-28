"""
记忆管理器 - 6类记忆的统一管理
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from .incremental_tree import FragmentNode, IncrementalRecursiveTree
from .mem_classifier import MemoryClassifier


class MemoryManager:
    """记忆管理器"""

    # 有效类型
    VALID_TYPES = ["code", "knowledge", "facts", "config", "session", "task"]

    def __init__(
        self,
        config_path: str = "memory/config.yaml",
        enable_auto_classify: bool = True
    ):
        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化组件
        self.classifier = MemoryClassifier(
            use_llm=self.config.get("use_llm_classifier", True),
            model_path=self.config.get("llm_model_path", "")
        )

        self.auto_classify = enable_auto_classify

        # 初始化 SQLite 元数据
        self.metadata_db = Path(self.config.get("metadata_db", "memory/metadata.db"))
        self._init_metadata_db()

        # 初始化递归摘要树
        self.summary_tree = IncrementalRecursiveTree(
            root_path=self.config.get("summaries_dir", "memory/summaries"),
            group_size=self.config.get("summary_group_size", 2),
            summary_max_tokens=self.config.get("summary_max_tokens", 700)
        )

        # 加载现有数据
        self._load_existing_data()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        path = Path(config_path)
        if path.exists():
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except ImportError:
                # 如果没有 yaml 库，尝试 json
                try:
                    import json
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except:
                    pass
        return {}

    def _init_metadata_db(self):
        """初始化SQLite数据库"""
        self.metadata_db.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fragments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_hash TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    project_id TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    is_indexed BOOLEAN DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_type ON fragments(type)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_project ON fragments(project_id)
            """)

    def _save_raw(self, text: str, mem_type: str, project_id: str, **kwargs) -> Path:
        """保存原始碎片到raw目录"""
        raw_dir = Path(self.config.get("raw_dir", "memory/raw"))
        type_dir = raw_dir / mem_type
        type_dir.mkdir(parents=True, exist_ok=True)

        # 生成唯一ID
        text_hash = self._hash_text(text)

        # 保存到文件
        file_path = type_dir / f"{text_hash[:16]}.jsonl"
        with file_path.open("a", encoding="utf-8") as f:
            entry = {
                "text": text,
                "type": mem_type,
                "project_id": project_id,
                "timestamp": kwargs.get("timestamp", datetime.now().isoformat()),
                "metadata": kwargs
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return file_path

    def _hash_text(self, text: str) -> str:
        """计算文本hash（简化版）"""
        import hashlib
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    def add_fragment(
        self,
        text: str,
        type: Optional[str] = None,
        project_id: Optional[str] = None,
        add_to_summary: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        添加记忆碎片

        Args:
            text: 碎片文本
            type: 类型（可选）
            project_id: 项目ID
            add_to_summary: 是否添加到摘要树（默认True）
            **kwargs: 其他元数据

        Returns:
            {
                "success": bool,
                "type": str,
                "text_hash": str,
                "path": str,
                "method": str
            }
        """
        if type is None and self.auto_classify:
            # 自动分类
            result = self.classifier.classify_auto(text)

            if len(result["candidates"]) > 1:
                print(f"[自动分类] 文本可能属于: {result['candidates']}")
                print(f"请指定类型，或直接确认一个")
                # 默认使用第一个候选
                type = result["candidates"][0]
            else:
                type = result["primary"]

        # 类型校验
        if type not in self.VALID_TYPES:
            raise ValueError(f"无效类型: {type}，可选: {self.VALID_TYPES}")

        # 保存原始碎片
        text_hash = self._hash_text(text)
        file_path = self._save_raw(text, type, project_id or "default", **kwargs)

        # 保存到元数据库
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO fragments (text_hash, type, project_id, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (text_hash, type, project_id, datetime.now().isoformat(), json.dumps(kwargs))
            )

        # 添加到递归摘要树（仅当需要时）
        if add_to_summary:
            fragment = FragmentNode(
                text=text,
                timestamp=datetime.now().isoformat(),
                project_id=project_id or "default",
                memory_type=type,
                metadata=kwargs
            )
            self.summary_tree.add_fragment(fragment)
            self.summary_tree.save_to_disk()

        return {
            "success": True,
            "type": type,
            "text_hash": text_hash,
            "path": str(file_path),
            "method": "auto" if self.auto_classify and type is None else "manual"
        }

    def get_global_summary(self, project_id: Optional[str] = None) -> str:
        """获取全局摘要"""
        return self.summary_tree.get_global_summary()

    def get_summary_depth(self) -> int:
        """获取摘要树深度"""
        return self.summary_tree.get_summary_depth()

    def list_types(self) -> List[str]:
        """列出所有记忆类型"""
        return self.VALID_TYPES

    def _load_existing_data(self):
        """加载现有数据（从raw目录）"""
        raw_dir = Path(self.config.get("raw_dir", "memory/raw"))
        if not raw_dir.exists():
            return

        for mem_type_dir in raw_dir.iterdir():
            if mem_type_dir.is_dir():
                for file_path in mem_type_dir.glob("*.jsonl"):
                    self._load_jsonl(file_path, mem_type_dir.name)

    def _load_jsonl(self, file_path: Path, mem_type: str):
        """从jsonl文件加载数据"""
        try:
            for line in file_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    fragment = FragmentNode(
                        text=entry["text"],
                        timestamp=entry.get("timestamp", datetime.now().isoformat()),
                        project_id=entry.get("project_id", "default"),
                        memory_type=mem_type,
                        metadata=entry.get("metadata", {})
                    )
                    self.summary_tree.add_fragment(fragment)
            self.summary_tree.save_to_disk()
        except Exception as e:
            print(f"[记忆管理器] 加载 {file_path} 失败: {e}")

    def clear(self):
        """清空所有记忆"""
        # 清空目录
        raw_dir = Path(self.config.get("raw_dir", "memory/raw"))
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # 清空数据库
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("DELETE FROM fragments")

        # 重置摘要树
        self.summary_tree.clear()
