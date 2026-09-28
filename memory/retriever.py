"""
BGE-M3 混合检索器
支持 dense + sparse 混合检索，适配本地 BGE-M3 模型
"""
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from .incremental_tree import IncrementalRecursiveTree
from .mem_manager import MemoryManager


class BGERetriever:
    """BGE-M3 混合检索器（使用 sentence-transformers）"""

    def __init__(
        self,
        model_path: str = "~/kuiwork/workdir2/litertlm/bge-m3-mlx-6bit",
        config_path: str = "memory/config.yaml"
    ):
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.config = self._load_config(config_path)

        self.max_length = self.config.get("model_max_length", 8192)

        # BGE-M3 向量维度
        self.dense_dim = 1024

        # 权重
        self.dense_weight = self.config.get("dense_weight", 0.7)
        self.sparse_weight = self.config.get("sparse_weight", 0.3)

        # Top-K 参数
        self.top_k_recall = self.config.get("top_k_recall", 8)
        self.top_k_rerank = self.config.get("top_k_rerank", 3)

        # 加载模型
        self._load_model()

        # 初始化向量存储
        self._init_vectors()

        # 存储每个文本的类型信息（不重置，保持 _init_vectors 的结果）
        # self.text_types: Optional[List[str]] = None  # 已经在 _init_vectors 中初始化

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        path = Path(config_path)
        if path.exists():
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except ImportError:
                try:
                    import json
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except:
                    pass
        return {}

    def _load_model(self):
        """加载 BGE-M3 模型"""
        try:
            print(f"[检索器] 加载模型: {self.model_path}")
            # 使用 transformers 直接加载，避免 sentence-transformers 依赖问题
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                device_map="cpu",
                ignore_mismatched_sizes=True
            )
            self.model.eval()
            print(f"[检索器] 模型加载完成")
            print(f"[检索器] Dense embedding 维度: {self.dense_dim}")

            # 初始化向量存储
            self._init_vectors()
        except Exception as e:
            print(f"[检索器] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.tokenizer = None

    def _init_vectors(self):
        """初始化向量存储"""
        if not hasattr(self, 'text_types') or self.text_types is None:
            self.dense_vectors: List[np.ndarray] = []
            self.sparse_vectors: List[np.ndarray] = []
            self.raw_texts: List[str] = []
            self.text_types: List[str] = []

        raw_dir = Path(self.config.get("raw_dir", "memory/raw"))
        embeddings_dir = Path(self.config.get("embeddings_dir", "memory/embeddings"))

        # 清空现有数据
        self.dense_vectors = []
        self.sparse_vectors = []
        self.raw_texts = []
        self.text_types = []

        # 优先从 embeddings 目录加载
        if embeddings_dir.exists():
            print(f"[检索器] 从 embeddings 目录加载向量...")
            for mem_type_dir in embeddings_dir.iterdir():
                if mem_type_dir.is_dir():
                    # 按 numeric order 加载文件
                    npy_files = sorted(mem_type_dir.glob("*.npy"), key=lambda p: int(p.stem))
                    for file_path in npy_files:
                        try:
                            # 只加载 dense 向量
                            dense_vec = np.load(file_path)
                            print(f"[检索器] 加载向量: {file_path}, 维度: {dense_vec.shape}")

                            # 添加到数组末尾
                            self.dense_vectors.append(dense_vec)

                        except Exception as e:
                            print(f"[检索器] 加载embeddings失败 {file_path}: {e}")

            # 完成向量加载
            self._finalize_vectors()
            return

        # 如果没有 embeddings，从 raw 目录加载文本
        if raw_dir.exists():
            print(f"[检索器] 从 raw 目录加载文本...")
            for mem_type_dir in raw_dir.iterdir():
                if mem_type_dir.is_dir():
                    for file_path in mem_type_dir.glob("*.jsonl"):
                        self._load_jsonl_vectors(file_path)

            # 重新编码所有文本
            if self.raw_texts:
                print(f"[检索器] 重新编码 {len(self.raw_texts)} 个文本...")
                self.encode_documents(self.raw_texts)

    def _finalize_vectors(self):
        """完成向量加载，确保 raw_texts 和 text_types 长度匹配"""
        target_len = len(self.dense_vectors)

        # 确保 raw_texts 长度匹配
        if len(self.raw_texts) > target_len:
            self.raw_texts = self.raw_texts[:target_len]
        elif len(self.raw_texts) < target_len:
            # 用空字符串补齐
            self.raw_texts.extend([""] * (target_len - len(self.raw_texts)))

        # 确保 text_types 长度匹配
        if self.text_types is None or len(self.text_types) != target_len:
            self.text_types = ["session"] * target_len
            print(f"[检索器] text_types 用默认值补齐到 {target_len}")

    def _load_jsonl_vectors(self, file_path: Path):
        """从jsonl文件加载文本"""
        try:
            # 从文件路径提取 type
            mem_type = file_path.parent.name

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.raw_texts.append(entry["text"])
                        if self.text_types is None:
                            self.text_types = []
                        self.text_types.append(mem_type)
        except Exception as e:
            print(f"[检索器] 加载文本失败 {file_path}: {e}")

    def encode_documents(self, texts: List[str]):
        """
        编码文档

        Args:
            texts: 文本列表
        """
        if self.model is None or self.tokenizer is None:
            print("[检索器] 模型未加载，请先调用 _load_model()")
            return

        try:
            print(f"[检索器] 编码 {len(texts)} 个文档 (dense + sparse)...")

            # 使用 transformers 的 encoding 方式
            encoded_inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # 使用 CPU
            device = "cpu"
            encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                dense_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # 使用 tf-idf 计算 sparse embedding
            vectorizer = TfidfVectorizer()
            sparse_embeddings = vectorizer.fit_transform(texts).toarray()

            self.dense_vectors = dense_embeddings.tolist()
            self.sparse_vectors = sparse_embeddings.tolist()
            self.raw_texts = texts
            # 如果已有 text_types 则保留，否则创建默认类型
            if not hasattr(self, 'text_types') or self.text_types is None:
                self.text_types = ["session"] * len(texts)
            # 确保 text_types 长度匹配 raw_texts
            elif len(self.text_types) != len(texts):
                # 用默认类型补齐或截断
                if len(self.text_types) < len(texts):
                    self.text_types.extend(["session"] * (len(texts) - len(self.text_types)))
                else:
                    self.text_types = self.text_types[:len(texts)]

            # 保存 vectorizer 用于查询
            self._query_vectorizer = vectorizer

            print(f"[检索器] 编码完成!")
            print(f"  Dense维度: {len(self.dense_vectors[0])}")
            print(f"  Sparse维度: {len(self.sparse_vectors[0])}")

            # 保存 embeddings 到磁盘
            self._save_embeddings()

        except Exception as e:
            print(f"[检索器] 编码失败: {e}")
            import traceback
            traceback.print_exc()

    def _save_embeddings(self):
        """保存 embeddings 到磁盘"""
        try:
            embeddings_dir = Path(self.config.get("embeddings_dir", "memory/embeddings"))
            embeddings_dir.mkdir(parents=True, exist_ok=True)

            (embeddings_dir / "dense").mkdir(parents=True, exist_ok=True)
            (embeddings_dir / "sparse").mkdir(parents=True, exist_ok=True)

            # 按 type 分组保存
            for i, (dense_vec, sparse_vec, text) in enumerate(zip(self.dense_vectors, self.sparse_vectors, self.raw_texts)):
                # 根据 text hash 生成文件名（保持与 _hash_text 一致）
                text_hash = self._hash_text(text)
                filename = f"{text_hash[:16]}"

                # 保存 dense 向量
                dense_path = embeddings_dir / "dense" / f"{filename}.npy"
                np.save(dense_path, dense_vec)

                # 保存 sparse 向量
                sparse_path = embeddings_dir / "sparse" / f"{filename}.npy"
                np.save(sparse_path, sparse_vec)

            print(f"[检索器] 已保存 {len(self.dense_vectors)} 个 embeddings 到 {embeddings_dir}")
        except Exception as e:
            print(f"[检索器] 保存 embeddings 失败: {e}")
            import traceback
            traceback.print_exc()

    def _hash_text(self, text: str) -> str:
        """生成文本的 hash"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def encode_queries(self, queries: List[str]):
        """
        编码查询

        Args:
            queries: 查询文本列表

        Returns:
            List[Dict] 每个查询的 dense 和 sparse 向量
        """
        if self.model is None or self.tokenizer is None:
            print("[检索器] 模型未加载，请先调用 _load_model()")
            return []

        try:
            print(f"[检索器] 编码 {len(queries)} 个查询...")

            # 编码查询（使用同样的方式）
            encoded_inputs = self.tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            device = "cpu"
            encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                dense_queries = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # 使用训练好的vectorizer
            if hasattr(self, '_query_vectorizer'):
                sparse_queries = self._query_vectorizer.transform(queries).toarray()
            else:
                # 如果没有训练过，创建新的（应该很少发生）
                vectorizer = TfidfVectorizer()
                sparse_queries = vectorizer.fit_transform(queries).toarray()

            return [
                {"dense": dense_queries[i], "sparse": sparse_queries[i]}
                for i in range(len(queries))
            ]

        except Exception as e:
            print(f"[检索器] 查询编码失败: {e}")
            return []

    def hybrid_retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量（默认 top_k_recall）

        Returns:
            召回的文本列表
        """
        if top_k is None:
            top_k = self.top_k_recall

        if len(self.raw_texts) == 0:
            print("[检索器] 没有可检索的文档，请先调用 encode_documents()")
            return []

        # 编码查询
        query_vectors = self.encode_queries([query])
        if not query_vectors:
            return []

        q_dense = query_vectors[0]["dense"]
        q_sparse = query_vectors[0]["sparse"]

        # 计算 dense 分数
        dense_scores = np.array([np.dot(q_dense, vec) for vec in self.dense_vectors])

        # 计算 sparse 分数
        sparse_scores = np.array([
            self._compute_lexical_matching_score(q_sparse, sparse_vec)
            for sparse_vec in self.sparse_vectors
        ])

        # 混合打分
        hybrid_scores = self.dense_weight * dense_scores + self.sparse_weight * sparse_scores

        # Top-K 召回
        top_indices = np.argsort(-hybrid_scores)[:top_k]

        # 构造返回结果，包含类型信息
        results = []
        for idx in top_indices:
            text_type = self.text_types[idx] if idx < len(self.text_types) else "session"
            results.append({
                "text": self.raw_texts[idx],
                "score": float(hybrid_scores[idx]),
                "index": int(idx),
                "type": text_type
            })

        # 如果有 reranker，进行重排
        if hasattr(self, "reranker") and self.reranker is not None:
            results = self.reranker.rerank(query, results, self.top_k_rerank)

        return results

    def _compute_lexical_matching_score(self, q_sparse: np.ndarray, doc_sparse: np.ndarray) -> float:
        """计算 lexical matching score"""
        return np.sum(np.minimum(q_sparse, doc_sparse))

    def set_reranker(self, reranker):
        """设置 reranker（预留接口）"""
        self.reranker = reranker

    def retrieve_with_details(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        检索并返回详细信息

        Returns:
            [
                {
                    "text": str,
                    "score": float,
                    "type": str,
                    "project_id": str
                },
                ...
            ]
        """

        if top_k is None:
            top_k = self.top_k_recall

        print(f"        [retrieve] dense_vectors: {len(self.dense_vectors)} vectors")

        if len(self.raw_texts) == 0:
            return []

        # 编码查询
        query_vectors = self.encode_queries([query])
        if not query_vectors:
            return []

        print(f"        [retrieve] dense_vectors after encode: {len(self.dense_vectors)} vectors")
        q_dense = query_vectors[0]["dense"]
        q_sparse = query_vectors[0]["sparse"]

        # 计算 dense 分数
        dense_scores = np.array([np.dot(q_dense, vec) for vec in self.dense_vectors])

        # 计算 sparse 分数
        sparse_scores = np.array([
            self._compute_lexical_matching_score(q_sparse, sparse_vec)
            for sparse_vec in self.sparse_vectors
        ])

        # 混合打分
        hybrid_scores = self.dense_weight * dense_scores + self.sparse_weight * sparse_scores

        # Top-K 召回
        top_indices = np.argsort(-hybrid_scores)[:top_k]

        results = []
        for idx in top_indices:
            text_type = self.text_types[idx] if idx < len(self.text_types) else "session"
            results.append({
                "text": self.raw_texts[idx],
                "score": float(hybrid_scores[idx]),
                "index": int(idx),
                "type": text_type
            })

        # 如果有 reranker，进行重排
        if hasattr(self, "reranker") and self.reranker is not None:
            results = self.reranker.rerank(query, results, self.top_k_rerank)

        return results
