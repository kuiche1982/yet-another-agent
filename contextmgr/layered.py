#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""contextmgr —— 分层持久知识库门面（code / proj / global 三层）

本模块是「三层独立持久 RagEngine + 磁盘增量缓存 + 跨层合并语料重排」的**实现之家**，
原住在 `swe_agent/management.py`。下沉理由见 `docs/contextmgr_design.md`：分层策略、
缓存目录策略、检索重排都属于**内容管理**；`RagEngine` 本身早已在本包内。

作用域语义（四层模型，2026-09-12 定稿）：
- Global Library：**跨进程唯一**（人手动投放的跨项目文档）；缓存 `global_kb/r{i}`
- 项目 Docs/KB：**每项目唯一**；缓存 `projects/<project>/proj_kb/r{i}`
- Code Library：**每项目唯一**；缓存 `projects/<project>/code_kb`
- Session：**每对话唯一** —— 不进本类，归 `ContextManager` 管辖

⚠️ 因此本类**混装两种作用域**（code/proj = 项目级，global = 全局级），实例必须由调用方
按 project key 缓存（见 harness 的 `_get_layered_kb`），**绝不能做成进程内单例** ——
否则先索引 A 项目、再跑 B 项目会检索到 A 的内容。

边界（与 inject.py 同规）：
1. **不 import `swe_agent.config`**：根路径 / 扩展名 / 忽略集 / 重排策略全部由构造期注入。
2. **不触网络**：可选模型重排走**可注入后端** `rerank_llm(system, user) -> str | None`，
   默认 `None` → 完全不调模型（model-free）。「要不要开 / 用哪个模型 / 模型加载了没」
   一律由 harness 注入的闭包在**调用期**判定，本模块不认识任何模型。
3. **不持有模块级可变状态**：本模块只有类与纯函数（实例缓存由调用方掌管）。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

from . import persist as kb_persist
from .compress import CompressionStrategy
from .embedder import BM25Embedder
from .inject import ref_location
from .manager import ContextManager as RagEngine
from .types import Source

logger = logging.getLogger(__name__)

# 代码源只收真实代码扩展名（修复旧 bug——md/json 走 ingest_code_file 错配进 code 源）
CODE_EXTS = {".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".rs", ".c", ".cpp",
             ".h", ".hpp", ".java"}

# 知识库源只收文档扩展名
LIB_EXTS = {".md", ".markdown", ".txt", ".rst"}

# code 源忽略目录：重型目录 + 非源码/噪声目录（避免把打包文档、模型权重、运行产物当代码 ingest）
INDEX_IGNORE_DIRS = {
    ".venv", "__pycache__", ".git", "node_modules",
    "litert-lm-cache", ".pytest_tmp", ".workbuddy", "agent_sandbox",
    "workbuddyrelateddocs", "models", "LFM2.5-350M-MLX-4bit",
    "Ling-3.0-tiny", "bge-m3-mlx-6bit", "designing-ai-agents",
    "KnowledgeBase", "docs", "sessions", "logs", "memory",
    ".rag_cache", "litertlm.egg-info",
}

# proj/global 层用的「最小忽略集」：KB 根（KnowledgeBase/、docs/）本身就是库内容，
# 绝不能把根目录名塞进 ignore_dirs 否则整库被跳过（已踩坑：proj_kb 一度 ingested=0）。
KB_INGEST_IGNORE_DIRS = {".venv", "__pycache__", ".git", "node_modules", ".workbuddy"}

# 模型重排后端契约：(system, user) -> 原始模型输出；返回 None/空 = 不重排（未启用/未就绪/失败）
RerankLLM = Callable[[str, str], Optional[str]]


def parse_order(raw: str, n: int) -> list[int]:
    """严格解析模型重排输出 {"order":[...]}：只接受 1..n 的合法下标（去重、保持顺序）。

    弱模型会带 ```围栏或前后缀，故允许一次「摘出最外层 {...}」的兜底；仍解析不出、
    或没有任何合法下标 → 返回 []，由调用方保持合并 BM25 序（fail-open，绝不污染排序）。
    """
    s = (raw or "").strip()
    if not s:
        return []
    try:
        v = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            return []
        try:
            v = json.loads(m.group(0))
        except Exception:
            return []
    if not isinstance(v, dict):
        return []
    seq = v.get("order")
    if seq is None:
        seq = v.get("ranking")
    if not isinstance(seq, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for x in seq:
        try:
            i = int(x)
        except Exception:
            continue
        if 1 <= i <= n and i not in seen:
            seen.add(i)
            out.append(i)
    return out


class LayeredKB:
    """三个独立持久 RagEngine 的统一门面：被动注入与主动 local_search 都走它。

    构造参数全部显式注入（harness 从 config 取值后传入），本类不反向依赖 config。
    """

    # 默认值：让 `LayeredKB.__new__(LayeredKB)` 构造的测试桩也能安全调用 `_model_rerank`
    # （未设实例属性 → 落到下面的 property → None → 不重排）。
    rerank_candidates: int = 8

    @property
    def rerank_llm(self) -> Optional[RerankLLM]:
        """模型重排后端（默认 None = 不重排）。子类可覆写为「调用期从 config 现取」的工厂。"""
        return getattr(self, "_rerank_llm", None)

    def __init__(self, *, code_root: str, code_cache: str,
                 proj_roots=(), proj_cache_dir: Optional[str] = None,
                 global_roots=(), global_cache_dir: Optional[str] = None,
                 code_ext=None, lib_ext=None,
                 code_ignore=None, kb_ignore=None,
                 rerank_llm: Optional[RerankLLM] = None, rerank_candidates: int = 8):
        self.code_root = str(code_root)
        self.code_cache = str(code_cache)
        self.proj_roots = [str(r) for r in (proj_roots or [])]
        self.proj_cache_dir = str(proj_cache_dir) if proj_cache_dir else ""
        self.global_roots = [str(r) for r in (global_roots or [])]
        self.global_cache_dir = str(global_cache_dir) if global_cache_dir else ""
        self.code_ext = set(code_ext if code_ext is not None else CODE_EXTS)
        self.lib_ext = set(lib_ext if lib_ext is not None else LIB_EXTS)
        self.code_ignore = set(code_ignore if code_ignore is not None else INDEX_IGNORE_DIRS)
        self.kb_ignore = set(kb_ignore if kb_ignore is not None else KB_INGEST_IGNORE_DIRS)
        self._rerank_llm = rerank_llm
        self.rerank_candidates = max(1, int(rerank_candidates))

        self.code_kb = self._new_engine()
        self.proj_kb = self._new_engine()
        self.global_kb = self._new_engine()
        # library 源 origin = "kb:<stem>"，rebuild 只留 stem 丢真实路径；这里并行建 stem→真实路径索引，
        # 供 passive/local_search 的 read_file 指针解析（library 文档按整篇读）。
        self._lib_path_index: dict[str, str] = {}
        self._build()

    @staticmethod
    def _new_engine() -> RagEngine:
        return RagEngine(budget_tokens=4096, strategy=CompressionStrategy.COMPRESS)

    def _index_lib_paths(self, root) -> None:
        """并行建 stem->真实路径索引（rebuild 的 library origin 只留 stem，这里补回全路径）。"""
        try:
            for p in Path(root).rglob("*"):
                if p.is_file() and p.suffix.lower() in self.lib_ext:
                    self._lib_path_index[f"kb:{p.stem}"] = str(p)
        except Exception:
            pass

    def _lib_realpath(self, origin: str) -> str:
        return self._lib_path_index.get(origin, "")

    def _build(self) -> None:
        # code：只收真实代码扩展名（修复旧 bug——md/json 错配进 code 源），并跳过 .venv 等重型目录
        try:
            kb_persist.rebuild(self.code_kb, self.code_root, self.code_cache,
                               "code", lib_ext=set(), code_ext=self.code_ext,
                               base=self.code_root, ignore_dirs=self.code_ignore)
        except Exception as e:  # fail-open：代码索引失败不阻断 agent
            logger.info('[LayeredKB] code 构建失败（降级：无代码检索）：%s', e)
        # proj：每个根独立 cache 子目录，积攒进同一 engine（rebuild 不跨目录 prune）
        for i, root in enumerate(self.proj_roots):
            if not Path(root).exists():
                continue
            self._index_lib_paths(root)
            cache = str(Path(self.proj_cache_dir) / f"r{i}") if self.proj_cache_dir else root
            try:
                kb_persist.rebuild(self.proj_kb, str(root), cache,
                                   "proj", lib_ext=self.lib_ext, code_ext=set(),
                                   ignore_dirs=self.kb_ignore)
            except Exception as e:
                logger.info('[LayeredKB] proj 根 %s 构建失败：%s', root, e)
        # global：同上
        for i, root in enumerate(self.global_roots):
            if not Path(root).exists():
                continue
            self._index_lib_paths(root)
            cache = (str(Path(self.global_cache_dir) / f"r{i}")
                     if self.global_cache_dir else root)
            try:
                kb_persist.rebuild(self.global_kb, str(root), cache,
                                   "global", lib_ext=self.lib_ext, code_ext=set(),
                                   ignore_dirs=self.kb_ignore)
            except Exception as e:
                logger.info('[LayeredKB] global 根 %s 构建失败：%s', root, e)

    def rebuild_all(self) -> None:
        """重新增量构建：丢弃内存态后重建（就地更新，不换实例，保调用方持有的引用有效）。"""
        self.code_kb = self._new_engine()
        self.proj_kb = self._new_engine()
        self.global_kb = self._new_engine()
        self._lib_path_index = {}
        self._build()

    # —— 跨层相关性：在「合并候选池」上重算 BM25 ——
    # 各层自己的 BM25 分数不可直接比较（IDF / avgdl 依本层语料规模与分布），层内
    # max-normalize 只能给出「本层相对相关性」——副作用是**每层 top-1 必然并列 1.00**，
    # 跨层排序于是退化成「比 raw 分」，而 raw 分恰恰是不可比的（实测 proj 1.00 与
    # code 1.00 并列）。故这里把三层候选并成**一个语料**重新 fit BM25：单一 IDF / avgdl
    # 空间 → 分数天然跨层可比，「全局相关性」才名副其实。
    @staticmethod
    def _doc_for(engine, f) -> str:
        """候选的检索文档表示，与 index.Index._doc_for 同源（L3 标签 + L1 关键词）。

        拿不到 L3（例如测试的 stub engine）时退化为片段正文前缀 —— 仍是同一空间内打分。
        """
        store = getattr(engine, "store", None)
        l3 = None
        if store is not None:
            l3 = getattr(store, "_l3", {}).get(getattr(f, "fid", ""))
        if l3 is not None:
            l1 = getattr(store, "_l1", {}).get(l3.fid)
            kp = " ".join(l1.key_points) if l1 else ""
            return f"{l3.label} {kp}".strip()
        return (getattr(f, "text", "") or "")[:200]

    @classmethod
    def _merged_bm25(cls, query: str, cands) -> "list[float] | None":
        """合并候选池上重算 BM25（单一 IDF/avgdl 空间）→ 跨层可比分数。

        query 为空 / 无候选 → None（调用方回退到层内归一化）。
        """
        if not (query or "").strip() or not cands:
            return None
        docs = [cls._doc_for(engine, f) for _layer, engine, f in cands]
        emb = BM25Embedder()
        emb.fit(docs)
        return [emb.similarity(query, d) for d in docs]

    @classmethod
    def _rerank(cls, pools, query: str = ""):
        """跨层候选池 → 合并语料 BM25 重排。

        返回 [(norm, raw, layer, frag)]（按相关性降序，已按 origin+offset 去重）。
        norm = 合并分 / 本批最高分（0~1，仅供展示与稳定排序）；**排序由合并分决定**。
        query 为空时退化为「层内 max-normalize + raw 裁决」（向后兼容旧调用）。
        """
        cands, seen = [], set()
        for layer, engine, frags in pools:
            for f in (frags or []):
                key = (getattr(f, "origin", ""), getattr(f, "offset", 0))
                if key in seen:
                    continue
                seen.add(key)
                cands.append((layer, engine, f))
        if not cands:
            return []
        # 经 cls 调用（非写死基类名）：子类覆写 _merged_bm25 时才真正生效
        # （harness 的 LayeredKB 是本类的子类；写死 LayeredKB 会让覆写被静默忽略）。
        scores = cls._merged_bm25(query, cands)
        if scores is None or not any(scores):
            # 回退（无 query / 合并分全 0）：层内 max-normalize，同分用 raw 分裁决
            by_layer: dict[str, list] = {}
            for layer, _e, f in cands:
                by_layer.setdefault(layer, []).append(f)
            tops = {k: max(((getattr(f, "score", 0.0) or 0.0) for f in v), default=0.0)
                    for k, v in by_layer.items()}
            pairs = []
            for layer, _e, f in cands:
                raw = getattr(f, "score", 0.0) or 0.0
                top_l = tops.get(layer, 0.0)
                pairs.append(((raw / top_l) if top_l > 0 else 0.0, raw))
        else:
            pairs = [(s, s) for s in scores]   # 合并语料分：排序键与展示分同源（跨层可比）
        top = max((p[0] for p in pairs), default=0.0) or 1.0
        out = [(p[0] / top, p[1], cands[i][0], cands[i][2]) for i, p in enumerate(pairs)]
        out.sort(key=lambda x: (-x[0], -x[1]))
        return out

    # 可选：调模型对合并候选**再确定一次相关性**（「必要时可以调用模型」）。
    # 契约：只喂前 N 条候选的「出处 + 一行摘要」，要求输出 {"order":[下标...]}（最相关在前）。
    # 三重防线：惰性门控（后端未注入即跳过）+ 严格校验（合法下标排列/子集）+ fail-open
    # （任何异常或解析失败 → 保持 BM25 序）。出处与分数由代码写入，不经模型。
    def _model_rerank(self, query: str, ranked: list) -> list:
        try:
            llm = self.rerank_llm
            if llm is None or not ranked or len(ranked) < 3:
                return ranked
            head = ranked[:max(1, int(self.rerank_candidates))]
            lines = []
            for i, (_norm, _raw, layer, f) in enumerate(head, 1):
                loc, _ptr = ref_location(self, f)
                text = re.sub(r"\s+", " ", (getattr(f, "text", "") or "")).strip()[:160]
                lines.append(f"[{i}] ({layer}) {loc}\n{text}")
            system = "你是检索结果排序器。只输出严格 JSON，不要开场白、不要 markdown 代码块围栏。"
            user = (f"用户问题：{query}\n\n候选片段：\n" + "\n".join(lines) +
                    f"\n\n按与用户问题的相关性从高到低排序，只输出 "
                    f'{{"order":[最相关的编号,...]}}，须包含 1~{len(head)} 全部编号且不重复。')
            raw_out = llm(system, user)      # 后端自决「未启用/未就绪/失败 → None」
            if not raw_out:
                return ranked
            order = parse_order(raw_out, len(head))
            if not order:
                return ranked
            picked = [head[i - 1] for i in order]
            chosen = set(order)
            picked += [item for i, item in enumerate(head, 1) if i not in chosen]
            picked += ranked[len(head):]
            return picked
        except Exception as e:  # fail-open：重排失败绝不影响检索可用性
            logger.debug('RAG 模型重排失败（保持合并 BM25 序）：%s', e)
            return ranked

    # 被动注入用：跨三层聚合检索 → 合并语料重排 → （可选模型重排）→ 按预算贪婪填充。
    def retrieve(self, query, budget_tokens: int = 2048, sources=None, per_layer_k: int = 8):
        """跨三层聚合检索 → 合并语料 BM25 重排 → 按 token 预算贪婪填充。

        与旧行为的两点差别：① 不再按 code→proj→global 固定层序遍历拼接（那会让 code 层
        轻量命中压住 kb 层高相关文档）；② 不再依赖「层内归一化」（那会让每层 top-1 并列
        1.00、跨层退化为比 raw 分）。现在三层候选并成一个 BM25 语料重算分数 → 跨层可比。
        返回的 Fragment 上 `score` = 归一化相关性（0~1，可跨层比较）。
        """
        budget = int(budget_tokens or 2048)
        pools = []
        for layer, engine, srcs in (
                ("code", self.code_kb, {Source.CODE}),
                ("proj", self.proj_kb, {Source.LIBRARY}),
                ("global", self.global_kb, {Source.LIBRARY})):
            try:
                pools.append((layer, engine, engine.retrieve(
                    query, budget_tokens=budget, sources=srcs, top_k=per_layer_k)))
            except Exception:
                pools.append((layer, engine, []))
        ranked = self._model_rerank(query, self._rerank(pools, query))
        out, used = [], 0
        for norm, _raw, _layer, f in ranked:
            cost = getattr(f, "tokens", 0) or 0
            if used + cost > budget and out:
                break
            f.score = norm   # 下游按归一化相关性展示/排序（可跨层比较）
            out.append(f)
            used += cost
        return out

    def search(self, query, scope: str = "all", top_k: int = 10, offset: int = 0):
        """按 scope 检索并按相关性返回 top_k 条。

        - all：三层各取候选池，**合并语料 BM25 重排后取全局 top_k**（相关性优先，无固定层配额）；
        - code / kb / global：单层内按自身相关性取 top_k。
        - offset：分页偏移（供 agent 多轮细化：先看前 k 条，再要下一批）。
        返回 [(layer, fragment)]，fragment.score 为归一化相关性（0~1）。
        """
        top_k = max(1, int(top_k or 10))
        offset = max(0, int(offset or 0))
        pool_k = max(top_k + offset, 8)
        layers = {
            "code":   [("code", self.code_kb, {Source.CODE})],
            "kb":     [("proj", self.proj_kb, {Source.LIBRARY})],
            "global": [("global", self.global_kb, {Source.LIBRARY})],
        }.get(scope) or [
            ("code", self.code_kb, {Source.CODE}),
            ("proj", self.proj_kb, {Source.LIBRARY}),
            ("global", self.global_kb, {Source.LIBRARY}),
        ]
        pools = []
        for layer, engine, srcs in layers:
            try:
                pools.append((layer, engine, engine.retrieve(
                    query, budget_tokens=4096, sources=srcs, top_k=pool_k)))
            except Exception:
                pools.append((layer, engine, []))
        ranked = self._model_rerank(query, self._rerank(pools, query))[offset:offset + top_k]
        results: list = []
        for norm, _raw, layer, f in ranked:
            f.score = norm
            results.append((layer, f))
        return results

    # 兼容旧接口：ingest_* 委托到对应层（分层索引在构造期已建好，运行时一般无需再 ingest）
    def ingest_code_file(self, path: str, name=None):
        return self.code_kb.ingest_code_file(path, name=name)

    def ingest_library(self, text: str, doc_id: str = "doc"):
        return self.proj_kb.ingest_library(text, doc_id=doc_id)
