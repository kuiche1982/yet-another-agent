"""KnowledgeBase/ 一键注入 ContextManager（Library + Code 两线；Session 线由运行时 append 产生）。

用法：
  # 默认（KeywordDistiller，model-free，可复现）注入并跑样例 query 验证
  PYTHONPATH=. python scripts/ingest_knowledge_base.py

  # 用 lfm 走真实 LLM 蒸馏（需模型在线）
  PYTHONPATH=. python scripts/ingest_knowledge_base.py --llm

  # 只注入、不跑 demo
  PYTHONPATH=. python scripts/ingest_knowledge_base.py --no-demo

  # 指定目录 + 自定义 query
  PYTHONPATH=. python scripts/ingest_knowledge_base.py --root path/to/KB --query "KV 量化怎么选"

  # 落盘持久化（Library/Code 按 mtime 增量重建；Session 线由 save/load_session 还原）
  PYTHONPATH=. python scripts/ingest_knowledge_base.py --persist

路由规则（与 contextmgr 三源一致）：
  .md/.markdown/.txt/.rst  -> Library（资料库，ingest_library）
  .py/.pyi                 -> Code（代码，ingest_code，AST 切分）

注入后可通过 cm.retrieve(query) / cm.build_context(query, budget) 取数据。
会话线（Session）在 agent 运行时由 cm.append(...) 产生，进程退出前 cm.save_session(dir)、重启后 cm.load_session(dir)。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# 允许从仓库根运行：scripts/ 上一级加入 path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contextmgr import (  # noqa: E402
    BM25Embedder,
    CompressionStrategy,
    ContextManager,
    KeywordDistiller,
    Source,
)
from contextmgr import persist as persist_mod  # noqa: E402
from contextmgr.llm_backends import LLMDistiller  # noqa: E402

_LIB_EXT = {".md", ".markdown", ".txt", ".rst"}
_CODE_EXT = {".py", ".pyi"}

_SAMPLE_QUERIES = [
    "judge mode 怎么用，软 guard 和硬 guard 怎么选",
    "KV 量化 CTKV 怎么选，内存怎么算",
    "thinking 模式要不要关，什么时候开",
    "analyzer 和 tester 怎么分层",
]


def _iter_files(root: str):
    for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in _LIB_EXT or ext in _CODE_EXT:
            yield path, ext


def build_knowledge_base(cm: ContextManager, root: str = "KnowledgeBase") -> dict:
    """扫描 root 目录，按扩展名路由注入 Library / Code 两线。可复用函数。"""
    root = os.path.abspath(root)
    counts = {"library": 0, "code": 0, "files": 0}
    for path, ext in _iter_files(root):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError as e:  # noqa: BLE001
            print(f"  [skip] 读不了 {path}: {e}")
            continue
        rel = os.path.relpath(path, _ROOT)
        stem = os.path.splitext(os.path.basename(path))[0]
        if ext in _LIB_EXT:
            n = cm.ingest_library(text, doc_id=f"kb:{stem}")
            counts["library"] += n
            print(f"  LIBRARY  {rel}  -> {n} 个 L0 片段")
        else:
            n = cm.ingest_code(text, name=rel)
            counts["code"] += n
            print(f"  CODE    {rel}  -> {n} 个 L0 片段 (AST)")
        counts["files"] += 1
    return counts


def _show_retrieval(cm: ContextManager, query: str, budget: int = 700):
    print(f"\n>>> query: {query}")
    refs = cm.retrieve(query, budget_tokens=budget, top_k=4)
    if not refs:
        print("   (无命中 RAG，仅依赖会话/系统)")
        return
    for f in refs:
        tag = f.origin or f.source.value
        print(f"   [{tag}] +{f.tokens}t  {f.text[:90].replace(chr(10), ' ')}")
    # 组装下一轮上下文，验证总 token 不超预算
    msgs = cm.build_context(query, budget_tokens=budget)
    total = sum(len(m["content"]) // 2 for m in msgs)  # 粗估
    print(f"   -> build_context 返回 {len(msgs)} 条 messages")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="KnowledgeBase", help="知识库目录（相对/绝对）")
    ap.add_argument("--llm", action="store_true", help="用 LLM(lfm2.5-2.6b) 蒸馏，否则 KeywordDistiller")
    ap.add_argument("--model", default="lfm2.5-2.6b", help="--llm 时的模型 id")
    ap.add_argument("--budget", type=int, default=4096, help="ContextManager 总预算 token")
    ap.add_argument("--no-demo", action="store_true", help="只注入不跑样例 query")
    ap.add_argument("--query", default="", help="自定义单条 query 测试（覆盖样例）")
    ap.add_argument("--persist", action="store_true",
                    help="落盘为 markdown（增量重建：源码 mtime 未变则磁盘 load，变了重派生）")
    ap.add_argument("--cache-dir", default=os.path.join(_ROOT, ".contextmgr_store"),
                    help="落盘目录（默认 <repo>/.contextmgr_store）")
    args = ap.parse_args()

    distiller_name = f"llm:{args.model}" if args.llm else "keyword"
    cm = ContextManager(
        budget_tokens=args.budget,
        embedder=BM25Embedder(),
        strategy=CompressionStrategy.COMPRESS,
        library_distiller=(LLMDistiller(model=args.model) if args.llm else KeywordDistiller()),
    )
    cm.set_system("你是开发 agent，可检索本地知识库与代码库作答。")

    print(f"=== 注入 {args.root} (distiller={'LLM:'+args.model if args.llm else 'Keyword'}"
          f"{' | PERSIST->'+args.cache_dir if args.persist else ''}) ===")
    if args.persist:
        cm.load_session(args.cache_dir)  # 先恢复会话线（buffer + Session 片段）
        counts = persist_mod.rebuild(cm, args.root, args.cache_dir, distiller_name,
                                     _LIB_EXT, _CODE_EXT, base=_ROOT)
        print(f"=== 增量重建完成: {counts['files']} 文件, "
              f"新派生 {counts['ingested']} 片段 (Library {counts['library']}/Code {counts['code']}), "
              f"磁盘 load {counts['loaded']}, 清理 {counts['pruned']} ===")
    else:
        counts = build_knowledge_base(cm, args.root)
        print(f"=== 注入完成(内存): {counts['files']} 文件, Library {counts['library']} 片段, "
              f"Code {counts['code']} 片段 ===")
    print("stats:", cm.stats())

    if args.no_demo:
        if args.persist:
            cm.save_session(args.cache_dir)  # 落盘会话线（若有）
        return 0

    queries = [args.query] if args.query else _SAMPLE_QUERIES
    for q in queries:
        _show_retrieval(cm, q)

    if args.persist:
        cm.save_session(args.cache_dir)  # 落盘会话线（若有）
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
