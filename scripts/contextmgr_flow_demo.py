"""contextmgr 端到端流程验证（可选，依赖真模型 lfm2.5-2.6b nothinking）。

这不是单测套件的一部分——单测全 model-free。本脚本证明：
  Library 文档 -> (lfm 蒸馏) L1/L2/L3 -> 索引 -> 查询 -> 回落 L0 -> 组装 context
  Code -> (AST, model-free) L1/L2/L3 -> 同样参与
  Session -> 高优先级 L0 -> 永远进上下文

用法: python scripts/contextmgr_flow_demo.py
"""

from __future__ import annotations

from contextmgr import BM25Embedder, CompressionStrategy, ContextManager, Source
from contextmgr.llm_backends import LLMDistiller


def main() -> None:
    cm = ContextManager(
        budget_tokens=900,
        strategy=CompressionStrategy.AGGRESSIVE,
        embedder=BM25Embedder(),
        library_distiller=LLMDistiller(model="lfm2.5-2.6b"),
    )

    # 1) Library 源：真实文档片段，走 lfm 蒸馏
    doc = (
        "Redis 作为缓存层时存在三大问题：缓存击穿（热点 key 失效瞬间大量请求打到 DB）、"
        "缓存穿透（查询不存在的数据，缓存与 DB 都没有）、缓存雪崩（大量 key 同时过期）。"
        "击穿可用互斥锁或逻辑过期；穿透用布隆过滤器或缓存空值；雪崩用随机过期时间或集群分片。"
    )
    n_lib = cm.ingest_library(doc, doc_id="redis_cache")
    print(f"[Library] 切出 {n_lib} 个 L0 片段")

    # 2) Code 源：AST 蒸馏（model-free）
    code = (
        "def get_user(uid):\n"
        "    cached = redis.get(f'user:{uid}')\n"
        "    if cached:\n"
        "        return json.loads(cached)\n"
        "    row = db.query(uid)\n"
        "    redis.setex(f'user:{uid}', 300, json.dumps(row))\n"
        "    return row\n"
    )
    n_code = cm.ingest_code(code, name="cache_layer.py")
    print(f"[Code] 切出 {n_code} 个 L0 片段 (AST, model-free)")

    # 3) Session 源：当前会话高优先级
    cm.append("user", "用户服务怎么防缓存击穿？")
    cm.append("assistant", "可以用互斥锁或逻辑过期。")
    cm.append("user", "布隆过滤器解决的是哪种？")

    # 4) 展示蒸馏产物（验证 lfm 真吐了 L1/L2/L3）
    print("\n=== L3 索引（最简略分级，可向下搜索）===")
    for l3 in cm.store.all_l3():
        used = l3.engine_meta.get("distiller", "keyword") if l3.engine_meta else "keyword"
        print(f"  [{l3.source.value:7s}] {l3.label[:70]!r}  (distill={used}, ptr={l3.l0_pointer[0]})")

    # 5) 查询 -> 检索 -> 回落 L0 -> 组装
    query = "缓存击穿 怎么处理"
    msgs = cm.build_context(query, budget_tokens=600)
    print(f"\n=== build_context(query={query!r}) -> {len(msgs)} 条消息, 会话策略 AGGRESSIVE ===")
    for m in msgs:
        role = m["role"]
        c = m["content"]
        head = c[:200].replace("\n", " ⏎ ")
        print(f"  {role:9s} | {head}")

    # 6) 验证 L0 是 Truth：被召回的 Library/Code 内容必须来自原始 L0 片段
    lib_fid = next((l3.l0_pointer[0] for l3 in cm.store.all_l3()
                    if l3.source is Source.LIBRARY), None)
    lib0 = cm.store.get_fragment(lib_fid) if lib_fid else None
    print("\n=== 验证 L0 为唯一 Truth（回落原文）===")
    print(f"  {lib_fid} 前 60 字 = {lib0.text[:60]!r}")
    print(f"  stats = {cm.stats()}")


if __name__ == "__main__":
    main()
