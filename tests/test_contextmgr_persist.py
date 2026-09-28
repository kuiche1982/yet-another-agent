"""contextmgr.persist —— model-free / tool-free 单测。

覆盖：L0/L1/L2 落盘 roundtrip（L3 不保存、加载时由 L1 重建）、
按源文件 mtime 增量重建（未变磁盘 load、变了重派生）、聚合 L2/L3 产物。
不引 LLM、不引网络。
"""

import json
import os
import tempfile

from contextmgr import ContextManager, BM25Embedder, KeywordDistiller, Source
from contextmgr import persist as P
from contextmgr.types import Fragment, L1Structured, L2Visual, L3Index


def _make_frag(fid="kb:t.md#p0", text="Redis 击穿用互斥锁。", origin="kb:t.md"):
    return Fragment(fid=fid, source=Source.LIBRARY, text=text, origin=origin,
                    priority=2, tokens=10, offset=0, length=len(text))


def _make_l1(fid="kb:t.md#p0"):
    return L1Structured(fid=fid, summary="Redis 击穿解法", key_points=["互斥锁", "逻辑过期"],
                        entities=["Redis", "互斥锁"], merged_with_l0=False)


def _make_l2(fid="kb:t.md#p0"):
    return L2Visual(fid=fid, diagram="flowchart LR\n  A[击穿] --> B[互斥锁]", tokens=8)


def test_roundtrip_l0_l1_l2_and_l3_rebuilt():
    with tempfile.TemporaryDirectory() as d:
        frag, l1, l2 = _make_frag(), _make_l1(), _make_l2()
        P.save_fragment(d, frag, l1, l2, src_mtime=1234.0, distiller="keyword")

        store = P.load_store(d)
        # L0 是 Truth，必须原样回来
        got = store.get_fragment(frag.fid)
        assert got is not None
        assert got.text == frag.text
        assert got.origin == "kb:t.md"
        # L1 / L2 回来
        assert store._l1[frag.fid].summary == "Redis 击穿解法"
        assert "互斥锁" in store._l2[frag.fid].diagram
        # L3 不落盘，但加载时由 L1 重建
        l3 = store.get_l3(frag.fid)
        assert l3 is not None
        assert l3.label == "Redis 击穿解法"          # 来自 L1.summary
        assert l3.l0_pointer == frag.l0_pointer()
        # 落盘文件不含 L3 段
        md = open(os.path.join(d, "kb_t.md_p0.md"), encoding="utf-8").read()
        assert "## L3 Index" not in md
        assert "## L2 Visual" in md and "```mermaid" in md


def test_incremental_rebuild_skips_unchanged():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        fpath = os.path.join(src, "doc.md")
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("Redis 击穿用互斥锁。\n\n缓存雪崩用随机过期。")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        calls = {"n": 0}

        def spy(text, doc_id="doc"):
            calls["n"] += 1
            return cm.ingest_library.__wrapped__(text, doc_id) if hasattr(cm.ingest_library, "__wrapped__") else cm.ingest_library(text, doc_id)

        # 包一层计数
        orig = cm.ingest_library
        cm.ingest_library = lambda text, doc_id="doc": (calls.__setitem__("n", calls["n"] + 1) or orig(text, doc_id))

        # 第一次：全量派生
        s1 = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert s1["ingested"] >= 1 and s1["loaded"] == 0
        assert calls["n"] == 1

        # 第二次（文件未变）：应磁盘 load，不重蒸馏
        s2 = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert s2["loaded"] >= 1 and s2["ingested"] == 0
        assert calls["n"] == 1          # 没再调 ingest -> 证明增量命中


def test_incremental_rebuild_rederives_on_change():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        fpath = os.path.join(src, "doc.md")
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("初版内容。")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        calls = {"n": 0}
        orig = cm.ingest_library
        cm.ingest_library = lambda text, doc_id="doc": (calls.__setitem__("n", calls["n"] + 1) or orig(text, doc_id))

        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert calls["n"] == 1

        # 改内容 + 把 mtime 推到未来
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("改后内容，新增一句。")
        os.utime(fpath, (10 ** 10, 10 ** 10))  # 10**9 = 2001 是过去，必须是未来才触发重派生

        s = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert s["ingested"] >= 1       # 变了 -> 重派生
        assert calls["n"] == 2


def test_incremental_rebuild_rederives_on_mtime_rollback():
    """mtime 回拨（git checkout 旧版 / rsync -t 保 mtime）也要能检出变更。

    纯 mtime 判定会漏：cached.mtime >= mtime 命中 → 走磁盘 load。
    加了 sha256 + size 兜底后，内容变了就该重派生。
    """
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        fpath = os.path.join(src, "doc.md")
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("初版内容。")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        calls = {"n": 0}
        orig = cm.ingest_library
        cm.ingest_library = lambda text, doc_id="doc": (calls.__setitem__("n", calls["n"] + 1) or orig(text, doc_id))

        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert calls["n"] == 1

        # 改内容 + 把 mtime **回拨到过去**（模拟 git checkout / rsync -t）
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("回拨版内容，和初版不同。")
        os.utime(fpath, (10 ** 9, 10 ** 9))  # 2001-09-09，比缓存里的 mtime 还旧

        s = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert s["ingested"] >= 1       # mtime 虽旧，但 sha 变了 -> 必须重派生
        assert calls["n"] == 2


def test_incremental_rebuild_skips_when_only_mtime_changed():
    """只碰 mtime、内容没变（touch / 编辑器保存无改动）→ 不该重派生。"""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        fpath = os.path.join(src, "doc.md")
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("内容始终不变。")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        calls = {"n": 0}
        orig = cm.ingest_library
        cm.ingest_library = lambda text, doc_id="doc": (calls.__setitem__("n", calls["n"] + 1) or orig(text, doc_id))

        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert calls["n"] == 1

        # 只 touch：mtime 推到未来，内容一字未改
        os.utime(fpath, (10 ** 10, 10 ** 10))

        s = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert s["loaded"] >= 1 and s["ingested"] == 0
        assert calls["n"] == 1          # sha + size 都吻合 -> 走磁盘 load


def test_frontmatter_carries_source_fingerprint():
    """有源文件时 frontmatter 必须带文件级 src_sha256 + src_size。"""
    with tempfile.TemporaryDirectory() as d:
        frag, l1, l2 = _make_frag(), _make_l1(), _make_l2()
        full = "Redis 击穿用互斥锁。\n\n缓存雪崩用随机过期。"
        P.save_fragment(d, frag, l1, l2, src_mtime=1234.0, distiller="keyword",
                        full_text=full)
        md = open(os.path.join(d, "kb_t.md_p0.md"), encoding="utf-8").read()
        import hashlib
        assert hashlib.sha256(full.encode("utf-8")).hexdigest() in md
        assert f"src_size: {len(full)}" in md


def test_aggregate_outputs_present():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        with open(os.path.join(src, "doc.md"), "w", encoding="utf-8") as fh:
            fh.write("Redis 击穿用互斥锁。")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)

        l2 = open(os.path.join(cache, "L2_overview.md"), encoding="utf-8").read()
        l3 = open(os.path.join(cache, "L3_index.md"), encoding="utf-8").read()
        assert "```mermaid" in l2 and "mindmap" in l2   # L2 脑图（mermaid 介质）
        assert "L3 Index" in l3 and "=>" in l3          # L3 纯文本索引 + 指针
        assert "```mermaid" not in l3                    # L3 非 mermaid


def test_code_source_incremental_on_code_change():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        fpath = os.path.join(src, "m.py")
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("def foo():\n    return 1\n")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        s1 = P.rebuild(cm, src, cache, "keyword", set(), {".py"}, base=src)
        assert s1["code"] >= 1

        # 改代码
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("def foo():\n    return 2\n\ndef bar():\n    return 3\n")
        os.utime(fpath, (10 ** 10, 10 ** 10))  # 10**9 = 2001 是过去，必须是未来才触发重派生
        s2 = P.rebuild(cm, src, cache, "keyword", set(), {".py"}, base=src)
        assert s2["code"] >= 2       # 代码变了 -> 重派生（增量重建）


def _boom_scan(_cache_dir):
    raise AssertionError("single-file diff must not rescan every cached fragment")


def test_single_file_change_never_rescans_fragments(monkeypatch):
    """A one-file edit must be an in-place diff, never a full fragment scan.

    Guards the 2026-09-12 fix: one edited file invalidated the whole manifest and fell back
    to _scan_cache, which reads back every cached .md fragment (50-140s when an agent edits
    code between retrievals). It must now only re-derive the edited file.
    """
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        a = os.path.join(src, "a.md")
        with open(a, "w", encoding="utf-8") as fh:
            fh.write("alpha doc about caching.")
        with open(os.path.join(src, "b.md"), "w", encoding="utf-8") as fh:
            fh.write("beta doc about queues.")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert os.path.isfile(os.path.join(cache, ".store.pkl"))
        assert os.path.isfile(os.path.join(cache, ".manifest.json"))

        with open(a, "w", encoding="utf-8") as fh:
            fh.write("alpha doc about caching, now rewritten.")

        monkeypatch.setattr(P, "_scan_cache", _boom_scan)
        s = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)

        assert s["ingested"] == 1, f"only the edited file should be re-derived: {s}"
        assert s["loaded"] >= 1, s
        assert any("rewritten" in f.text for f in cm.store.all_fragments())
        assert any("queues" in f.text for f in cm.store.all_fragments())   # untouched file kept


def test_touch_only_change_reuses_pickle_and_rederives_nothing(monkeypatch):
    """touch (mtime moves, content identical) must reuse the pickle and re-derive nothing."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        fpath = os.path.join(src, "doc.md")
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("Redis 击穿用互斥锁。")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        os.utime(fpath, (10 ** 10, 10 ** 10))

        monkeypatch.setattr(P, "_scan_cache", _boom_scan)
        s = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert s["ingested"] == 0 and s["loaded"] >= 1, s


def test_deleted_file_is_pruned_without_fragment_scan(monkeypatch):
    """A removed source file must be pruned in place, without a fragment rescan."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        with open(os.path.join(src, "a.md"), "w", encoding="utf-8") as fh:
            fh.write("alpha doc about caching.")
        b = os.path.join(src, "b.md")
        with open(b, "w", encoding="utf-8") as fh:
            fh.write("beta doc about queues.")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        os.remove(b)

        monkeypatch.setattr(P, "_scan_cache", _boom_scan)
        s = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)

        assert s["pruned"] >= 1, s
        assert not any("queues" in f.text for f in cm.store.all_fragments())
        assert any("caching" in f.text for f in cm.store.all_fragments())


def _rewrite_manifest_version(cache: str, version) -> None:
    """Rewrite the manifest's indexer version. version=None drops the key (pre-gate cache shape)."""
    path = os.path.join(cache, ".manifest.json")
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    if version is None:
        man.pop("version", None)
    else:
        man["version"] = version
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh)


def test_legacy_manifest_without_version_forces_full_rederive():
    """A cache predating the indexer-version gate must be re-derived once, in full.

    Guards the 2026-09-12 finding: incremental judging only compares source-file fingerprints,
    so a change to the derivation logic (e.g. adding Library lineno) could never reach files
    that were not edited. A manifest carrying no version reads as 0, which must be treated as
    stale -- even though every file fingerprint still matches.
    """
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        with open(os.path.join(src, "doc.md"), "w", encoding="utf-8") as fh:
            fh.write("Redis cache breakdown is solved with a mutex.")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        calls = {"n": 0}
        orig = cm.ingest_library
        cm.ingest_library = lambda text, doc_id="doc": (calls.__setitem__("n", calls["n"] + 1) or orig(text, doc_id))

        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert calls["n"] == 1

        _rewrite_manifest_version(cache, None)

        s = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert s["schema_rebuild"] == 1, s
        assert calls["n"] == 2, "legacy manifest must re-derive despite unchanged content"
        assert s["ingested"] >= 1, s


def test_older_manifest_version_forces_full_rederive():
    """An explicitly older indexer version must be treated exactly like a legacy manifest."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        with open(os.path.join(src, "doc.md"), "w", encoding="utf-8") as fh:
            fh.write("Content that never changes.")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        calls = {"n": 0}
        orig = cm.ingest_library
        cm.ingest_library = lambda text, doc_id="doc": (calls.__setitem__("n", calls["n"] + 1) or orig(text, doc_id))

        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert calls["n"] == 1

        _rewrite_manifest_version(cache, P.SCHEMA_VERSION - 1)

        s = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert s["schema_rebuild"] == 1, s
        assert calls["n"] == 2, "older schema version must re-derive despite unchanged content"


def test_matching_manifest_version_keeps_incremental_reuse():
    """Same version -> fast/incremental path stays intact and nothing is re-derived."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        with open(os.path.join(src, "doc.md"), "w", encoding="utf-8") as fh:
            fh.write("Content that never changes.")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        calls = {"n": 0}
        orig = cm.ingest_library
        cm.ingest_library = lambda text, doc_id="doc": (calls.__setitem__("n", calls["n"] + 1) or orig(text, doc_id))

        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)

        s = P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert s["schema_rebuild"] == 0, s
        assert s["ingested"] == 0, s
        assert s["loaded"] >= 1, s
        assert calls["n"] == 1


def test_gate_rebuild_records_current_schema_version():
    """After a version-gated rebuild the manifest must declare the current indexer version."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        with open(os.path.join(src, "doc.md"), "w", encoding="utf-8") as fh:
            fh.write("Content that never changes.")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        _rewrite_manifest_version(cache, None)

        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)

        version, items = P._load_manifest(os.path.join(cache, ".manifest.json"))
        assert version == P.SCHEMA_VERSION, version
        assert items is not None and len(items) >= 1, items


def test_shrinking_source_file_sweeps_orphan_fragment():
    """A shrunken source file must not leave .md files behind for fids it no longer yields.

    A doc going from two chunks to one orphans kb:doc#p1; without the sweep it stays on disk
    and load_store / _scan_cache would read it back as a live fragment (a ghost hit pointing
    at text that no longer exists).

    A version rewrite forces the reconcile path here (the incremental path already removes
    the file when it drops the fragment), so this guards the reconcile path's own sweep.
    """
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "KB")
        os.makedirs(src)
        cache = os.path.join(d, "cache")
        fpath = os.path.join(src, "doc.md")
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("alpha paragraph about caching.\n\nbeta paragraph about queues.")

        cm = ContextManager(budget_tokens=4096, embedder=BM25Embedder(),
                            library_distiller=KeywordDistiller())
        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)
        assert os.path.isfile(os.path.join(cache, "kb_doc_p0.md"))
        assert os.path.isfile(os.path.join(cache, "kb_doc_p1.md"))

        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("alpha paragraph about caching.")
        _rewrite_manifest_version(cache, P.SCHEMA_VERSION - 1)

        P.rebuild(cm, src, cache, "keyword", {".md"}, set(), base=src)

        assert os.path.isfile(os.path.join(cache, "kb_doc_p0.md"))
        assert not os.path.isfile(os.path.join(cache, "kb_doc_p1.md")), \
            "orphaned fragment file must be swept, not left on disk"
        assert not any(f.fid == "kb:doc#p1" for f in cm.store.all_fragments()), \
            "dropped fragment must not linger in the store"
