#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_inject_baseline.py —— 固化「注入层」的逐字节基线快照（迁移前的行为锚点）。

为什么需要：注入层（`_rag_block` / `local_search` / `_ref_location` / `_clip`）即将从
`swe_agent/management.py` 下沉到 `contextmgr`。这类搬迁最大的风险不是「功能丢了」——
那是会立刻报错的；而是「看起来都对，但格式/口径悄悄漂了」。必须先把当前行为冻成
逐字节快照，搬完再比对；否则只能靠肉眼，等于没有防线。

设计取舍（重要）：
- 只固化**注入层的格式化行为**，不固化 BM25 打分 / 切块 / token 估算。后者的数值随
  tokenizer 实现（有/无 tiktoken，同段文本差约 2 倍）变化，且与「格式是否漂移」无关。
  故输入片段全部手工钉死（fid / text / origin / lineno / score），KB 用 `_StubKB` 顶替。
- 参数（RAG_INJECT_MAX_CHARS / RAG_INJECT_SUMMARY / LOCAL_SEARCH_*）在生成与校验两侧
  统一钉死，保证「同一组输入 → 同一串字节」。
- 全程 model-free / tool-free：不开摘要、不触网络、不触向量模型。

用法：
    python scripts/gen_inject_baseline.py            # 生成/覆盖快照
    python scripts/gen_inject_baseline.py --check     # 只校验（不写盘），漂移则 exit 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from contextmgr.types import Fragment, Source          # noqa: E402
from swe_agent import config as C                      # noqa: E402
from swe_agent import management as M                  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "inject_baseline.json"

# —— 参数钉死：生成与校验必须用同一组值，否则比的是参数不是行为 ——
PINNED = {
    "RAG_INJECT_MAX_CHARS": 240,
    "RAG_INJECT_SUMMARY": 0,
    "LOCAL_SEARCH_TOP_K": 3,
    "LOCAL_SEARCH_SNIPPET_CHARS": 160,
}

CODE_MULTILINE = "\n".join(
    "def handler_%02d(payload):\n    return payload.get('key_%02d', 0)" % (i, i)
    for i in range(1, 13)
)
LONG_SINGLE_LINE = "x" * 600


class _StubKB:
    """顶替 LayeredKB：只为注入层提供固定输入，避开真实检索的数值不确定性。"""

    LIB_REALPATH = {"kb:workflow_design": "KnowledgeBase/workflow_design.md"}

    def __init__(self, refs, results):
        self._refs = list(refs)
        self._results = list(results)

    def _lib_realpath(self, origin: str) -> str:
        return self.LIB_REALPATH.get(origin, "")

    def retrieve(self, query, budget_tokens=2048, sources=None, per_layer_k=8):
        return list(self._refs)

    def search(self, query, scope="all", top_k=10, offset=0):
        return list(self._results)


def _code_frag(fid, text, origin, lineno, end_lineno, score=0.0):
    return Fragment(fid=fid, source=Source.CODE, text=text, origin=origin,
                    lineno=lineno, end_lineno=end_lineno, score=score)


def _lib_frag(fid, text, origin, lineno=0, end_lineno=0, score=0.0):
    return Fragment(fid=fid, source=Source.LIBRARY, text=text, origin=origin,
                    lineno=lineno, end_lineno=end_lineno, score=score)


def _refs():
    """钉死的注入输入：一条代码片段（带行号）+ 两条文档片段（一条会触发截断）。"""
    return [
        _lib_frag("f-lib-long", CODE_MULTILINE, "kb:workflow_design",
                  lineno=101, end_lineno=105, score=0.95),
        _code_frag("f-code", "def prepare_messages(self, model_context_length=64000):\n"
                             "    return []",
                   "swe_agent/management.py", 973, 975, score=0.87),
        _lib_frag("f-lib-short", "Judge mode: post-hoc binary review.", "kb:workflow_design",
                  score=0.31),
    ]


@contextmanager
def _stub_kb(stub):
    """把 module 级 `_get_layered_kb` 换成 stub（`local_search` 内部自己取 KB）。"""
    orig = M._get_layered_kb
    M._get_layered_kb = lambda: stub
    try:
        yield
    finally:
        M._get_layered_kb = orig


@contextmanager
def _pinned_config():
    """临时钉死参数并在退出时**原样还原**（pytest 同进程内跑，不能污染全局 config）。"""
    saved = {k: getattr(C, k, None) for k in PINNED}
    for k, v in PINNED.items():
        setattr(C, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                delattr(C, k)
            else:
                setattr(C, k, v)


def run_cases() -> dict:
    """产出全部基线用例的输出（纯函数：同一输入 → 同一结果，且不留全局副作用）。"""
    refs = _refs()
    stub = _StubKB(refs, [("proj", refs[0]), ("code", refs[1])])

    def _identity(msgs, *a, **k):
        return msgs

    cm = M.ContextManager(kb=stub, backend=_identity, compress_backend=_identity)

    clip_cases = []
    with _pinned_config():
        for name, text, max_chars in (
            ("empty", "", 240),
            ("short_untouched", "one short line", 240),
            ("multiline_line_boundary", CODE_MULTILINE, 240),
            ("single_line_char_split", LONG_SINGLE_LINE, 240),
            ("zero_budget", "text", 0),
        ):
            out, clipped = M.ContextManager._clip(text, max_chars)
            clip_cases.append({"case": name, "in": text, "max_chars": max_chars,
                               "out": out, "clipped": bool(clipped)})

        loc_cases = []
        for name, frag in (
            ("code_with_lines", _code_frag("a", "x", "swe_agent/management.py", 485, 507)),
            ("code_without_lines", _code_frag("b", "x", "README.md", 0, 0)),
            ("library_with_lines", _lib_frag("c", "x", "kb:workflow_design", 101, 105)),
            ("library_without_lines", _lib_frag("d", "x", "kb:workflow_design")),
            ("library_unknown_origin", _lib_frag("e", "x", "kb:missing_doc", 7, 9)),
        ):
            loc, pointer = M._ref_location(stub, frag)
            loc_cases.append({"case": name, "loc": loc, "pointer": pointer})

        with _stub_kb(stub):
            local_out = M.local_search("judge mode", scope="all")
            local_paged = M.local_search("judge mode", scope="all", top_k=2, offset=1)

            rag_block = cm._rag_block(refs, "judge mode", 2048)

    return {
        "pinned": dict(PINNED),
        "clip": clip_cases,
        "ref_location": loc_cases,
        "rag_block": rag_block,
        "local_search": local_out,
        "local_search_paged": local_paged,
    }


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="注入层逐字节基线快照")
    ap.add_argument("--check", action="store_true", help="只校验不写盘；漂移则 exit 1")
    args = ap.parse_args(argv)

    cur = run_cases()
    text = _dump(cur)
    if args.check:
        if not FIXTURE.is_file():
            print("!! 快照不存在：%s" % FIXTURE, file=sys.stderr)
            return 1
        want = FIXTURE.read_text(encoding="utf-8")
        if want == text:
            print("baseline OK: %s" % os.path.relpath(FIXTURE, REPO))
            return 0
        import difflib
        diff = difflib.unified_diff(want.splitlines(), text.splitlines(),
                                    "baseline", "current", lineterm="", n=2)
        print("\n".join(list(diff)[:80]), file=sys.stderr)
        print("!! 注入层输出与基线不一致（逐字节）", file=sys.stderr)
        return 1
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(text, encoding="utf-8")
    print("wrote %s" % os.path.relpath(FIXTURE, REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
