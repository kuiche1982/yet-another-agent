#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""注入层逐字节基线（迁移护栏）。

背景：注入层（`_rag_block` / `local_search` / `_ref_location` / `_clip`）正从
`swe_agent/management.py` 下沉到 `contextmgr`。搬迁最危险的不是「功能丢了」（会立刻报错），
而是「看起来都对、但格式/口径悄悄漂了」—— 这种回归在人工 review 下几乎必然漏掉。

本测试把快照（`tests/fixtures/inject_baseline.json`）与**当前实现的实际输出**逐字节比对。
快照由 `scripts/gen_inject_baseline.py` 生成，输入片段与参数全部钉死（model-free、无检索
数值不确定性），所以「同一组输入 → 同一串字节」是硬契约。

注意：快照只固化**格式化行为**，不固化 BM25 打分 / 切块 / token 估算 —— 后者随 tokenizer
实现（有无 tiktoken）变化，与「格式是否漂移」无关。快照过期时的正确动作是：
确认漂移是**有意**的 → 跑 `python scripts/gen_inject_baseline.py` 重新固化；否则修实现。
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FIXTURE = Path(__file__).parent / "fixtures" / "inject_baseline.json"


def _load_generator():
    """按路径加载生成器（scripts/ 不是包，故用 spec_from_file_location）。"""
    path = REPO / "scripts" / "gen_inject_baseline.py"
    spec = importlib.util.spec_from_file_location("gen_inject_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_generator()


def _first_diff(a: str, b: str) -> str:
    """定位首处差异并给出可读上下文：长行只报字符偏移与窗口，不整行倾倒。"""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            if max(len(a), len(b)) <= 200:
                return "char %d: baseline %r != current %r" % (i, a, b)
            lo = max(0, i - 30)
            return ("char %d (of %d): baseline ...%r... != current ...%r..."
                    % (i, len(a), a[lo:i + 30], b[lo:i + 30]))
    return "common prefix; length %d -> %d, tail baseline %r vs current %r" % (
        len(a), len(b), a[len(b):][:60], b[len(a):][:60])


def test_injection_layer_matches_byte_baseline():
    """Current injection output must equal the recorded snapshot byte-for-byte."""
    want = FIXTURE.read_text(encoding="utf-8")
    got = GEN._dump(GEN.run_cases())
    if got != want:
        wl, gl = want.splitlines(), got.splitlines()
        for i, (a, b) in enumerate(zip(wl, gl), 1):
            if a != b:
                pytest.fail("injection baseline drift at line %d: %s" % (i, _first_diff(a, b)))
        pytest.fail("injection baseline drift: line count %d -> %d" % (len(wl), len(gl)))
    assert json.loads(got) == json.loads(want)


def test_baseline_covers_every_injection_entrypoint():
    """The snapshot must actually exercise all four moved functions."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for key in ("clip", "ref_location", "rag_block", "local_search", "local_search_paged"):
        assert key in data, "baseline fixture missing section: %s" % key
    assert len(data["clip"]) >= 5, "clip needs empty / short / multiline / single-line / zero-budget"
    assert len(data["ref_location"]) >= 5, "ref_location needs code+library x with/without lineno"
    assert "### [1]" in data["rag_block"], "rag_block must keep its numbered entry header"
    assert "相关性" in data["rag_block"], "rag_block must keep the relevance field"
    assert "read_file " in data["rag_block"], "rag_block must keep the read_file pointer"
    assert "[proj]" in data["local_search"], "local_search must keep the layer tag"


def test_run_cases_leaves_no_global_config_side_effect():
    """run_cases() pins config values temporarily and must restore them (same-process pytest)."""
    before = {k: getattr(GEN.C, k) for k in GEN.PINNED}
    GEN.run_cases()
    after = {k: getattr(GEN.C, k) for k in GEN.PINNED}
    assert before == after, "run_cases leaked pinned config: %r -> %r" % (before, after)
