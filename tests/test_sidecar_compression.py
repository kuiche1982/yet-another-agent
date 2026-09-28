#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/lfm_sidecar.compress_content 的回归测试（mock LFM，确定性，不依赖实时模型）。

背景：context 压缩此前完全没有单测，e2e 才是它第一次实战，且 1.2B LFM 做抽象概括会
臆测/中英互译/编造计数（同一源文件被压出 16/26/58 三种矛盾函数数）。修复后默认走
确定性抽取式截断 + 忠实闸门，本测试固化该行为。
"""
import sys
import os

# 让脚本在仓库内可直接 python 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import swe_agent.config as C
from swe_agent import lfm_sidecar as SC


def _mk(text, key_tokens):
    """构造 >2000 字符的输入：原文嵌关键 token，再补 padding 撑过阈值。"""
    pad = "\n# padding\n" + "\n".join(f"    # line {i}" for i in range(150))
    return (text + pad)[:3000]


PYTEST_OUT = _mk(
    "============================= test session starts ==============================\n"
    "collected 3 items\n"
    "test_game_of_life.py .F.                                                    [100%]\n"
    "=================================== FAILURES ===================================\n"
    "______________________________ test_step_basic _______________________________\n"
    "    def test_step_basic():\n"
    "        g = GameOfLife(4, 4)\n"
    "        g.set(1, 1)\n"
    "        g.step()\n"
    '>       assert g.get(1, 1) == 0, "huo-xibao-wu-lin-ju-ying-si-wang"\n'
    "E       AssertionError: huo-xibao-wu-lin-ju-ying-si-wang\n"
    "test_game_of_life.py:23: AssertionError\n"
    "=========================== short test summary info ============================\n"
    "FAILED test_game_of_life.py::test_step_basic - AssertionError: huo-xibao-wu-lin-ju-ying-si-wang\n"
    "1 failed, 2 passed in 0.31s\n",
    ['AssertionError', 'test_game_of_life.py', 'huo-xibao-wu-lin-ju-ying-si-wang', 'test_step_basic'],
)

SRC_OUT = _mk(
    "class GameOfLife:\n"
    "    def __init__(self, w, h):\n"
    "        self.grid = [[0]*w for _ in range(h)]\n"
    "    def step(self):\n"
    "        pass\n"
    "    def display(self):\n"
    "        pass\n"
    "    def set(self, x, y):\n"
    "        self.grid[y][x] = 1\n"
    "    def get(self, x, y):\n"
    "        return self.grid[y][x]\n",
    ['def step', 'def display', 'def set', 'def get'],
)

KEY_PYTEST = ['AssertionError', 'test_game_of_life.py',
              'huo-xibao-wu-lin-ju-ying-si-wang', 'test_step_basic']


def setup_module(_):
    C.SIDECAR_ENABLED = True


def test_extractive_preserves_key_tokens():
    """默认 extractive：pytest 输出的关键 token 全部保留，尾部失败块在。"""
    C.CONTENT_COMPRESS_STRATEGY = "extractive"
    out, comp = SC.compress_content(PYTEST_OUT, kind="tool output")
    assert comp is True
    for t in KEY_PYTEST:
        assert t in out, f"extractive 丢失关键 token: {t!r}"
    assert "FAILURES" in out, "尾部错误块被截断"


def test_source_skipped_verbatim():
    """源码工件：绝不压缩，原样返回，所有方法签名保留。"""
    C.CONTENT_COMPRESS_STRATEGY = "extractive"
    out, comp = SC.compress_content(SRC_OUT, kind="source file")
    assert comp is False
    assert out.lstrip().startswith("class GameOfLife")
    for m in ['def step', 'def display', 'def set', 'def get']:
        assert m in out


def test_lfm_guard_rejects_token_dropping_summary():
    """lfm 模式：LFM 返回丢 token 的幻觉摘要 → 忠实闸门拦下，回退原文。"""
    C.CONTENT_COMPRESS_STRATEGY = "lfm"
    SC.M.chat_text_messages = lambda *a, **k: "Test failed. The cell should have died. No implementation."
    out, comp = SC.compress_content(PYTEST_OUT, kind="tool output")
    assert comp is False
    assert out == PYTEST_OUT, "闸门未回退原文"


def test_lfm_guard_accepts_faithful_summary():
    """lfm 模式：LFM 返回保留关键 token 的忠实摘要 → 采纳。"""
    C.CONTENT_COMPRESS_STRATEGY = "lfm"
    SC.M.chat_text_messages = lambda *a, **k: (
        "FAILED test_game_of_life.py::test_step_basic - AssertionError: "
        "huo-xibao-wu-lin-ju-ying-si-wang (cell died)")
    out, comp = SC.compress_content(PYTEST_OUT, kind="tool output")
    assert comp is True
    assert out.startswith("[LFM 压缩摘要")


def test_short_input_not_compressed():
    """未超阈值的短输入：直接返回原文。"""
    C.CONTENT_COMPRESS_STRATEGY = "extractive"
    short = "just a short message"
    out, comp = SC.compress_content(short, kind="tool output")
    assert comp is False and out == short


if __name__ == "__main__":
    setup_module(None)
    for fn in [test_extractive_preserves_key_tokens, test_source_skipped_verbatim,
               test_lfm_guard_rejects_token_dropping_summary,
               test_lfm_guard_accepts_faithful_summary, test_short_input_not_compressed]:
        fn()
        print(f"PASS {fn.__name__}")
    print("ALL PASS")
