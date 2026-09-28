#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LFM（本地 rapid-mlx）端到端验证：用 LFM2.5-1.2B 同时充当 executor / planner / reviewer，
验证重构后的 ToolRegistry 分发 + 注册式钩子（接手保护、自动完成任务、测试反馈、计数、
plan_mode 守卫）端到端可用。完全本地、无需 GLM 远程，跑得快。

场景：接手含 bug 的 GameOfLife 实现（step 坐标写反）—— 继承实现受接手模式保护，
执行器只能 edit_file 最小修复；#84 隐藏验收闸门（blinker）负责最终放行。
"""
import os
import sys

# —— 全部模型用本地 LFM（TEXT_JSON，rapid-mlx 本地服务，无需 GLM 远程）——
os.environ["EXECUTOR_MODEL"] = "LFM2.5-1.2B"
os.environ["PLANNER_MODEL"] = "LFM2.5-1.2B"   # 等价 --no-glm：本地自规划
os.environ["REVIEWER_MODEL"] = "LFM2.5-1.2B"  # 等价 --no-glm：本地自评审

sys.path.insert(0, "~/kuiwork/workdir2/litertlm")
from swe_agent import config, supervisor
from swe_agent.log import logger

# ---- 1) 有 bug 的继承实现（step 坐标写反 new[x][y]）----
BUGGY_IMPL = '''# game_of_life.py  —— 他人实现（半成品，含 bug）
class GameOfLife:
    def __init__(self, height=20, width=20):
        self.height = height
        self.width = width
        self.grid = [[0] * self.width for _ in range(self.height)]

    def _count_neighbors(self, y, x):
        c = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    c += self.grid[ny][nx]
        return c

    def step(self):
        new = [[0] * self.width for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                n = self._count_neighbors(y, x)
                if self.grid[y][x] == 1 and n in (2, 3):
                    new[x][y] = 1   # BUG: 坐标写反
                elif self.grid[y][x] == 0 and n == 3:
                    new[x][y] = 1   # BUG: 坐标写反
        self.grid = new

    def display(self):
        return "\\n".join("".join("#" if c else "." for c in row) for row in self.grid)
'''

# ---- 2) 空心绿自测（只测 still-life / init，对 step bug 视而不见）----
HOLLOW_SELFTEST = '''# test_game_of_life.py  —— 既有测试夹具（接手模式保护，禁止覆盖）
import pytest
from game_of_life import GameOfLife


def test_init():
    g = GameOfLife()
    assert isinstance(g, GameOfLife)
    assert g.width == 20 and g.height == 20
    assert all(all(c == 0 for c in row) for row in g.grid)


def test_block_still_life():
    g = GameOfLife(5, 5)
    for (y, x) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        g.grid[y][x] = 1
    g.step()
    for (y, x) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        assert g.grid[y][x] == 1
'''

# ---- 3) 权威隐藏验收（正确断言 blinker 演变，能戳穿 step bug）----
USER_HIDDEN = '''# test_acceptance_user.py  —— 人工投放的权威验收（优先级高于 Planner 生成）
import pytest
from game_of_life import GameOfLife


def test_blinker_oscillates():
    g = GameOfLife(5, 5)
    for (y, x) in [(1, 0), (1, 1), (1, 2)]:
        g.grid[y][x] = 1
    g.step()
    for y in (0, 1, 2):
        assert g.grid[y][1] == 1, f"blinker cell ({y},1) should be alive"
    assert g.grid[1][0] == 0, "blinker should not extend to (1,0)"
    assert g.grid[1][2] == 0, "blinker should not extend to (1,2)"
'''

# ---- 落盘种子 ----
config.WORKSPACE.mkdir(parents=True, exist_ok=True)
config.HIDDEN_TESTS_DIR.mkdir(parents=True, exist_ok=True)
(config.WORKSPACE / "game_of_life.py").write_text(BUGGY_IMPL, encoding="utf-8")
(config.WORKSPACE / "test_game_of_life.py").write_text(HOLLOW_SELFTEST, encoding="utf-8")
(config.HIDDEN_TESTS_DIR / "test_acceptance_user.py").write_text(USER_HIDDEN, encoding="utf-8")
logger.info('%s', f"[seed] 继承实现  -> {config.WORKSPACE / 'game_of_life.py'}")
logger.info('%s', f"[seed] 空心绿自测 -> {config.WORKSPACE / 'test_game_of_life.py'}")
logger.info('%s', f"[seed] 权威隐藏验收 -> {config.HIDDEN_TESTS_DIR / 'test_acceptance_user.py'}")

TAKEOVER_TASK = (
    "# 技术栈：python, pytest, OOP\n"
    "## 任务：接手现有代码\n"
    "工作区 agent_sandbox/ 里已经有别人写的 `game_of_life.py`（GameOfLife 类）"
    "和一份测试 `test_game_of_life.py`。这些是**已有代码**，不要从头重写整个项目。\n"
    "请接手这份实现：读懂现有代码，运行 `test_game_of_life.py` 确保其通过；"
    "如果发现有逻辑错误，用 edit_file 做最小修复让测试由红变绿。\n"
    "目标：确认 GameOfLife 的 step/init/display 行为正确，并通过最终验收。"
)

sys.argv = ["swe_agent", TAKEOVER_TASK]
supervisor.main()
