#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单格对比运行器（run_compare_one.py）
====================================
用途：在【独立子进程】中跑一个对比单元 —— (executor 模型, 任务类型)，
      planner/reviewer 统一由 glm-4.7 充当「大脑」。

为什么必须独立子进程？
  config 模块在 import 时按 EXECUTOR_MODEL 解析出 MODEL_PATH / SERVED_NAME
  （rapid-mlx 要加载的权重），这是模块级常量，运行期改 EXECUTOR_MODEL 不会
  重新计算。若在同一进程里切换 ling↔qwen，本地服务会加载错权重。
  用子进程 → 每个格 fresh import → 权重正确。

对比设计（受控变量）：
  - 大脑：始终 glm-4.7（planner + reviewer）
  - 双手：ling-3.0-tiny  vs  qwen-2.5-coder-7b（两者都走本地 rapid-mlx / TEXT_JSON，传输路径一致，公平）
  - 任务：
      new     = 从零构建 CLI 康威生命游戏（DEFAULT_TASK，空工作区）
      bugfix  = 接手他人有 bug 的实现并修复（隐藏验收闸门 test_acceptance_user.py 能戳穿 bug）
  - bugfix 的种子实现【去掉了 # BUG 注释】，是真正的「找 bug」任务，而非读注释抄答案。

结果追加写入 compare_results.jsonl（driver 负责在开始处清空）。
"""
import os
import sys
import time
import json
import shutil

MODEL = sys.argv[1] if len(sys.argv) > 1 else "ling"
TASK = sys.argv[2] if len(sys.argv) > 2 else "new"
assert MODEL in ("ling", "qwen", "lfm", "qwenlm"), MODEL
assert TASK in ("new", "bugfix"), TASK

# ⚠️ 必须在 import swe_agent.config 之前设置，否则模块级权重解析会用到默认值
# 大脑原定 glm-4.7，但实测 glm-4.7（含 -flash）当前对「生成完整执行契约」的长生成会
# 卡死（read timeout=600s 零输出），故改用当前可用的 glm-4.5-flash 作为大脑（49.6s 出有效契约）。
# 验证用 LOCAL_BRAIN=1：强制 Planner/Reviewer 走本地降级（不依赖远程 GLM，规避其间歇性限流），
# 专注观测 executor 历史重置效果。
if os.environ.get("LOCAL_BRAIN"):
    os.environ["PLANNER_MODEL"] = ""
    os.environ["REVIEWER_MODEL"] = ""
else:
    # 尊重调用方通过 env 指定的大脑（如 glm-4.1v-thinking-flashx 免费额度模型）；
    # 未指定时回落到 glm-4.5-flash（实测生成完整契约最稳，49.6s 出有效契约）。
    os.environ["PLANNER_MODEL"] = os.environ.get("PLANNER_MODEL") or "glm-4.5-flash"
    os.environ["REVIEWER_MODEL"] = os.environ.get("REVIEWER_MODEL") or "glm-4.5-flash"
# ling/qwen/lfm 三者都走本地 rapid-mlx（TEXT_JSON，传输路径一致，公平）；
# lfm 取本地 LFM2.5-1.2B 权重（与 ling 同后端，隔离「模型本身」的影响）。
if MODEL == "ling":
    os.environ["EXECUTOR_MODEL"] = "Ling-3.0-Tiny"
elif MODEL == "lfm":
    os.environ["EXECUTOR_MODEL"] = "LFM2.5-1.2B"
elif MODEL == "qwenlm":
    # qwen-2.5-coder-7b 在 rapid-mlx 上对真实 prompt 会卡死（服务端不返回）；
    # 改用 LM Studio 的 qwen2.5.1-coder-7b-instruct（同一模型权重），走 TEXT_JSON，契约不变。
    os.environ["EXECUTOR_MODEL"] = "qwen-lm-tjson"
else:
    os.environ["EXECUTOR_MODEL"] = "qwen-2.5-coder-7b"

REPO = "~/kuiwork/workdir2/litertlm"
sys.path.insert(0, REPO)
from swe_agent import config, supervisor          # noqa: E402
from swe_agent import state                       # noqa: E402
from swe_agent.log import logger


# ----------------------------------------------------------------------
# bugfix 种子：有 bug 的继承实现（step 坐标写反 new[x][y]），去掉 # BUG 注释
# ----------------------------------------------------------------------
BUGGY_IMPL = '''# game_of_life.py  —— 他人实现（半成品）
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
                    new[x][y] = 1
                elif self.grid[y][x] == 0 and n == 3:
                    new[x][y] = 1
        self.grid = new

    def display(self):
        return "\\n".join("".join("#" if c else "." for c in row) for row in self.grid)
'''

# 空心绿自测：只测 still-life / init，对 step bug 视而不见（受接手模式保护，禁止覆盖）
HOLLOW_SELFTEST = '''import pytest
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

# 权威隐藏验收：正确断言 blinker 演变，能戳穿 step bug（位于 WORKSPACE 之外，executor 不可见/不可改）
USER_HIDDEN = '''import pytest
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

TAKEOVER_TASK = (
    "# 技术栈：python, pytest, OOP\n"
    "## 任务：接手现有代码\n"
    "工作区 agent_sandbox/ 里已经有别人写的 `game_of_life.py`（GameOfLife 类）"
    "和一份测试 `test_game_of_life.py`。这些是**已有代码**，不要从头重写整个项目。\n"
    "请接手这份实现：读懂现有代码，运行 `test_game_of_life.py` 确保其通过；"
    "如果发现有逻辑错误，用 edit_file 做最小修复让测试由红变绿。\n"
    "目标：确认 GameOfLife 的 step/init/display 行为正确，并通过最终验收。"
)


def _clear(d: "config.WORKSPACE.__class__"):
    d.mkdir(parents=True, exist_ok=True)
    for p in d.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def main():
    _clear(config.WORKSPACE)
    for p in config.HIDDEN_TESTS_DIR.glob("*.py"):
        p.unlink()

    if TASK == "bugfix":
        (config.WORKSPACE / "game_of_life.py").write_text(BUGGY_IMPL, encoding="utf-8")
        (config.WORKSPACE / "test_game_of_life.py").write_text(HOLLOW_SELFTEST, encoding="utf-8")
        (config.HIDDEN_TESTS_DIR / "test_acceptance_user.py").write_text(USER_HIDDEN, encoding="utf-8")
        logger.info('%s', f"[seed] 继承实现(bug) -> {config.WORKSPACE / 'game_of_life.py'}")
        logger.info('%s', f"[seed] 空心绿自测   -> {config.WORKSPACE / 'test_game_of_life.py'}")
        logger.info('%s', f"[seed] 权威隐藏验收 -> {config.HIDDEN_TESTS_DIR / 'test_acceptance_user.py'}")
        task_text = TAKEOVER_TASK
    else:
        logger.info('%s', f'[seed] 空工作区（从零构建）：{config.WORKSPACE}')
        task_text = supervisor.DEFAULT_TASK

    t0 = time.time()
    # 验证用：允许通过环境变量压低最大轮数，缩短单次验证耗时（不影响功能）
    _cap = int(os.environ.get("CELL_MAX_ITER", "30"))
    if _cap != config.MAX_ITER:
        config.MAX_ITER = _cap
        logger.info('%s', f'[compare] CELL_MAX_ITER={_cap} → 已覆盖 config.MAX_ITER')
    try:
        sys.argv = ["swe_agent", task_text]
        supervisor.main()
    except Exception as e:  # 运行异常也要落结果，避免整轮无记录
        logger.info('%s', f'[compare] 运行异常：{e}')

    fv = state.GLOBAL_STATE.get("final_validation") or {}
    passed = bool(fv.get("passed"))
    verdict = "success" if passed else "failed"
    # 记录「实际生效的大脑」：若 glm 调用次数 > 0 说明远程 Planner 真实参与了规划；
    # 否则说明 planner 因网络超时静默降级为本地自规划，该格大脑并非远程模型，需在报告中标注。
    brain_used = config.PLANNER_MODEL if config.STATS.get("glm_calls", 0) > 0 else "local-fallback"
    _per_round = config.STATS.get("input_chars_per_round", []) or []
    result = {
        "model": MODEL,
        "task": TASK,
        "executor_model": config.EXECUTOR_MODEL,
        "reset_on_task_done": config.EXECUTOR_RESET_ON_TASK_DONE,
        "reset_keep_turns": config.EXECUTOR_RESET_KEEP_TURNS,
        "brain_intended": config.PLANNER_MODEL,
        "brain_used": brain_used,
        "verdict": verdict,
        "rounds": config.STATS.get("rounds", 0),
        "executor_calls": config.STATS.get("calls", 0),
        "glm_calls": config.STATS.get("glm_calls", 0),
        "reset_events": config.STATS.get("reset_events", 0),
        "input_chars_total": config.STATS.get("input_chars_total", 0),
        "input_chars_first_round": _per_round[0] if _per_round else 0,
        "input_chars_last_round": _per_round[-1] if _per_round else 0,
        "input_chars_max": max(_per_round) if _per_round else 0,
        "duration_s": round(time.time() - t0, 1),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    res_path = os.path.join(REPO, "compare_results.jsonl")
    with open(res_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info('%s', f'[compare] 结果已记录：{result}')


if __name__ == "__main__":
    main()
