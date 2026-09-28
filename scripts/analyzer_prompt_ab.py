#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyzer 提示词 A/B 对比（同一模型 qwen2.5.1-coder-7b，只切换 system prompt）。

A = 磁盘当前版 (roles._ANALYZER_SYSTEM)
B = 用户修订版

控制变量：
- 临时剔除 web_search（避免远程 zhipu 调用 / 代理坑），analyzer 工具集 = read_file/grep/glob/finish_analysis
- 关闭 analyzer 远程 fallback（C.ANALYZER_FALLBACK_MODEL=""），全程本地 qwen
- 不修改 roles.py / config.py，仅运行时 monkeypatch
"""
import os, sys, time

REPO = "~/kuiwork/workdir2/litertlm"
os.chdir(REPO)
sys.path.insert(0, REPO)

import logging
logging.disable(logging.WARNING)

from swe_agent import dbg, config as C, registry as Reg
from swe_agent import roles as R
from swe_agent.log import logger

# ---- 控制变量：关远程 fallback + 去 web_search ----
C.ANALYZER_FALLBACK_MODEL = ""

# ---- 加载插件拿工具集，再剔除 web_search ----
dbg._ensure_plugins()
Reg.ROLE_TOOLS["analyzer"] = {"read_file", "grep", "glob", "finish_analysis"}
logger.info('%s %s', '[setup] analyzer 工具集 =', sorted(Reg.ROLE_TOOLS['analyzer']))
logger.info('%s %s', '[setup] ANALYZER_FALLBACK_MODEL =', repr(C.ANALYZER_FALLBACK_MODEL))
logger.info('%s %s', '[setup] analyzer 模型 =', __import__('swe_agent.models', fromlist=['M']).role_model_id('analyzer'))
logger.info('')

PROMPT_A = R._ANALYZER_SYSTEM  # 磁盘当前版

PROMPT_B = """你是软件需求分析师（Requirements Analyst）。
职责：你不写代码，你负责把用户的需求理解透彻，并补齐必要的领域知识，输出一份完整且【克制的】需求与关键资料摘要，供下游 Planner / 实现者使用。
工作方式：
- 你被授权按需调用工具来辅助理解，仅在必要时调用工具，不要为了调用而调用。
- 先吃透用户需求。遇到你【不了解或无法确认】的需求（某算法的具体规则、某库 / API 的用法等），先收集资料再下结论：可借助联网检索，也可阅读当前代码来加深理解；不要凭猜测给出不准确的结论。
- 对于常识行问题、无需查证的需求（如常见标准库功能），无需联网，直接基于任务描述整理即可。
输出要求（理解充分后调用结束动作提交摘要）：
- 克制：只输出与用户描述直接相关的需求要点 + 关键参考资料，不要写实现方案、不要写代码。
- 输出应贴合用户描述。例如「用 Python 实现斐波那契」应输出：斐波那契数列的计算规则（F(0)=0、F(1)=1、F(n)=F(n-1)+F(n-2)）、常见的边界约定等；「实现康威生命游戏」应输出：生存 / 死亡规则（存活邻居数为 2 或 3 则存活、恰好 3 则新生等）。
- 若工作区已有相关代码，指出其设计思路、接口形状、测试约定、潜在坑点等事实。
"""

TOPICS = [
    "用 Python 实现斐波那契数列（fibonacci）",
    "用 Python 实现康威生命游戏（Conway's Game of Life）",
    "用 Python 实现快速排序（quicksort）",
    "用 Python 写一个读取 CSV 文件并统计每列平均值的脚本",
]

RUNS = [("A_当前磁盘版", PROMPT_A), ("B_用户修订版", PROMPT_B)]

logger.info('%s', '=' * 70)
logger.info('%s', f'共 {len(TOPICS)} 个 topic × {len(RUNS)} 个 prompt = {len(TOPICS) * len(RUNS)} 次 analyzer 调用')
logger.info('%s', '=' * 70)

for label, prompt in RUNS:
    for topic in TOPICS:
        R._ANALYZER_SYSTEM = prompt
        t0 = time.time()
        try:
            out = R.run_analyzer(topic, max_steps=6)
        except Exception as e:
            out = f"[EXCEPTION] {type(e).__name__}: {e}"
        dt = time.time() - t0
        logger.info('%s', '\n' + '#' * 70)
        logger.info('%s', f'# {label} | {topic}')
        logger.info('%s', f'# 耗时 {dt:.1f}s | 输出长度 {len(out)} 字')
        logger.info('%s', '#' * 70)
        logger.info('%s', out or '（空：未产出调研）')

logger.info('%s', '\n' + '=' * 70)
logger.info('%s', '全部 A/B 对比结束')
logger.info('%s', '=' * 70)
