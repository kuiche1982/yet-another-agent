#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚焦验证：开 web_search 后，conway 这个「两版都翻车」的 topic，B（已落盘）能否靠「先查资料」补出真实规则。
A = 旧磁盘版（硬编码）；B = 当前落盘的 B 版（R._ANALYZER_SYSTEM）。
开 install_logging 以观察模型是否真的调用了 web_search。
"""
import os, sys, time, logging
REPO = "~/kuiwork/workdir2/litertlm"
os.chdir(REPO); sys.path.insert(0, REPO)
logging.disable(logging.WARNING)

from swe_agent import dbg, config as C, registry as Reg
from swe_agent import roles as R
from swe_agent.log import logger

C.ANALYZER_FALLBACK_MODEL = ""           # 关远程 fallback，纯本地 qwen
dbg._ensure_plugins()
Reg.ROLE_TOOLS["analyzer"] = {"read_file", "grep", "glob", "web_search", "finish_analysis"}
dbg.install_logging()                       # 打印每轮 tools（看是否调 web_search）

PROMPT_A = """你是软件需求分析师（Requirements Analyst）。

职责：在动手写代码之前，把用户的开发需求理解透彻，并补齐必要的领域知识，输出一份【克制的】需求与关键资料摘要，供下游 Planner / 实现者使用。

工作方式：
- 你被授权按需调用工具来辅助理解（例如联网检索资料、阅读当前工作区已有代码）。是否调用、调用哪个，由你根据「理解需求」的目标自行判断，不要为了调用而调用。
- 先吃透用户需求。遇到你【不了解或无法确认】的需求（某算法的具体规则、某库 / API 的用法等），先收集资料再下结论：可借助联网检索，也可阅读当前代码来加深理解；不要凭猜测给出不准确的结论。
- 对于已知、无需查证的需求（如常见标准库功能），无需联网，直接基于任务描述整理即可。

输出要求（理解充分后调用结束动作提交摘要）：
- 克制：只输出与用户描述直接相关的需求要点 + 关键参考资料，不要写实现方案、不要写代码。
- 输出应贴合用户描述。例如「用 Python 实现斐波那契」应输出：斐波那契数列的计算规则（F(0)=0、F(1)=1、F(n)=F(n-1)+F(n-2)）、常见的边界约定等；「实现康威生命游戏」应输出：生存 / 死亡规则（存活邻居数为 2 或 3 则存活、恰好 3 则新生等）。
- 若工作区已有相关代码，指出其接口形状、测试约定、潜在坑点等事实。
"""

PROMPT_B = R._ANALYZER_SYSTEM  # 已落盘为 B 版

topic = "用 Python 实现康威生命游戏（Conway's Game of Life）"

for label, p in [("A_旧磁盘版", PROMPT_A), ("B_已落盘版", PROMPT_B)]:
    R._ANALYZER_SYSTEM = p
    logger.info('%s', '\n' + '=' * 70)
    logger.info('%s', f'=== {label} | conway | web_search ENABLED ===')
    logger.info('%s', '=' * 70)
    t0 = time.time()
    out = R.run_analyzer(topic, max_steps=6)
    logger.info('%s', f'\n##### {label} 最终结果 | {time.time() - t0:.1f}s | {len(out)}字 #####')
    logger.info('%s', out or '（空）')
