#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只为生成 planner 的真实 curl 请求体（fib + conway 两个 topic），不执行 curl。
从 swe_agent.roles 实读 _PLANNER_SYSTEM / _PLANNER_JSON_SCHEMA / _workspace_listing，
把真实 analyzer 输出作为 research_findings 注入，等价于 harness 真实链路。
用法（终端，qwen 已常驻）：
  curl -s --noproxy '*' -m 120 -X POST "http://localhost:1234/v1/chat/completions" -H "Content-Type: application/json" -d @/tmp/planner_fib.json
  curl -s --noproxy '*' -m 120 -X POST "http://localhost:1234/v1/chat/completions" -H "Content-Type: application/json" -d @/tmp/planner_conway.json
"""
import os, sys, json
REPO = "~/kuiwork/workdir2/litertlm"
os.chdir(REPO); sys.path.insert(0, REPO)

from swe_agent import dbg
from swe_agent import roles as R
from swe_agent.log import logger
dbg._ensure_plugins()

system = R._PLANNER_SYSTEM
schema = R._PLANNER_JSON_SCHEMA

# 真实 fib analyzer 输出（取自 /tmp/analyzer_ab.log B版 第65行）
fib_analyzer = (
    "斐波那契数列的计算规则为F(0)=0、F(1)=1、F(n)=F(n-1)+F(n-2)。"
    "常见的边界约定是F(0)和F(1)是基准值，对于负数索引通常不定义。"
    "接口形状为函数fibonacci，参数n（int），返回值为斐波那契数列的第n个数。"
    "测试约定是输入非负整数，输出对应的斐波那契数列值。"
    "潜在坑点是负数索引未定义和大数计算时递归方法可能导致栈溢出。"
)
# 真实 conway analyzer 输出（取自 /tmp/analyzer_conway_ws.log B版 第108行）
conway_analyzer = (
    "康威生命游戏的规则是：在无限的二维网格中，每个细胞有两种状态：活或死。"
    "每个细胞与它的八个邻居（水平、垂直和对角线方向）互动。"
    "在每一步中，根据以下规则更新细胞状态：任何活细胞如果少于两个活邻居则死亡；"
    "如果有超过三个活邻居则死亡；如果有两个或三个活邻居则保持不变；"
    "如果正好有三个活邻居，则新生一个活细胞。初始模式称为种子，后续世代通过应用这些规则同时更新。\n"
    "要点：\n"
    "  - 细胞有两种状态：活或死\n"
    "  - 每个细胞与八个邻居互动\n"
    "  - 活细胞的更新规则：少于两个活邻居死亡，超过三个活邻居死亡，两个或三个活邻居保持不变，正好三个活邻居新生\n"
    "  - 初始模式称为种子"
)

TOPICS = {
    "fib":     ("用 Python 实现斐波那契数列（fibonacci）", fib_analyzer),
    "conway":  ("用 Python 实现康威生命游戏（Conway's Game of Life）", conway_analyzer),
}

for name, (task, analyzer_out) in TOPICS.items():
    user = (
        f"【开发任务】\n{task}\n\n"
        f"【当前工作区文件清单】\n{R._workspace_listing()}\n\n"
        f"【Analyzer 调研发现（只读探查得到的事实，请据此规划，不要重复探查）】\n{analyzer_out}"
    )
    body = {
        "model": "qwen2.5.1-coder-7b-instruct",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": schema,
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    path = f"/tmp/planner_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)
    logger.info('%s', f'[ok] 写入 {path}')
    logger.info('%s', f"""  curl: curl -s --noproxy '*' -m 120 -X POST "http://localhost:1234/v1/chat/completions" -H "Content-Type: application/json" -d @/tmp/planner_{name}.json""")
    logger.info('')
logger.info('%s %s %s %s', 'system prompt 长度:', len(system), '| schema 字段数:', len(schema.get('json_schema', {}).get('schema', {}).get('properties', {})))
