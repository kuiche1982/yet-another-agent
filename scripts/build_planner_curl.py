#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只为生成 planner 的真实 curl 请求体 + 命令文本（不执行 curl）。
从 swe_agent.roles 实际读取 _PLANNER_SYSTEM / _PLANNER_JSON_SCHEMA / _workspace_listing，
把 fib analyzer 真实输出作为 research_findings 注入，等价于 harness 真实链路。
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
analyzer_out = (
    "斐波那契数列的计算规则为F(0)=0、F(1)=1、F(n)=F(n-1)+F(n-2)。"
    "常见的边界约定是F(0)和F(1)是基准值，对于负数索引通常不定义。"
    "接口形状为函数fibonacci，参数n（int），返回值为斐波那契数列的第n个数。"
    "测试约定是输入非负整数，输出对应的斐波那契数列值。"
    "潜在坑点是负数索引未定义和大数计算时递归方法可能导致栈溢出。"
)

task = "用 Python 实现斐波那契数列（fibonacci）"
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

with open("/tmp/planner_body.json", "w", encoding="utf-8") as f:
    json.dump(body, f, ensure_ascii=False, indent=2)

logger.info('%s', '请求体已写入 -> /tmp/planner_body.json')
logger.info('%s', '=' * 60)
logger.info('%s', '直接复制下面这条 curl 到终端执行（qwen 已常驻）：')
logger.info('%s', '=' * 60)
logger.info('%s', 'curl -s --noproxy \'*\' -m 120 -X POST "http://localhost:1234/v1/chat/completions" \\')
logger.info('%s', '  -H "Content-Type: application/json" \\')
logger.info('%s', '  -d @/tmp/planner_body.json')
