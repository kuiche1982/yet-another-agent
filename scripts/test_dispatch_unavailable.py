#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回归测试：ToolRegistry.dispatch 对「运行时不可用工具」返回结构化回执，
而非让异常冒泡炸掉 agent loop / 退化为 None。

覆盖：
  1) 已注册但运行时抛非 TypeError 异常（网络/MCP/插件不可用）→ "error: 工具 X 暂时不可用：<原因>"
  2) 传参错误（TypeError）→ 仍走原有 "error: 工具 X 参数错误" 分支（不回归）
  3) 未注册动作 → "error: 未知动作"
  4) 正常工具 → 原样返回结果
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swe_agent.registry import ToolRegistry, ToolDef
from swe_agent.log import logger


def ok_run(**kw):
    return "ok-result"


def flaky_run(**kw):
    raise RuntimeError("network down / mcp not configured")


def bad_sig_run(x):
    return str(x)


TOOLS = [
    ToolDef(name="oktool", description="d", category="fs", schema={}, run=ok_run),
    ToolDef(name="flaky", description="d", category="fs", schema={}, run=flaky_run),
    ToolDef(name="badp", description="d", category="fs", schema={}, run=bad_sig_run),
]
for t in TOOLS:
    ToolRegistry.register(t)


def main():
    # 1) 运行时异常 -> 暂时不可用回执
    r1 = ToolRegistry.dispatch({"action": "flaky"})
    assert r1.startswith("error: 工具 flaky 暂时不可用："), r1
    assert "RuntimeError" in r1, r1
    logger.error('%s %s', 'PASS [runtime-error] ->', r1)

    # 2) TypeError -> 参数错误分支（不回归）
    r2 = ToolRegistry.dispatch({"action": "badp"})  # bad_sig_run 缺 x
    assert r2.startswith("error: 工具 badp 参数错误："), r2
    logger.error('%s %s', 'PASS [type-error]   ->', r2)

    # 3) 未知动作
    r3 = ToolRegistry.dispatch({"action": "nope"})
    assert "未知动作" in r3, r3
    logger.info('%s %s', 'PASS [unknown]      ->', r3)

    # 4) 正常工具
    r4 = ToolRegistry.dispatch({"action": "oktool"})
    assert r4 == "ok-result", r4
    logger.info('%s %s', 'PASS [ok]           ->', r4)

    logger.info('%s', 'ALL OK')


if __name__ == "__main__":
    main()
