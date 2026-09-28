#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py —— 向后兼容启动器（原单文件 Agent 已拆分为 swe_agent/ 分层包）

原 demo.py（5518 行）已按「SWE-Agent 分层调度 + 插件化 Harness」架构拆分：
    swe_agent/
    ├── config.py     全局配置与常量（单一事实来源）
    ├── registry.py   自描述工具注册中心 + 自描述传感器注册中心（架构核心抽象）
    ├── state.py      全局运行态 + 运行时辅助
    ├── tools.py      Worker 工具集群（自描述，注册进 ToolRegistry）
    ├── llm_glm.py    GLM 远程 provider（标准 OpenAI toolcall；Planner / Reviewer / 故障路由）
    ├── llm_lmstudio.py LM Studio 本地 provider（标准 OpenAI toolcall；Executor 后端）
    ├── models.py     provider / 模型目录 + 统一 chat_messages 路由（native_tools 单一能力）
    ├── harness.py    客观校验层（自描述传感器流水线 + 终态校验）
    ├── workers.py    Worker 编排（planner / executor / reviewer 封装）
    ├── supervisor.py 调度层（Supervisor 状态机，驱动 run_agent / main）
    └── cli.py        命令行入口

运行方式：
    python demo.py "你的开发任务"          # 批处理
    python -m swe_agent "你的开发任务"     # 等价入口
    python demo.py                        # 进入交互 REPL

说明：原 demo 历史副本见 demo copy*.py（backup/ 已清理）。
"""

import os
import sys

# 确保仓库根目录在 sys.path，使 `import swe_agent` 可用
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from swe_agent.supervisor import main

if __name__ == "__main__":
    main()
