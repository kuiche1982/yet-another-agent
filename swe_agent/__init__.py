#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent —— 分层调度 + 插件化 Harness 架构（对应 SWE-Agent 设计文档）

模块分层：
- config.py    ：全局配置与常量（单一事实来源）
- registry.py  ：自描述工具注册中心 + 自描述传感器注册中心（架构核心抽象）
- state.py     ：全局运行态 + 运行时辅助
- tools.py     ：Worker 工具集群（自描述，注册进 ToolRegistry）
- llm_glm.py   ：GLM 远程 provider（标准 OpenAI toolcall；Planner / 故障路由）
- llm_lmstudio.py：LM Studio 本地 provider（标准 OpenAI toolcall；Executor 后端）
- models.py    ：provider / 模型目录 + 统一 chat_messages 路由（native_tools 单一能力）
- harness.py   ：客观校验层（自描述传感器流水线 + 终态校验）
- supervisor.py：调度层（Supervisor 状态机，驱动 run_agent / main）
- cli.py       ：命令行入口

设计红线（对齐设计文档）：
1. 感知层永远是代码（harness 传感器），绝不交给 LLM；
2. 调度层只消费结构化 Fact，不读原始日志；
3. 工具/传感器自描述，插件化扩展不改动内核。
"""

from .config import WORKSPACE, REPO_ROOT  # noqa: F401

__all__ = ["WORKSPACE", "REPO_ROOT"]
