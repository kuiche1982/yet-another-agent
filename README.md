# litertlm

个人学习项目：用本地模型（litert-lm / MLX 量化）动手实现和践行 agent harness 的各种方法——SWE-agent 式的任务编排、上下文管理、记忆、评测与插件化。**实验性质，仅供参考，不保证可用，勿用于生产或关键任务。**

## 是什么

一个从零搭建的软件工程 agent harness，核心思路：

- **编排**：`swe_agent/supervisor.py` 装配三层嵌套循环（L1 attempt → L2 round → L3 step），在每层设闸门（pytest / tester / lint）控制任务推进与回退。
- **钩子与守卫**：`swe_agent/hooks.py` 统一事件总线（HOOK_HUB），工具调用前后可挂 guard（写入/读取大小、stall、unsolvable 等），实现"工具调用前拦截、调用后改写"。
- **工具注册**：`swe_agent/tools.py` 的 ToolRegistry 负责工具分发与安全边界。
- **上下文管理**（`contextmgr/`）：压缩、蒸馏、分块、检索与 RAG 注入，控制长任务中的上下文开销。
- **记忆**（`memory/`）：增量递归摘要树、记忆分类、BGE-M3 向量检索、持久化。
- **评测**（`judge/` + `tests/verify_*.py`）：针对意图识别、PII 探测、结构化输出、LSP 联动等的探测与回归。
- **插件**（`plugins/`）：Claude Code 风格的 config-only / tool 插件（lsp、ocr、web_fetch、websearch）。
- **语音输入**（`voiceinput/`）：macOS 录音 → whisper.cpp 转写，供无人值守链路消费。

## 目录

```
swe_agent/    agent harness 主体（supervisor、layers、guards、tools、roles、plugins、skills、LSP、MCP）
contextmgr/   上下文管理与 RAG 注入
memory/       记忆系统（摘要树、分类、检索）
judge/        意图/边界/探测类评测脚本
plugins/      插件（lsp、ocr、web_fetch、websearch）
scripts/      实验与端到端评测脚本（fib、conway、go bench、e2e 等）
docs/         架构设计与实验记录（以代码为准）
tests/        pytest 用例 + verify_*.py 探测
examples/     Go 示例（game of life、todo）
voiceinput/   Mac 语音输入转写工具
```

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest            # 跑测试（tests/ + verify_*.py）
```

需要本地 LLM 推理（litert-lm 0.15 / MLX 量化模型），密钥与模型路径通过 `.env`（已 gitignore）注入，代码内只保留占位值。

## 说明

- 本项目是**学习笔记性质的代码**，大量取舍面向"验证某一种方法是否可行"，不追求工程完备性。
- 部分脚本与文档带有明显实验痕迹（临时脚本、一次性评测记录），请按需取用。
- 更多设计思路见 `docs/ARCHITECTURE.md` 与 `docs/harness_design_map.md`。

## License

见 [LICENSE](./LICENSE)。
