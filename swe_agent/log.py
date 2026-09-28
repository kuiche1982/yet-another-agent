#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/log.py —— harness 统一日志（基于 Python 标准库 logging）。

替代旧的手搓 trace.py（裸 print + 自定义 env 开关 SWE_TRACE）。用 logging 级别表达观测
粒度，不再自己造开关：

- logger.debug    : 详细追踪（旧 trace()）。默认不显示；LOGLEVEL=DEBUG 或 set_trace(True) 打开。
- logger.info     : 正常运行进度（analyzer_start / loop 结束 / 电池进度等）。默认显示。
- logger.warning  : 非致命异常、retry、模型慢。
- logger.error    : 调用失败、reload 失败等。
- logger.critical : 根因直报（旧 critical()）。永远显示。

日志去向：
- 控制台 StreamHandler：按级别过滤，默认 INFO（debug 不显示）。
- 文件 <SWE_AGENT_HOME>/projects/<project>/logs/harness.log（FileHandler，始终 DEBUG，
  惰性创建）：完整记录，事后排查。与 LM Studio I/O 抓包（lmstudio_requests.jsonl）、
  e2e 电池日志（e2e_battery/）同处该项目的 logs/。

级别设定（启动期读取）：
- 环境变量 LOGLEVEL（DEBUG/INFO/WARNING/ERROR）优先；
- 向后兼容 SWE_TRACE=1 -> DEBUG；
- 否则默认 INFO。
运行期开关：set_trace(True/False) 即时调整控制台级别，无需重启（保留旧能力）。
"""
import logging
import os

from . import config as C

# 日志目录：项目作用域（<SWE_AGENT_HOME>/projects/<project>/logs），与 e2e 电池日志、
# LM Studio I/O 抓包同处。**绝不在 import 期创建目录**：本模块会被 pytest 收集阶段 import，
# 一 import 就 makedirs 会凭空造出「某个项目名」的目录树（旧行为即如此，路径一旦依赖
# project 名就会污染 home）。
_LOGS_DIR = str(C.LOGS_DIR)
_LOGFILE = os.path.join(_LOGS_DIR, "harness.log")


class _LazyFileHandler(logging.FileHandler):
    """首次真正写日志时才建目录 / 打开文件。

    FileHandler 自带的 delay=True 只延迟「打开文件」；父目录不存在时首次 emit 仍会
    FileNotFoundError。故这里在打开前补一次 makedirs，做到「不写日志就不落目录」。
    """

    def emit(self, record):
        if self.stream is None:      # delay=True 且尚未打开
            try:
                os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)
            except Exception:
                pass
        super().emit(record)


logger = logging.getLogger("harness")
logger.setLevel(logging.DEBUG)  # 捕获全部；由 handler 级别决定实际输出
logger.propagate = False

_FMT = logging.Formatter(
    "[%(asctime)s.%(msecs)03d] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_console = logging.StreamHandler()
_console.setFormatter(_FMT)
logger.addHandler(_console)

_file = _LazyFileHandler(_LOGFILE, encoding="utf-8", delay=True)
_file.setFormatter(_FMT)
_file.setLevel(logging.DEBUG)
logger.addHandler(_file)


def _env_level() -> int:
    """启动级别：LOGLEVEL 优先；向后兼容 SWE_TRACE=1 -> DEBUG；否则 INFO。"""
    lvl = os.environ.get("LOGLEVEL")
    if lvl:
        return getattr(logging, lvl.upper(), logging.INFO)
    if os.environ.get("SWE_TRACE") == "1":
        return logging.DEBUG
    return logging.INFO


_console.setLevel(_env_level())


def set_trace(on: bool) -> None:
    """运行期打开/关闭详细追踪（debug 级别），无需重启。"""
    os.environ["SWE_TRACE"] = "1" if on else "0"
    _console.setLevel(logging.DEBUG if on else logging.INFO)


def is_trace() -> bool:
    """实时读取 SWE_TRACE（保留旧行为：每次调用都读，支持运行期切换）。"""
    return os.environ.get("SWE_TRACE", "0") == "1"


def _clip(v, n: int = 200) -> str:
    """任意值转字符串并截断到 n 字符，避免日志刷屏。"""
    s = str(v)
    if len(s) <= n:
        return s
    return s[:n] + f"...(截断,原{len(s)}字符)"
