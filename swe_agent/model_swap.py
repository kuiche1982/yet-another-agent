#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/model_swap.py —— LM Studio 模型换模编排（BUILD 层工程能力）

问题背景：本机显存下，压缩副驾（lfm2.5-2.6b）与主 executor（qwen2.5.1-coder-7b-instruct）
无法同时驻留显存。对话压缩（compact.py）需要副驾，而 executor 平时常驻。
因此压缩前必须：unload qwen → load 2.6b，压缩后：unload 2.6b → reload qwen。

本模块提供：
  - load_model / unload_model：对 LM Studio /api/v1/models/{load,unload} 的薄封装；
  - SidecarCompressSession：上下文管理器，进入时换模、退出时（含异常）还原 executor；
  - with_sidecar_compression：函数式包装，把任意「压缩动作」包进一次换模会话。

所有操作失败一律 fail-open：换模异常会被记录但【不静默吞掉数据】；退出时无论如何
都尽量把 executor（qwen）重新加载回来，保证下游 agent 还能继续干活。

依赖：仅标准库（urllib），无第三方包要求，可在 managed python 下运行。
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

from . import config as C
from swe_agent.log import logger

LMSTUDIO_BASE = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234").rstrip("/")

# 默认副驾 / 主 executor（在 LM Studio 中的模型 key）。
DEFAULT_SIDECAR = C.SIDECAR_COMPRESS_MODEL          # lfm2.5-2.6b
DEFAULT_EXECUTOR = C.EXECUTOR_MODEL                 # qwen2.5.1-coder-7b-instruct
DEFAULT_CTX = C.SIDECAR_COMPRESS_CTX
DEFAULT_FLASH = C.SIDECAR_COMPRESS_FLASH


# ----------------------------------------------------------------------
# 底层 HTTP 封装
# ----------------------------------------------------------------------
def _api(method: str, path: str, payload: Optional[dict] = None, timeout: int = 180):
    url = f"{LMSTUDIO_BASE}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace")), resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio {method} {path} -> HTTP {e.code}: {body[:300]}")
    except Exception as e:  # 连接失败 / 超时 等
        raise RuntimeError(f"LM Studio {method} {path} -> {e}")


def list_loaded() -> list:
    """当前已加载的模型 id 列表。"""
    try:
        out, _ = _api("GET", "/v1/models", timeout=10)
        return [m.get("id") for m in out.get("data", [])]
    except Exception:
        return []


def load_model(model_key: str, context_length: Optional[int] = None,
               flash_attention: bool = DEFAULT_FLASH, timeout: int = 240) -> dict:
    """加载一个模型。返回 LM Studio 的 load 响应。已加载则幂等跳过。"""
    if is_loaded(model_key):
        logger.info('%s', f'[model_swap] {model_key} 已加载，跳过 load（幂等）')
        return {"status": "already_loaded", "instance_id": model_key}
    body = {"model": model_key, "echo_load_config": True}
    if context_length:
        body["context_length"] = context_length
    if flash_attention:
        body["flash_attention"] = True
    out, _ = _api("POST", "/api/v1/models/load", body, timeout=timeout)
    # LM Studio 偶尔异步加载：轮询直到出现在本实例列表
    for _ in range(20):
        if model_key in list_loaded():
            return out
        time.sleep(0.5)
    return out


def unload_model(instance_id: str, timeout: int = 60) -> dict:
    """卸载一个已加载实例。instance_id 即模型 key。"""
    out, _ = _api("POST", "/api/v1/models/unload", {"instance_id": instance_id}, timeout=timeout)
    return out


def is_loaded(model_key: str) -> bool:
    return model_key in list_loaded()


# ----------------------------------------------------------------------
# 换模会话（核心）
# ----------------------------------------------------------------------
class SidecarCompressSession:
    """上下文管理器：进入时卸主模型载副驾，退出时（含异常）还原主模型。

    用法：
        with SidecarCompressSession():
            summary = M.chat_text_messages(..., model_override="lfm2.5-2.6b")
    """

    def __init__(self, sidecar: str = DEFAULT_SIDECAR, executor: str = DEFAULT_EXECUTOR,
                 ctx: int = DEFAULT_CTX, flash: bool = DEFAULT_FLASH, enabled: bool = True):
        self.sidecar = sidecar
        self.executor = executor
        self.ctx = ctx
        self.flash = flash
        self.enabled = enabled
        self._restored = False

    def __enter__(self):
        if not self.enabled:
            return self
        # 1) 卸主 executor（若已加载），给副驾腾显存
        if is_loaded(self.executor):
            try:
                unload_model(self.executor)
            except Exception as e:
                logger.info('%s', f'[model_swap] 卸载主模型 {self.executor} 失败（{e}），继续尝试加载副驾。')
        # 2) 载副驾（若未加载）
        if not is_loaded(self.sidecar):
            try:
                load_model(self.sidecar, self.ctx, self.flash)
            except Exception as e:
                # 副驾加载失败：__enter__ 抛异常时 __exit__ 不会被调用，
                # 必须在这里手动把主模型重新加载回来，否则 executor 会残留未加载态。
                try:
                    if not is_loaded(self.executor):
                        load_model(self.executor, self.ctx, self.flash)
                except Exception:
                    pass
                raise RuntimeError(f"[model_swap] 加载副驾 {self.sidecar} 失败：{e}")
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        if self.enabled:
            self.restore()
        return False  # 不吞异常

    def restore(self):
        """把 executor（qwen）重新加载回来。幂等、容错。"""
        if self._restored:
            return
        # 先卸副驾，释放显存
        if is_loaded(self.sidecar):
            try:
                unload_model(self.sidecar)
            except Exception as e:
                logger.info('%s', f'[model_swap] 卸载副驾 {self.sidecar} 失败（{e}）。')
        # 再载主 executor（务必还原，保证 agent 继续可用）
        if not is_loaded(self.executor):
            try:
                load_model(self.executor, self.ctx, self.flash)
            except Exception as e:
                logger.info('%s', f'[model_swap] ⚠️ 还原主模型 {self.executor} 失败（{e}）！executor 可能不可用，请手动 load。')
        self._restored = True


def with_sidecar_compression(action: Callable, *args,
                             sidecar: str = DEFAULT_SIDECAR,
                             executor: str = DEFAULT_EXECUTOR,
                             enabled: bool = True, **kwargs):
    """把任意「压缩动作」包进一次换模会话。

    with_sidecar_compression(lambda: compress(...))  # 自动 unload qwen → load 2.6b → 压缩 → 还原
    """
    with SidecarCompressSession(sidecar=sidecar, executor=executor, enabled=enabled):
        return action(*args, **kwargs)


if __name__ == "__main__":
    # 自检：打印当前已加载模型 + 注册表 key
    logger.info('%s %s', 'LMSTUDIO_BASE =', LMSTUDIO_BASE)
    logger.info('%s %s', 'loaded       =', list_loaded())
    logger.info('%s %s', 'sidecar      =', DEFAULT_SIDECAR)
    logger.info('%s %s', 'executor     =', DEFAULT_EXECUTOR)
