#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/management.py —— 两个横切子系统（均走 per-agent hook，条件绑定）

1) ModelManager —— LM Studio 模型生命周期（per-loop load / unload）
   - 用户需求：本地模型在 loop 里每次执行时 load，执行完 unload（显存只驻留一两个模型）。
   - 绑定方式：工厂在 assemble 时按模型配置 load_unload 标志决定是否把 load/unload
     绑进 LoopConfig.hooks 的 pre_loop / post_loop。GLM/zhipu 标志为 False → 不绑 →
     Agent.run 跑 hook 时列表为空 → 零调用零开销（用户明确要求：不需要就不传 hook 动作）。
   - 端点：LM Studio 模型管理 API `POST {base}/api/v1/models/load` 与
     `POST {base}/api/v1/models/unload`（注意与对话 API 的 `/v1` 前缀不同，管理 API 走 `/api/v1`）。
     `LMSTUDIO_BASE_URL` 默认含 `/v1`，故 _mgmt_base() 会剥离 `/v1` 再拼管理路径。
     实现 fail-open：调用失败（端点不可用 / 模型不在库）仅告警、不抛异常、不影响 agent 运行。

2) ContextManager —— 长度感知的上下文压缩（before_step 中间件）
   - 用户需求：在发往模型前，按「目标模型输入期望长度」自动提取并重构大模型请求的上下文。
     当前后端 = 调用 LFM 压缩（复用 compact 通道）；后端可插拔（未来 sliding window 等）。
   - 绑定方式：作为 LoopConfig.before_step 中间件，每步发模型前拦截 messages，超阈值则压缩。
   - 顺手根治旧 bug：原 compact.py 把 model_override 散落在副驾里（静默换模型）；
     ContextManager 把「压缩后端模型」收口为显式配置，不再隐式劫持 executor 模型。

两者都是 AOP 式挂在边界上，**Agent 核心（agent.py）不直接 import 本模块**——
核心只跑 LoopConfig.hooks / before_step 里已有的 handler。复用了现有
ToolRegistry 工具机制、compact 的 LFM 压缩通道、config 的上下文窗口字段。
"""

import json
import hashlib
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import config as C
from . import models as M
from swe_agent.log import logger
from contextmgr import (
    ContextManager as RagEngine,
    Source,
    CompressionStrategy,
    BM25Embedder,
    compress_two_tier as _ctx_compress_two_tier,
    estimate_messages as _estimate_messages,
)
from contextmgr import inject as _inject
from contextmgr import layered as _layered
from contextmgr import persist as kb_persist
from contextmgr import prepare as _prepare
from .registry import tool, ToolRegistry


# ======================================================================
# ModelManager
# ======================================================================
class ModelManager:
    """LM Studio 模型生命周期管理（per-loop load/unload）。

    设计为纯机械资源管理，不碰 prompt / tools / 控制流。所有方法 fail-open。
    """

    @staticmethod
    def _mgmt_base() -> str:
        """LM Studio 管理 API 基址：剥离对话前缀 `/v1` 再拼 `/api/v1`。

        对话 API 前缀 `/v1`（如 `/v1/chat/completions`、`/v1/models`），
        但模型管理 API 前缀是 `/api/v1`（如 `/api/v1/models/load`、
        `/api/v1/models/unload`），两者不同。`LMSTUDIO_BASE_URL` 默认含 `/v1`，
        故此处剥离后再拼管理路径。
        """
        base = C.LMSTUDIO_BASE_URL
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base.rstrip("/")

    @staticmethod
    def _lmstudio_post(path: str, payload: dict) -> Optional[dict]:
        """向 LM Studio 管理 API 发 POST，fail-open：任何异常/非 2xx 仅返回 None（调用方视为 no-op）。

        path 必须以 `/api/v1/...` 形式传入（区别于对话 API 的 `/v1/...`）。
        """
        url = f"{ModelManager._mgmt_base()}{path}"
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", "replace")
            if resp.status >= 400:
                logger.info('%s', f'[ModelManager] {path} 返回 {resp.status}: {body[:200]}')
                return None
            try:
                return json.loads(body)
            except Exception:
                return {"raw": body}
        except Exception as e:
            # 当前 LM Studio 版本未暴露该端点时会走到这里 → 静默 no-op
            logger.info('%s', f'[ModelManager] {path} 调用失败（{type(e).__name__}: {e}）；若为本机 LM Studio 不支持程序化 load/unload，可忽略。')
            return None

    @classmethod
    def _is_loaded(cls, key: str) -> bool:
        """查 chat API 的 /v1/models，判断模型是否已加载（幂等 load 用）。"""
        try:
            url = f"{cls._mgmt_base()}/v1/models"
            with urllib.request.urlopen(url, timeout=10) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            return any(m.get("id") == key for m in d.get("data", []))
        except Exception:
            return False

    @classmethod
    def load(cls, model_id: str) -> None:
        """加载指定模型（仅本地 lmstudio provider 有意义；其余 provider 直接跳过）。

        管理 API 契约：`POST /api/v1/models/load`，body `{"model": "<model_key>"}`。
        幂等：已加载则跳过，避免重复 load 触发瞬态 500 / 把常驻模型作没。
        """
        spec = M.MODELS.get(model_id)
        if not spec or spec.get("provider") != "lmstudio":
            return
        key = spec.get("model_name", model_id)
        if cls._is_loaded(key):
            logger.info('%s', f'[ModelManager] {key} 已加载，跳过 load（幂等）')
            return
        logger.info('%s', f'[ModelManager] 加载本地模型：{key}')
        cls._lmstudio_post("/api/v1/models/load", {"model": key})

    @classmethod
    def unload(cls, model_id: str) -> None:
        """卸载指定模型。

        管理 API 契约：`POST /api/v1/models/unload`，body `{"instance_id": "<model_key>"}`。
        """
        spec = M.MODELS.get(model_id)
        if not spec or spec.get("provider") != "lmstudio":
            return
        key = spec.get("model_name", model_id)
        logger.info('%s', f'[ModelManager] 卸载本地模型：{key}')
        cls._lmstudio_post("/api/v1/models/unload", {"instance_id": key})

    # —— 供工厂作为 per-agent hook 绑定（ctx 参数占位以匹配 hook 签名 fn(ctx)）——
    @classmethod
    def _load_hook(cls, model_id: str) -> Callable[[Any], None]:
        def _h(_ctx):
            cls.load(model_id)
        return _h

    @classmethod
    def _unload_hook(cls, model_id: str) -> Callable[[Any], None]:
        def _h(_ctx):
            cls.unload(model_id)
        return _h


# ======================================================================
# ContextManager
# ======================================================================
def _default_compress_backend(messages: List[Dict[str, Any]], ctx: Any) -> List[Dict[str, Any]]:
    """默认压缩后端：复用 compact 的 LFM 压缩通道（长度感知，按 config 上下文窗口判定）。

    后续可插拔 sliding window / 分层摘要等：只需在工厂里换掉 backend。
    """
    from . import compact as _compact
    # 与原 supervisor._sidecar.auto_compact 行为一致：估算超阈值才压缩，否则原样返回
    return _compact.maybe_auto_compact(messages)


# ======================================================================
# contextmgr 集成：共享 KB（工厂自动扫工作区）+ 两级压缩 + RAG 组装
# ======================================================================
_KB_CACHE: dict[str, "RagEngine"] = {}
_COMPRESS_BACKEND: Optional[Callable] = None  # 二级语义兜底：LFM compact callable，工厂构造期注入

def set_compress_backend(fn: Optional[Callable]) -> None:
    """工厂在构造期注入 LFM 语义压缩兜底（本地 lmstudio → 接 compact；远程/无模型 → None）。"""
    global _COMPRESS_BACKEND
    _COMPRESS_BACKEND = fn


# ======================================================================
# 分层本地 RAG 知识库（code 代码 / proj 项目KB / global 公共KB 三个独立持久实例）
# ======================================================================
# 实现（三层门面 + 跨层合并语料重排 + 磁盘增量缓存）已下沉 `contextmgr.layered`：
# 分层策略与缓存目录策略属内容管理。本文件只留**配置适配层**（下面的 LayeredKB 子类）
# 与按项目键控的实例注册表 —— 即「读 config → 构造期注入参数」这段接线。
# 下面四个常量是 contextmgr 侧常量表的同名别名（兼容既有引用与测试）。
_INGEST_IGNORE_DIRS = _layered.INDEX_IGNORE_DIRS
_CODE_EXTS = _layered.CODE_EXTS
_LIB_EXTS = _layered.LIB_EXTS
_KB_INGEST_IGNORE_DIRS = _layered.KB_INGEST_IGNORE_DIRS


def _make_rerank_llm():
    """可选模型重排后端（注入给 contextmgr）。

    判定全部在**调用期**做：未启用 / 模型未加载 → 返回 None（contextmgr 侧保持合并 BM25 序）。
    """
    def _llm(system: str, user: str):
        if not getattr(C, "RAG_RERANK", False):
            return None
        model = C.RAG_RERANK_MODEL
        if not ContextManager._summary_model_ready(model):
            return None
        return M.chat_text(role="analyzer", system=system, user=user, temperature=0.0,
                           max_tokens=max(32, int(getattr(C, "RAG_RERANK_MAX_TOKENS", 512))),
                           model_override=model)
    return _llm


class LayeredKB(_layered.LayeredKB):
    """配置适配层：从 config 取根路径/缓存目录/扩展集/重排后端，注入 contextmgr 的实现类。

    实现（三层门面 + 跨层合并语料重排 + 磁盘增量缓存）全在 `contextmgr.layered.LayeredKB`。
    这里刻意保持**薄**：只做「读 config → 构造期注入」——因为 contextmgr 侧不得 import config
    （否则值会被冻结进其模块命名空间，monkeypatch / e2e 改 env 全部失效）。
    """

    @property
    def rerank_llm(self):
        """覆写父类 property：**每次调用现读 config**，不走构造期快照。"""
        return _make_rerank_llm()

    def __init__(self):
        super().__init__(
            code_root=str(C.WORKSPACE),
            code_cache=str(C.CODE_KB_DIR),
            proj_roots=[str(p) for p in C.KB_ROOTS],
            proj_cache_dir=str(C.PROJ_KB_DIR),
            global_roots=[str(p) for p in C.KB_GLOBAL_ROOTS],
            global_cache_dir=str(C.GLOBAL_KB_DIR),
            code_ext=set(_CODE_EXTS),
            lib_ext=set(_LIB_EXTS),
            code_ignore=set(_INGEST_IGNORE_DIRS),
            kb_ignore=set(_KB_INGEST_IGNORE_DIRS),
            rerank_candidates=int(C.RAG_RERANK_CANDIDATES),
        )


_LAYERED_KB_BY_PROJECT: "dict[str, LayeredKB]" = {}


def _get_layered_kb() -> "LayeredKB":
    """按项目作用域取分层知识库（首次调用触发磁盘增量构建，之后复用同一实例）。"""
    key = C.PROJECT_NAME
    kb = _LAYERED_KB_BY_PROJECT.get(key)
    if kb is None:
        kb = LayeredKB()
        _LAYERED_KB_BY_PROJECT[key] = kb
    return kb


def _ref_location(kb, f) -> tuple[str, str]:
    """解析片段的真实出处与 read_file 指针（注入层与 local_search 共用，保证口径一致）。

    实现已下沉 `contextmgr.inject.ref_location`（内容管理归 contextmgr）；此处保留同名
    转发以兼容既有调用点与测试口径。
    """
    return _inject.ref_location(kb, f)


# 模型重排输出的严格解析已下沉 contextmgr.layered（保留同名别名，兼容既有调用点与测试）
_parse_order = _layered.parse_order


@tool(
    name="local_search",
    description=(
        "在本地知识库与代码库中检索相关资料（优先于联网搜索 web_search）。"
        "分层覆盖：项目代码(code)、项目私有知识库(kb，默认 KnowledgeBase/ 与 docs/)、"
        "公共知识库(global)。结果按相关性（合并语料 BM25，跨层可比）降序返回，每条给出"
        "「出处文件:行号 + 正文片段 + read_file 指针」：片段够判断相关性，要读全文用指针。"
        "可用 offset 翻页做多轮细化检索。"
    ),
    category="fs",
    schema={"type": "object", "properties": {
        "query": {"type": "string", "description": "检索关键词或自然语言问题"},
        "scope": {"type": "string", "enum": ["all", "code", "kb", "global"],
                  "description": ("检索范围：all=三层都查并按相关性取全局前 top_k；"
                                 "code=仅代码；kb=仅项目知识库；global=仅公共知识库。默认 all")},
        "top_k": {"type": "integer", "description": "返回条数上限（默认 5）"},
        "offset": {"type": "integer", "description": "分页偏移，跳过前 offset 条（默认 0；看下一批就把它加上已看的条数）"},
    }, "required": ["query"]},
    examples=[
        '{"action":"local_search","query":"Judge-Mode 是什么"}',
        '{"action":"local_search","query":"如何实现上下文压缩","scope":"kb"}',
        '{"action":"local_search","query":"read_file 实现","scope":"code","top_k":5}',
        '{"action":"local_search","query":"检索排序","top_k":5,"offset":5}',
    ],
    when_to_use="需要查找本地代码实现、设计文档、知识库条目时优先用本工具；不要先用 web_search。",
)
def local_search(query: str, scope: str = "all", top_k: int = 0, offset: int = 0) -> str:
    """在本地分层知识库检索，返回「出处文件:行号 + 正文片段 + read_file 指针」。

    - 片段按**行/段落边界**截断到 C.LOCAL_SEARCH_SNIPPET_CHARS（不是硬切字符），
      足够 agent 判断相关性；要读全文用返回的 read_file 指针。
    - offset 支持分页（先看前 k 条，再要下一批），供 agentic 多轮细化检索。
    - 默认条数 C.LOCAL_SEARCH_TOP_K，与被动注入的 TOP_N 对齐（避免「工具给 N 条 →
      agent 读 N 个文件」的 1:1 耦合）。

    fail-open：知识库不可用或为空时返回友好提示，不抛异常、不阻断 agent。
    """
    try:
        k = max(1, int(top_k) if top_k else int(C.LOCAL_SEARCH_TOP_K))
        off = max(0, int(offset or 0))
        kb = _get_layered_kb()
        results = kb.search(query, scope=scope, top_k=k, offset=off)
    except Exception as e:
        return f"local_search_error: 检索失败：{e}"
    if not results:
        return ("local_search: 未找到相关本地资料（可换关键词或 scope，也可用 offset 翻页；"
                "或确认 KB_ROOTS / KB_GLOBAL_ROOTS 已配置并包含目标文档）。")
    return _inject.format_local_search(
        results, kb=kb, scope=scope, offset=off,
        snippet_chars=int(C.LOCAL_SEARCH_SNIPPET_CHARS))


class ContextManager:
    """对话 buffer 的**有状态拥有者** + 长度感知压缩 + 循环防护（fences）状态。

    - 持有 self._msgs（对话历史），对外暴露 append/add/set_system/reset/to_list；
      循环体（Agent 单步 / supervisor 回调）一律经 ctx.cm 读写，不再裸透传 List[Dict]。
    - 压缩是内部不变量：compress_if_needed() 估算超阈值才调 backend（默认 LFM 副驾），
      否则原样。时序铁律（先压后 load 工作模型）由调用方在 round 起始处调一次落实，
      压缩完 LFM 即自卸。
    - 循环防护（fences）已迁出本类：具名限制 registry 见 swe_agent/guard.py，
      由 RunState.guard 持有、Agent._run_loop 按 scope 自动重置（不再依赖 cm）。
    """

    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None,
                 backend: Optional[Callable] = None, threshold: Optional[int] = None,
                 kb: Optional["RagEngine"] = None,
                 kb_budget: int = 4096,
                 kb_strategy: "CompressionStrategy" = CompressionStrategy.COMPRESS,
                 compress_backend: Optional[Callable] = None):
        self._msgs = list(messages or [])
        # 真相日志（L0 truth）：永远记「原文」（压缩前的完整内容）。_msgs 是被压缩/裁剪的
        # 工作集（发往模型用）；_truth 是完整会话史，供 recall 与「会话历史不丢失」保证。
        # 二者仅在「压缩后的工具结果」上不同：_msgs 记压缩版，_truth 记原文。
        self._truth: list[dict] = list(messages or [])
        self.backend = backend or _default_compress_backend
        # 目标预算：默认按 config 上下文窗口 - 自动压缩缓冲
        self.threshold = threshold if threshold is not None else (
            C.CONTEXT_WINDOW - C.AUTOCOMPACT_BUFFER)
        # —— contextmgr 集成（RAG 引擎 + 压缩引擎）——
        # kb 默认挂「分层持久知识库 LayeredKB（code/proj/global 三独立实例 + 磁盘增量缓存）」；
        # 首次构造触发一次增量构建，之后跨 run 复用，根治「整仓同步 ingest 拖垮首轮 RAG」的债。
        # 压缩 = 两级：model-free 一级（contextmgr.compress）→ LFM 语义二级兜底（compress_backend，构造期注入）。
        self.kb = kb or _get_layered_kb()
        self.kb_budget = kb_budget
        self.kb_strategy = kb_strategy
        self.compress_backend = (
            compress_backend if compress_backend is not None
            else (_COMPRESS_BACKEND or backend))
        # 增量渲染钩子（REPL 流式输出用）：每次 append/add 立即回调，实现「边跑边出」，
        # 而不是等整轮 run() 结束再一次性打印。UNATTEND 路径从不设置 → None → 零开销零影响。
        self.on_message: Optional[Callable[[Dict[str, Any]], None]] = None

    # —— buffer 读写 ——
    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self._msgs

    def to_list(self) -> List[Dict[str, Any]]:
        return self._msgs

    def _emit(self, msg: Dict[str, Any]) -> None:
        """增量渲染钩子：新消息入 buffer 即刻回调（REPL 流式输出）。

        fail-open：显示回调抛异常绝不阻断 agent 主链路（渲染是旁支，不是真相源）。
        """
        hook = self.on_message
        if hook is None:
            return
        try:
            hook(msg)
        except Exception as e:  # 显示层异常不影响执行
            logger.debug('on_message hook error: %s', e)

    def append(self, role: str, content: Any, *, original: Any = None,
               tool_call_id: Any = None, **kw) -> "ContextManager":
        """追加一条消息到工作集（_msgs）与真相日志（_truth）。

        - _msgs：发往模型的「工作集」，可被 compress_if_needed 压缩/裁剪（含抽取式压缩后的工具结果）。
        - _truth：永远记原文（original 优先，否则 content），不随压缩丢失 —— 这是 recall 与
          「会话历史不丢失」的数据源。
        - original 由 agent 在抽取式压缩前传入（见 agent.py _apply_toolcall），保证压缩掉的
          工具结果原文仍可回溯。
        """
        msg = {"role": role, "content": content}
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id
        msg.update(kw)
        self._msgs.append(msg)

        truth_content = content if original is None else original
        tmsg = {"role": role, "content": truth_content}
        if tool_call_id is not None:
            tmsg["tool_call_id"] = tool_call_id
        tmsg.update(kw)
        self._truth.append(tmsg)

        self._emit(msg)
        return self

    def add(self, msg: dict) -> "ContextManager":
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            original = msg.get("_original", None)
            kw = {k: v for k, v in msg.items()
                  if k not in ("role", "content", "_original")}
            self.append(role, content, original=original, **kw)
        return self

    def truth_list(self) -> list[dict]:
        """返回完整会话史（原文，未压缩）—— recall / 会话恢复用。"""
        return self._truth

    def set_system(self, content: str) -> "ContextManager":
        if self._msgs and self._msgs[0].get("role") == "system":
            self._msgs[0]["content"] = content
        else:
            self._msgs.insert(0, {"role": "system", "content": content})
        return self

    def reset(self, messages: Optional[List[Dict[str, Any]]]) -> "ContextManager":
        self._msgs = list(messages or [])
        return self

    # —— 长度感知压缩（内部不变量，两级）——
    def compress_if_needed(self, ctx: Any = None) -> "ContextManager":
        est = self._est_msgs(self._msgs)
        if est >= self.threshold:
            self._msgs = self._compress_two_tier(self._msgs, self.threshold)
        return self

    def as_hook(self) -> Callable[[Any], None]:
        """返回 before_step 签名 (ctx) -> None 的 handler（压缩当前 buffer）。"""
        def _h(ctx):
            self.compress_if_needed(ctx)
        return _h

    # —— contextmgr 接入：ingest / RAG / 两级压缩 / prepare_messages ——
    def ingest_code_file(self, path: str, name: Optional[str] = None) -> "ContextManager":
        """把单个代码文件喂进 RAG 知识库（L0: Source=Code，按符号切分）。"""
        if self.kb is not None and hasattr(self.kb, "ingest_code_file"):
            self.kb.ingest_code_file(path, name=name)
        return self

    def ingest_library(self, text: str, doc_id: str = "doc") -> "ContextManager":
        """把文档文本喂进 RAG 知识库（L0: Source=Library）。"""
        if self.kb is not None and hasattr(self.kb, "ingest_library"):
            self.kb.ingest_library(text, doc_id=doc_id)
        return self

    def ingest_workspace(self, root=None, exts=None, max_lines: int = 2000) -> "ContextManager":
        """分层 KB 在构造期已通过 persist.rebuild 建好；运行时增量重建走 kb.rebuild_all()。"""
        if self.kb is not None and hasattr(self.kb, "rebuild_all"):
            try:
                self.kb.rebuild_all()
            except Exception:
                pass
        return self

    # —— 长度/预算工具（实现收口于 contextmgr.tokenize.estimate_messages）——
    @staticmethod
    def _est_msgs(messages) -> int:
        return _estimate_messages(messages)

    # —— 两级压缩：model-free 一级 → LFM 语义二级兜底 ——
    def _compress_two_tier(self, messages, budget: int):
        """**薄转发** `contextmgr.compress_two_tier`（策略在 contextmgr，后端由 harness 注入）。

        唯一的 harness 侧输入是 `self.compress_backend`（副驾/LFM 压缩通道）；把它透传过去，
        使「单轮/整组溢出」时能升级大上下文模型语义压缩（不丢 user、不砍 tool）。
        """
        return _ctx_compress_two_tier(list(messages), budget,
                                      compress_backend=self.compress_backend)

    # —— RAG 检索（model-free BM25；只查 library/code，排序由 LayeredKB 跨层归一化给出）——
    def _rag_refs(self, query: str, rag_budget: int):
        """检索本地资料/代码，按全局相关性**精选前 TOP_N 条**。

        只取前 N 条是「注入正文」模式的前提：注入 N 条正文后 agent 无需再逐个 read_file，
        从而打破「N 条文件指针 → N 次 read_file（且被单轮 5-工具上限丢弃）」的 1:1 耦合。
        """
        if self.kb is None:
            return []
        try:
            refs = self.kb.retrieve(query, budget_tokens=rag_budget,
                                   sources={Source.LIBRARY, Source.CODE})
        except Exception:
            return []
        top_n = max(0, int(C.RAG_INJECT_TOP_N))
        return _inject.select_refs(refs, top_n)

    # —— 注入层：把检索到的 L0 正文按「出处 + 行号」打包（给正文，agent 不必再读文件）——
    def _ref_location(self, f) -> tuple[str, str]:
        """解析片段的真实出处与 read_file 指针（委托模块级 _ref_location，与 local_search 同口径）。"""
        return _ref_location(self.kb, f)

    @staticmethod
    def _clip(text: str, max_chars: int) -> tuple[str, bool]:
        """抽取式截断（保头 + 保尾）。

        实现已下沉 `contextmgr.inject.clip`；此处保留同名转发，兼容既有调用点与测试。
        """
        return _inject.clip(text, max_chars)

    @staticmethod
    def _summary_model_ready(model: str) -> bool:
        """副驾摘取模型是否可用：本地模型要求已加载（否则请求会打到未加载模型上白等重试）。

        仅在「确有超长片段需要摘取」时才被调用（惰性），故短文本路径零网络开销、测试保持 model-free。
        """
        spec = M.MODELS.get(model)
        if not spec:
            return False
        if spec.get("provider") != "lmstudio":
            return True   # 远程模型无需本地加载
        return ModelManager._is_loaded(spec.get("model_name", model))

    @staticmethod
    def _summarize_one(query: str, text: str, model: str) -> str:
        """调副驾模型做「针对用户问题的摘取式总结」（必要时才用）。

        任何失败 / 可疑输出 → 返回空串，由调用方回退抽取式截断。出处行号由代码写入，不经模型，
        故模型即便出错也不会污染 provenance。
        """
        system = ("你是检索结果提炼器。只输出提炼后的要点，不要开场白、不要 Markdown 代码块围栏。"
                  "必须保留原文中的具体事实：函数名/变量名/数值/命令/结论/接口签名；"
                  "不得编造、不得翻译、不得添加原文没有的内容。")
        user = (f"用户问题：{query}\n\n从下面的文档片段中摘出与问题相关的要点，"
                f"与问题无关的信息全部丢弃；若整段都与问题无关，只输出「（与问题无关）」。\n\n"
                f"【文档片段】\n{text}")
        out = M.chat_text(role="analyzer", system=system, user=user,
                          temperature=0.2,
                          max_tokens=max(64, int(C.RAG_INJECT_SUMMARY_MAX_TOKENS)),
                          model_override=model)
        out = (out or "").strip()
        if out.startswith("```"):   # 弱模型偶发 ``` 围栏
            parts = out.split("```", 2)
            out = parts[1] if len(parts) > 1 else out
            if out.startswith("json"):
                out = out[4:]
            out = out.strip()
        return out

    def _inject_options(self) -> "_inject.InjectOptions":
        """从 config 组装注入参数（**调用期**读取：monkeypatch / e2e 改 env 即时生效）。

        刻意不在 contextmgr 侧 import config —— 否则值会被冻结进其模块命名空间。
        """
        return _inject.InjectOptions(
            max_chars=int(C.RAG_INJECT_MAX_CHARS),
            allow_summary=bool(C.RAG_INJECT_SUMMARY),
            summary_min_tokens=int(C.RAG_INJECT_SUMMARY_MIN_TOKENS),
            summary_max_calls=int(C.RAG_INJECT_SUMMARY_MAX_CALLS),
            summary_model=C.RAG_INJECT_SUMMARY_MODEL,
            snippet_chars=int(C.LOCAL_SEARCH_SNIPPET_CHARS),
            top_n=int(C.RAG_INJECT_TOP_N),
        )

    def _summarize_backend(self):
        """返回可注入 contextmgr 的摘取后端（模型侧知识留在 harness，不污染 contextmgr）。"""
        def _fn(query: str, text: str, model: str) -> str:
            return self._summarize_one(query, text, model)
        return _fn

    def _rag_block(self, refs, query: str, budget_tokens: int) -> str:
        """把检索结果打包成注入块（实现已下沉 `contextmgr.inject.rag_block`）。

        - 相关性：上游 LayeredKB 已跨层全局归一化排序（高分在前）。
        - 正文：默认直接注入 L0 原文片段（agent 无需再 read_file）；单条超 MAX_CHARS 时，
          若副驾模型可用则先做「针对问题的摘取式总结」，失败再抽取式 head+tail 截断。
        - 出处行号恒由代码写入 → 永不丢失、不会被模型编造。
        """
        return _inject.rag_block(refs, query, self._inject_options(), kb=self.kb,
                                 summarize=self._summarize_backend(),
                                 summarize_ready=self._summary_model_ready)

    @staticmethod
    def _strip_reasoning(m: Dict[str, Any]) -> Dict[str, Any]:
        """出向消息清理（仅剥副本，不改动 buffer）：

        - 剥除 reasoning_content（模型 output-only 字段，回发会触发 LM Studio 400）。
        - 带 tool_calls 的 assistant 消息 content 归零为 None（OpenAI 协议规范：tool-call 消息
          content 应为 null；同时保证 UNATTEND 回发历史与改动前一致——tool 轮 content 恒为 None，
          不因本修改把模型思考文本回灌进 executor 上下文）。
        REPL 渲染读取的是 cm.to_list() 的 buffer（未经本函数），故不受影响。
        """
        m = {k: v for k, v in m.items() if k != "reasoning_content"}
        if m.get("role") == "assistant" and m.get("tool_calls") and m.get("content"):
            m = {**m, "content": None}
        return m

    # —— 召回：从完整会话史（_truth，原文未压缩）按需取回详细信息 ——
    def recall(self, query: str, budget: Optional[int] = None) -> List[Dict[str, Any]]:
        """**薄转发** `contextmgr.prepare.recall_from_truth`。

        这是「压缩过后仍能在后续需要时 recall 详细信息」的能力来源——工作集 _msgs 被压缩/裁剪，
        但 _truth 永远保留原文，这里按 query 把被压缩掉的历史细节重新取回。

        harness 侧只负责给出「真相日志 + 预算口径」（默认取 `self.kb_budget`，与 RAG 预算同源）；
        打分算法在 contextmgr（与 `contextmgr/manager.py` 共用同一份实现，不再各写一遍）。
        """
        return _prepare.recall_from_truth(self._truth, query, budget or int(self.kb_budget))

    # —— 对外主接口：按 model_context_length + user_input 自动决策 RAG/压缩/滑动窗口 ——
    def prepare_messages(self, model_context_length: int = 64000,
                         user_input: str = "",
                         recall_query: str = "") -> List[Dict[str, Any]]:
        """**薄转发** `contextmgr.prepare.prepare_messages`（组装算法已下沉 contextmgr）。

        harness 侧只注入「后端的接线」——检索、注入块组装、两级压缩、召回、出向清理：
        - `rag_refs` / `rag_block`：LayeredKB 检索 + 出处行号注入（含可选副驾摘取）；
        - `compress_fn`：`self._compress_two_tier`（压缩后端 = `self.compress_backend`）；
        - `truth`：真相日志（未压缩原文），供 opt-in 召回；
        - `sanitize`：`self._strip_reasoning`（OpenAI 协议出向清理）。

        对 self._msgs **只读**：返回组装视图，不缩 buffer（循环防护计数仍按全量历史算，归 harness）。
        组装算法与不变量（如「tool 轮不重复注入 user」）见 contextmgr/prepare.py。
        """
        return _prepare.prepare_messages(
            self._msgs,
            model_context_length=model_context_length,
            user_input=user_input,
            recall_query=recall_query,
            truth=self._truth,
            rag_refs=self._rag_refs,
            rag_block=self._rag_block,
            compress_fn=self._compress_two_tier,
            recall_fn=self.recall,
            sanitize=self._strip_reasoning,
        )
