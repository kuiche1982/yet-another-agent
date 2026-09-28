#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pre-test.py —— swe_agent × 模型 兼容性预检（手工运行，零改盘）

把 swe_agent 里【所有对大模型的出向请求形态】抽成一组探针，逐个打向当前后端，
在真正跑 e2e 之前回答一个问题：**这个模型/后端能不能撑住 harness 的全部调用方式**。

覆盖的请求形态（每条都对应 harness 里一处真实调用点）：

  #  探针                           harness 来源
  --  ----------------------------  --------------------------------------------------
   0  server.reachable              GET /v1/models（连通性）
   1  conn.reuse                    同一 keep-alive 连接连发 2 次（后端连接复用 bug 探测，命中即自动规避）
   1  catalog.registered            models.MODELS 目录（未登记 → ModelUnknown / judge 直接 notsure）
   2  model.listed                  目标 id 在 /v1/models 列表里（本地模型是否已加载）
   3  chat.plain                    compact/摘要：chat_text_messages（无 tools、无 response_format）
   4  chat.json_schema              planner：_PLANNER_JSON_SCHEMA + strict=true（response_format）
   5  chat.json_object              降级形态（GLM 用；本地 LM Studio 实测拒收 → 记 INFO 不判死）
   6  tools.auto                    executor 一轮：工具全集 + tool_choice="auto"
   7  tools.required_single         analyzer 收口：裁剪到单工具 + tool_choice="required"
   8  tools.roundtrip               第二轮回灌：assistant(tool_calls) + tool(tool_call_id) → 是否 400
   9  tools.parallel                一轮多 tool_call（parallel_tool_calls=C.PARALLEL_TOOL_CALLS）
  10  judge.schema                  models.judge：system="" + JUDGE_SCHEMA + temperature=0
  11  extra.thinking_off            extra_body.chat_template_kwargs.enable_thinking=False（Ling）
  12  param.max_tokens              LMSTUDIO_MAX_TOKENS 是否被接受
  13  resp.reasoning_content        响应是否带 reasoning_content（harness 出向会剥除，仅观测）
  14  compat.tool_choice_function   OpenAI 风格 function 对象 tool_choice（harness 已裁剪规避）

判定：
  BLOCKER 探针（3/4/6/7/8/10，外加 0）任一 FAIL → INCOMPATIBLE
  其余 FAIL/异常 → WARN；只有 WARN → DEGRADED；全绿 → READY

用法：
  .venv/bin/python pre-test.py                      # 测 harness 当前解析到的全部角色模型（去重）
  .venv/bin/python pre-test.py --model Ling-3.0-Tiny
  .venv/bin/python pre-test.py --auto               # 测 /v1/models 里 server 上报的全部模型
  .venv/bin/python pre-test.py --quick              # 只跑 0/1/2/3/4/6/7（快，约 4 次请求）
  .venv/bin/python pre-test.py --json report.json   # 落盘报告（默认不写盘）
  .venv/bin/python pre-test.py --verbose            # 打印模型回复摘录

退出码：0 = 无 FAIL；1 = 至少一个 FAIL。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------- 常量（BUILD 层）
PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"

# 判定为「不兼容」的硬门槛探针（缺一个 harness 就会静默降级/卡死）
BLOCKERS = {
    "server.reachable",
    "conn.reuse",
    "chat.plain",
    "chat.json_schema",
    "tools.auto",
    "tools.required_single",
    "tools.roundtrip",
    "judge.schema",
}

# 失败时的定位/修复提示（按探针名）
HINTS = {
    "server.reachable": "后端没起来。LM Studio 默认 http://localhost:1234/v1；用 --base-url 指定别的端点。",
    "conn.reuse": "【后端 bug，非模型问题】同一 TCP keep-alive 连接上的第二次请求返 404，新连接正常。"
                  "swe_agent 的 llm_lmstudio._get_client() 是全局单例（连接复用）→ 第一轮之后全部请求 404。"
                  "修法（择一）：① 每次请求新建 OpenAI client；② 传 http_client=httpx.Client(headers={'Connection':'close'})"
                  " 关掉 keep-alive；③ 换后端。pre-test 已自动切到「每探针新建 client」以测出模型真实能力。",
    "catalog.registered": "模型 id 没登记进 swe_agent/models.py 的 MODELS 目录 → role_spec 抛 ModelUnknown，"
                          "judge 直接返回 notsure。加一条 MODELS 条目（provider/fc/max_tokens）。",
    "model.listed": "模型不在 /v1/models 列表里 = 后端没加载它。LM Studio 需先加载；"
                    "SWE_MODEL_LOAD_UNLOAD=1 时 harness 会自行 load，否则必须人工常驻。",
    "chat.plain": "连纯文本补全都失败 → compact/摘要整条链路不可用，先查后端与模型加载状态。",
    "chat.json_schema": "planner 走 response_format=json_schema(strict)。两条死法：①后端不支持（换后端 / 改 "
                        "roles._planner_response_format）；②支持但生成太慢、超出 LMSTUDIO_TIMEOUT（调大 "
                        "LMSTUDIO_TIMEOUT、或调小 LMSTUDIO_MAX_TOKENS、或精简 _PLANNER_JSON_SCHEMA）。"
                        "判死后 planner 静默降级为本地自规划（契约质量骤降）。",
    "chat.json_object": "GLM 走 json_object；本地 LM Studio 实测拒收（400）属正常，harness 已按 provider 分流。",
    "tools.auto": "executor 主链路。模型不吐 tool_calls → 退化成文本解析，弱模型基本跑不完任务。",
    "tools.required_single": "analyzer 收口靠「裁剪单工具 + tool_choice=required」强制交卷。"
                             "后端不支持 required → analyzer 永远不交 finish_analysis，只能靠 fallback/超时旁路。",
    "tools.roundtrip": "第二轮回灌（assistant(tool_calls)+tool 结果）被拒 → 多轮工具循环直接断，"
                       "executor/tester 只能跑一轮。",
    "tools.parallel": "一轮只回 1 个 tool_call。harness 支持一轮多调用（PARALLEL_TOOL_CALLS），"
                      "但此模型/后端不并发 → 步数变多、预算吃紧。",
    "judge.schema": "judge 路径（system=\"\" + JUDGE_SCHEMA + temperature=0）。失败 → 全部 judge 判 notsure，"
                    "analyzer 的 no-tool 分支与 tester 验收失去裁判。",
    "extra.thinking_off": "extra_body.chat_template_kwargs 不被接受。思考类模型（Ling）会吃掉 token 预算 → 空输出。",
    "param.max_tokens": "max_tokens 超过后端窗口 → 400。调小 LMSTUDIO_MAX_TOKENS。",
    "resp.reasoning_content": "响应带 reasoning_content（output-only）。harness 出向已剥除；"
                              "若自行拼 messages 回传该字段会 400。",
    "compat.tool_choice_function": "OpenAI 风格 function 对象 tool_choice 被拒（LM Studio 已知）。"
                                   "harness 已用「裁剪单工具 + required」规避，此处仅确认规避是否必要。",
}

# ---------------------------------------------------------------- 轻量输出
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") != "1"
_C = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m",
      "INFO": "\033[36m", "SKIP": "\033[90m", "R": "\033[0m", "B": "\033[1m"}


def _c(s: str, k: str) -> str:
    return f"{_C[k]}{s}{_C['R']}" if _USE_COLOR else s


def _clip(s: Any, n: int = 120) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------- harness 契约加载
@dataclass
class Harness:
    """swe_agent 的真实契约（tools/schema/配置）。导入失败时退化为内置最小定义。"""
    ok: bool = False
    err: str = ""
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    max_tokens: int = 8192
    lm_timeout: float = 240.0        # llm_lmstudio.LMSTUDIO_TIMEOUT（单次 HTTP 硬超时）
    parallel: bool = True
    role_models: Dict[str, str] = field(default_factory=dict)
    models_catalog: Dict[str, dict] = field(default_factory=dict)
    tools: Dict[str, List[dict]] = field(default_factory=dict)
    planner_schema: dict = field(default_factory=dict)
    judge_schema: dict = field(default_factory=dict)
    judge_prompt: str = "判断下面这段内容是否算有效产出。只输出 JSON。\n\n内容：\n{content}"
    stop_tool: Dict[str, str] = field(default_factory=dict)


def load_harness(base_url: Optional[str] = None, api_key: Optional[str] = None) -> Harness:
    """尽量复用 swe_agent 的真实构件（与 e2e 完全同一份契约）；导入失败则回落内置定义。"""
    h = Harness()
    try:
        import logging
        _lg = logging.getLogger("harness")
        _old = _lg.level
        _lg.setLevel(logging.ERROR)          # 静音 registry 的「工具未注册」INFO
        try:
            from swe_agent import config as C
            from swe_agent import models as M
            import swe_agent.tools  # noqa: F401  触发工具注册（否则 glm_tools 返回空集）
            from swe_agent.registry import ToolRegistry
            from swe_agent.roles import _PLANNER_JSON_SCHEMA
        finally:
            _lg.setLevel(_old)

        h.base_url = base_url or C.LMSTUDIO_BASE_URL
        h.api_key = api_key or C.LMSTUDIO_API_KEY
        h.max_tokens = int(C.LMSTUDIO_MAX_TOKENS)
        try:
            from swe_agent.llm_lmstudio import LMSTUDIO_TIMEOUT
            h.lm_timeout = float(LMSTUDIO_TIMEOUT)
        except Exception:
            pass
        h.parallel = bool(getattr(C, "PARALLEL_TOOL_CALLS", True))
        h.models_catalog = {k: {kk: vv for kk, vv in v.items() if kk != "fc"}
                            for k, v in M.MODELS.items()}
        for role in ("planner", "analyzer", "executor"):
            mid = M.role_model_id(role)
            if mid:
                h.role_models[role] = mid
        for role, env in (("tester", "TESTER_MODEL"), ("judge", "JUDGE_MODEL"),
                          ("compact", "SIDECAR_COMPRESS_MODEL")):
            mid = getattr(C, env, "") or ""
            if mid:
                h.role_models[role] = mid
        for role in ("executor", "analyzer", "tester"):
            try:
                h.tools[role] = ToolRegistry.glm_tools(role)
            except Exception:
                h.tools[role] = []
        h.planner_schema = _PLANNER_JSON_SCHEMA
        h.judge_schema = M.JUDGE_SCHEMA
        tmpl = (C.JUDGE_PROMPTS or {}).get("analyzer")
        if tmpl:
            h.judge_prompt = tmpl
        h.stop_tool = {"analyzer": "finish_analysis", "tester": "finish_verify", "executor": "complete"}
        h.ok = True
    except Exception as e:
        h.err = f"{type(e).__name__}: {e}"
        # —— 内置最小契约（保证脚本在 swe_agent 不可用/被改坏时仍能独立跑）——
        h.base_url = base_url or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        h.api_key = api_key or os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
        h.max_tokens = int(os.environ.get("LMSTUDIO_MAX_TOKENS", "8192"))
        h.tools = {
            "executor": [_t("shell", "在 shell 里执行一条命令", {"command": "要执行的命令"})],
            "analyzer": [_t("finish_analysis", "结束只读分析并提交摘要", {"summary": "发现摘要"})],
            "tester": [_t("finish_verify", "提交验收结论", {"result": "pass/fail"}, ["result"])],
        }
        h.planner_schema = {
            "type": "json_schema",
            "json_schema": {"name": "execution_plan", "strict": "true", "schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"},
                               "tasks": {"type": "array", "items": {"type": "string"}}},
                "required": ["summary", "tasks"], "additionalProperties": False}},
        }
        h.judge_schema = {
            "type": "json_schema",
            "json_schema": {"name": "judge_result", "strict": "true", "schema": {
                "type": "object",
                "properties": {"result": {"type": "string", "enum": ["yes", "no", "notsure"]},
                               "reason": {"type": "string"}},
                "required": ["result", "reason"], "additionalProperties": False}},
        }
        h.stop_tool = {"analyzer": "finish_analysis", "tester": "finish_verify", "executor": "shell"}
    if base_url:
        h.base_url = base_url
    if api_key:
        h.api_key = api_key
    return h


def _t(name: str, desc: str, props: Dict[str, str], req: Optional[List[str]] = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object",
                       "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
                       "required": req or list(props)}}}


# ---------------------------------------------------------------- 探针基础设施
@dataclass
class Result:
    name: str
    status: str
    latency: float = 0.0
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


def _http_status(e: Exception) -> Optional[int]:
    code = getattr(e, "status_code", None)
    if code:
        return int(code)
    try:                                   # openai.APIStatusError 带 response
        return int(e.response.status_code)  # type: ignore[union-attr]
    except Exception:
        return None


def _err_text(e: Exception, n: int = 160) -> str:
    body = ""
    try:
        body = e.response.text[:n]  # type: ignore[union-attr]
    except Exception:
        pass
    return f"{type(e).__name__}: {_clip(str(e), n)}" + (f" | body={_clip(body, n)}" if body else "")


class Runner:
    def __init__(self, h: Harness, timeout: float, verbose: bool = False):
        self.h = h
        self.timeout = timeout
        self.verbose = verbose
        from openai import OpenAI
        self.OpenAI = OpenAI
        self.client = self.new_client()
        self.seen_reasoning = False
        # 后端 keep-alive 复用 bug 的规避开关：True 时每次请求新建 client（新 TCP 连接）。
        # 由 probe_conn_reuse 实测后置位 —— 否则第一轮之后的探针全是假 FAIL。
        self.reuse_broken = False

    def new_client(self):
        return self.OpenAI(base_url=self.h.base_url, api_key=self.h.api_key,
                           timeout=self.timeout, max_retries=0)

    # —— 单次调用（与 llm_lmstudio 同构：tools / response_format 互斥，extra_body 透传）——
    def call(self, messages: List[dict], model: str, *,
             temperature: float = 0.3, max_tokens: Optional[int] = None,
             tools: Optional[List[dict]] = None, tool_choice: Optional[Any] = None,
             response_format: Optional[dict] = None,
             parallel_tool_calls: Optional[bool] = None,
             extra_body: Optional[dict] = None):
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": int(max_tokens or min(2048, self.h.max_tokens)),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"
            payload["parallel_tool_calls"] = bool(
                parallel_tool_calls if parallel_tool_calls is not None else self.h.parallel)
        elif response_format is not None:
            payload["response_format"] = response_format
        client = self.new_client() if self.reuse_broken else self.client
        return client.chat.completions.create(**payload, extra_body=extra_body or {})

    @staticmethod
    def unpack(resp) -> Tuple[str, List[dict], Any, Any]:
        """→ (content, tool_calls[{id,name,arguments}], finish_reason, reasoning)"""
        try:
            msg = resp.choices[0].message
        except Exception:
            return "", [], None, None
        tcs = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or ""}
               for tc in (getattr(msg, "tool_calls", None) or [])]
        return (msg.content or "").strip(), tcs, resp.choices[0].finish_reason, getattr(msg, "reasoning_content", None)

    def note_reasoning(self, reasoning) -> None:
        if reasoning:
            self.seen_reasoning = True


# ---------------------------------------------------------------- 探针
def probe_server(r: Runner, mid: str, ctx: dict) -> Result:
    t0 = time.time()
    try:
        models = r.client.models.list()
        ids = [getattr(m, "id", "") for m in getattr(models, "data", []) or []]
        ctx["server_models"] = ids
        return Result("server.reachable", PASS, time.time() - t0,
                      f"{len(ids)} 个模型在线", {"ids": ids[:20]})
    except Exception as e:
        return Result("server.reachable", FAIL, time.time() - t0,
                      _err_text(e), {"error": str(e)})


def probe_conn_reuse(r: Runner, mid: str, ctx: dict) -> Result:
    """连接复用探针：同一 client（keep-alive）连发两次最小请求。

    某些后端（实测 mtplx server）在同一 TCP 连接上的第二次请求返 404，而新连接正常 ——
    swe_agent 的 llm_lmstudio._get_client() 是全局单例，于是「第一轮之后全部请求 404」。
    这是【后端 bug】而非模型能力问题；检出后 Runner 自动改为每探针新建 client，
    免得后续 13 项全变成假 FAIL。
    """
    r.client = r.new_client()                       # 从干净连接开始（models.list 可能已污染）
    msgs = [{"role": "user", "content": "回复 ok。"}]
    codes: List[int] = []
    err2 = ""
    t0 = time.time()
    for i in range(2):
        try:
            r.client.chat.completions.create(model=mid, messages=msgs, temperature=0.0, max_tokens=16)
            codes.append(200)
        except Exception as e:
            codes.append(_http_status(e) or 0)
            if i == 1:
                err2 = _err_text(e, 120)
    dt = time.time() - t0
    if codes[0] == 200 and codes[1] != 200:
        r.reuse_broken = True
        return Result("conn.reuse", FAIL, dt,
                      f"第1轮 200 / 第2轮 {codes[1]}（keep-alive 复用后 {err2}）；"
                      f"后续探针自动改为每轮新建连接", {"codes": codes})
    if codes[0] != 200:
        return Result("conn.reuse", WARN, dt,
                      f"首轮即 {codes[0]}（{err2 or '详见 chat.plain'}）", {"codes": codes})
    return Result("conn.reuse", PASS, dt, f"连续 2 次请求均 200（连接复用正常）", {"codes": codes})


def probe_catalog(r: Runner, mid: str, ctx: dict) -> Result:
    cat = r.h.models_catalog
    if mid in cat:
        spec = cat[mid]
        return Result("catalog.registered", PASS, 0.0,
                      f"provider={spec.get('provider')} max_tokens={spec.get('max_tokens')}", spec)
    return Result("catalog.registered", WARN, 0.0,
                  f"{mid!r} 不在 MODELS 目录（已知：{', '.join(list(cat)[:8])}…）", {})


def probe_listed(r: Runner, mid: str, ctx: dict) -> Result:
    ids = ctx.get("server_models")
    if ids is None:                                  # server 不可达 → 跳过
        return Result("model.listed", SKIP, 0.0, "server 未连通，无法核对", {})
    if mid in ids:
        return Result("model.listed", PASS, 0.0, "后端已加载该模型", {})
    low = mid.lower()
    near = [i for i in ids if low in i.lower() or i.lower() in low]
    return Result("model.listed", WARN, 0.0,
                  f"{mid!r} 不在 /v1/models 列表中" + (f"（近似：{near[:3]}）" if near else ""),
                  {"server_ids": ids[:20]})


def probe_plain(r: Runner, mid: str, ctx: dict) -> Result:
    msgs = [{"role": "system", "content": "你是压缩助手，把内容压成一句话。"},
            {"role": "user", "content": "请把下面内容压缩成一句话：斐波那契数列 F(0)=0, F(1)=1, "
                                        "F(n)=F(n-1)+F(n-2)，是最经典的递归示例。"}]
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, max_tokens=512)
    except Exception as e:
        return Result("chat.plain", FAIL, time.time() - t0, _err_text(e), {})
    content, tcs, fr, rs = r.unpack(resp)
    r.note_reasoning(rs)
    dt = time.time() - t0
    if content:
        return Result("chat.plain", PASS, dt, f"{len(content)} 字符",
                      {"finish_reason": fr, "content": _clip(content, 200)})
    return Result("chat.plain", FAIL, dt, "软空响应（200 但无 content / 无 tool_call）",
                  {"finish_reason": fr})


def _strip_fence(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[1] if len(parts) > 1 else s
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        s = s[i:j + 1]
    return s


def probe_json_schema(r: Runner, mid: str, ctx: dict) -> Result:
    """planner 路径：真实 _PLANNER_JSON_SCHEMA（strict=true）。

    探针用小 max_tokens（PROBE_TOKENS）快速取样本，再按 harness 真实预算
    （LMSTUDIO_MAX_TOKENS）线性外推总耗时 —— 外推超过 LMSTUDIO_TIMEOUT 即判 FAIL：
    真实场景里 planner 会直接被 240s 硬超时掐断，静默降级为本地自规划。
    """
    PROBE_TOKENS = 2048
    msgs = [{"role": "system", "content": "你是规划助手，只输出契约 JSON。"},
            {"role": "user", "content": "【开发任务】用 Python 实现斐波那契数列函数并写测试。"}]
    mt = min(PROBE_TOKENS, r.h.max_tokens)
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, max_tokens=mt, response_format=r.h.planner_schema)
    except Exception as e:
        st = FAIL
        det = _err_text(e)
        if "timed out" in str(e).lower():
            det = f"超时（{r.timeout:.0f}s 内未产出完整契约）"
        return Result("chat.json_schema", st, time.time() - t0, det, {"http": _http_status(e)})
    content, tcs, fr, rs = r.unpack(resp)
    r.note_reasoning(rs)
    dt = time.time() - t0
    if not content:
        return Result("chat.json_schema", FAIL, dt, "软空响应（无 content）", {"finish_reason": fr})
    try:
        obj = json.loads(_strip_fence(content))
        keys = sorted(obj.keys()) if isinstance(obj, dict) else []
        return Result("chat.json_schema", PASS, dt, f"JSON OK，顶层键={keys}",
                      {"keys": keys, "content": _clip(content, 160)})
    except Exception:
        # 按 harness 真实 token 预算外推；超 LMSTUDIO_TIMEOUT 视为不可达（而非「不支持」）
        est = dt * (r.h.max_tokens / max(mt, 1))
        over = est > r.h.lm_timeout
        note = (f"{mt}token 用 {dt:.1f}s → 外推 {r.h.max_tokens}token 需 ~{est:.0f}s"
                f"（LMSTUDIO_TIMEOUT={r.h.lm_timeout:.0f}s，{'超出' if over else '未超出'}）")
        if over:
            return Result("chat.json_schema", FAIL, dt,
                          f"生成过慢：{note} → planner 必被硬超时掐断，静默降级本地自规划",
                          {"finish_reason": fr, "est_seconds": round(est, 1),
                           "content": _clip(content, 160)})
        st = WARN if (fr == "length" or not content.rstrip().endswith(("}", "]"))) else FAIL
        return Result("chat.json_schema", st, dt,
                      f"无法解析为 JSON（finish_reason={fr}）；{note}",
                      {"content": _clip(content, 200)})


def probe_json_object(r: Runner, mid: str, ctx: dict) -> Result:
    """GLM 降级形态。本地 LM Studio 拒收 400 属已知/已规避 → 记 INFO，不判死。"""
    msgs = [{"role": "user", "content": "输出 JSON：{\"summary\":\"ok\",\"tasks\":[\"a\"]}"}]
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, max_tokens=512, response_format={"type": "json_object"})
    except Exception as e:
        return Result("chat.json_object", INFO, time.time() - t0,
                      f"拒收（本地后端已知行为）：{_err_text(e, 110)}", {"http": _http_status(e)})
    content, _, fr, _ = r.unpack(resp)
    try:
        json.loads(_strip_fence(content))
        return Result("chat.json_object", PASS, time.time() - t0, "json_object 可用", {})
    except Exception:
        return Result("chat.json_object", WARN, time.time() - t0,
                      "接受 json_object 但输出不是 JSON", {"content": _clip(content, 120)})


def probe_tools_auto(r: Runner, mid: str, ctx: dict) -> Result:
    """executor 一轮：工具全集 + tool_choice=auto。"""
    tools = r.h.tools.get("executor") or r.h.tools.get("analyzer") or []
    if not tools:
        return Result("tools.auto", FAIL, 0.0, "harness 未提供工具 schema（import 失败？）", {})
    msgs = [{"role": "system", "content": "你是执行助手，用工具推进任务。"},
            {"role": "user", "content": "请调用 shell 工具执行 `echo pretest-ok`。"}]
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, max_tokens=min(1024, r.h.max_tokens), tools=tools, tool_choice="auto")
    except Exception as e:
        return Result("tools.auto", FAIL, time.time() - t0, _err_text(e), {"http": _http_status(e)})
    content, tcs, fr, rs = r.unpack(resp)
    r.note_reasoning(rs)
    dt = time.time() - t0
    names = [t["name"] for t in tcs]
    valid = [n for n in names if n in {x["function"]["name"] for x in tools}]
    if valid:
        st = PASS if len(valid) == len(names) else WARN
        extra = "" if st == PASS else f"（幻觉工具名：{[n for n in names if n not in valid]}）"
        return Result("tools.auto", st, dt, f"tool_calls={names}{extra}",
                      {"tool_calls": names, "finish_reason": fr})
    if content:
        return Result("tools.auto", WARN, dt,
                      "只吐文本、没调工具（executor 会退化成文本解析）", {"content": _clip(content, 160)})
    return Result("tools.auto", FAIL, dt, "软空响应（无 tool_call 无 content）", {"finish_reason": fr})


def probe_required_single(r: Runner, mid: str, ctx: dict) -> Result:
    """analyzer 收口：裁剪到单工具 + tool_choice="required"（harness 的真强制手法）。"""
    atools = r.h.tools.get("analyzer") or []
    stop = r.h.stop_tool.get("analyzer", "finish_analysis")
    single = [t for t in atools if t["function"]["name"] == stop]
    if not single:
        single = [_t(stop, "结束只读分析并提交摘要", {"summary": "发现摘要"})]
    msgs = [{"role": "system", "content": "你是需求分析师，用工具探查后提交摘要。"},
            {"role": "user", "content": "任务：用 Python 实现斐波那契。请直接调用 finish_analysis 提交摘要，"
                                        "summary 里写「F(0)=0,F(1)=1,F(n)=F(n-1)+F(n-2)」。"}]
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, max_tokens=min(1024, r.h.max_tokens),
                      tools=single, tool_choice="required")
    except Exception as e:
        return Result("tools.required_single", FAIL, time.time() - t0, _err_text(e),
                      {"http": _http_status(e)})
    content, tcs, fr, rs = r.unpack(resp)
    r.note_reasoning(rs)
    dt = time.time() - t0
    names = [t["name"] for t in tcs]
    if names:
        ok = stop in names
        return Result("tools.required_single", PASS if ok else WARN, dt,
                      f"tool_calls={names}" + ("" if ok else f"（期望 {stop}）"),
                      {"tool_calls": names, "args": _clip(tcs[0]["arguments"], 160)})
    if content:
        return Result("tools.required_single", WARN, dt,
                      "required 下仍未调工具、只吐文本", {"content": _clip(content, 160)})
    return Result("tools.required_single", FAIL, dt, "软空响应", {"finish_reason": fr})


def probe_roundtrip(r: Runner, mid: str, ctx: dict) -> Result:
    """第二轮回灌：assistant(tool_calls) + tool(tool_call_id) 交替，验证不 400。"""
    tname = "shell"
    etools = r.h.tools.get("executor") or []
    if not any(t["function"]["name"] == tname for t in etools):
        etools = etools + [_t(tname, "在 shell 里执行一条命令", {"command": "要执行的命令"})]
    msgs = [
        {"role": "user", "content": "执行 `echo hi`。"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_pretest_1", "type": "function",
                         "function": {"name": tname, "arguments": json.dumps({"command": "echo hi"})}}]},
        {"role": "tool", "tool_call_id": "call_pretest_1", "content": "hi"},
        {"role": "user", "content": "命令输出是什么？一句话回答。"},
    ]
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, max_tokens=512, tools=etools, tool_choice="auto")
    except Exception as e:
        return Result("tools.roundtrip", FAIL, time.time() - t0, _err_text(e),
                      {"http": _http_status(e)})
    content, tcs, fr, rs = r.unpack(resp)
    r.note_reasoning(rs)
    dt = time.time() - t0
    if content or tcs:
        return Result("tools.roundtrip", PASS, dt,
                      f"第二轮 OK（{('tool_calls=' + str([t['name'] for t in tcs])) if tcs else str(len(content)) + ' 字符'}）",
                      {"content": _clip(content, 120)})
    return Result("tools.roundtrip", WARN, dt, "第二轮软空（不 400，但无产出）", {"finish_reason": fr})


def probe_parallel(r: Runner, mid: str, ctx: dict) -> Result:
    etools = r.h.tools.get("executor") or []
    if len(etools) < 2:
        return Result("tools.parallel", SKIP, 0.0, "工具数 < 2，无法测并发调用", {})
    a, b = etools[0]["function"]["name"], etools[1]["function"]["name"]
    msgs = [{"role": "user", "content": f"请【同时】调用 {a} 与 {b} 两个工具（同一轮，不要分开）。"}]
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, max_tokens=min(1024, r.h.max_tokens), tools=etools,
                      tool_choice="auto", parallel_tool_calls=True)
    except Exception as e:
        return Result("tools.parallel", WARN, time.time() - t0, _err_text(e, 120),
                      {"http": _http_status(e)})
    content, tcs, fr, rs = r.unpack(resp)
    r.note_reasoning(rs)
    dt = time.time() - t0
    n = len(tcs)
    if n >= 2:
        return Result("tools.parallel", PASS, dt, f"一轮返回 {n} 个 tool_call", {"names": [t["name"] for t in tcs]})
    if n == 1:
        return Result("tools.parallel", WARN, dt, "一轮只返回 1 个 tool_call（harness 支持多，后端未并发）", {})
    return Result("tools.parallel", WARN, dt, "未返回 tool_call（无法判断并发能力）",
                  {"content": _clip(content, 120)})


def probe_judge(r: Runner, mid: str, ctx: dict) -> Result:
    """judge 路径：system="" + JUDGE_SCHEMA + temperature=0（与 models.judge 一致）。"""
    msgs = [{"role": "system", "content": ""},
            {"role": "user", "content": r.h.judge_prompt.format(
                content="斐波那契数列：F(0)=0，F(1)=1，F(n)=F(n-1)+F(n-2)。边界：n 为负应报错。")}]
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, temperature=0.0, max_tokens=min(512, r.h.max_tokens),
                      response_format=r.h.judge_schema)
    except Exception as e:
        return Result("judge.schema", FAIL, time.time() - t0, _err_text(e), {"http": _http_status(e)})
    content, _, fr, rs = r.unpack(resp)
    r.note_reasoning(rs)
    dt = time.time() - t0
    if not content:
        return Result("judge.schema", FAIL, dt, "软空响应（无 content）", {"finish_reason": fr})
    try:
        val = json.loads(_strip_fence(content))
        res = str(val.get("result", "")).strip().lower()
        ok = res in ("yes", "no", "notsure")
        return Result("judge.schema", PASS if ok else WARN, dt,
                      f"result={res!r}" + ("" if ok else "（非法取值）"), {"raw": _clip(content, 160)})
    except Exception:
        return Result("judge.schema", WARN, dt, "judge 输出无法解析（models.judge 会按 notsure 处理）",
                      {"content": _clip(content, 160)})


def probe_thinking_off(r: Runner, mid: str, ctx: dict) -> Result:
    etools = r.h.tools.get("executor") or []
    msgs = [{"role": "user", "content": "调用 shell 执行 echo pretest-thinking。"}]
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, max_tokens=min(512, r.h.max_tokens),
                      tools=etools or None, tool_choice="auto" if etools else None,
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    except Exception as e:
        return Result("extra.thinking_off", WARN, time.time() - t0,
                      f"extra_body 被拒（思考类模型会吃掉 token 预算）：{_err_text(e, 110)}",
                      {"http": _http_status(e)})
    content, tcs, fr, rs = r.unpack(resp)
    dt = time.time() - t0
    if content or tcs:
        return Result("extra.thinking_off", PASS, dt, "接受 chat_template_kwargs 且正常产出", {})
    return Result("extra.thinking_off", WARN, dt, "接受参数但产出为空", {"finish_reason": fr})


def probe_max_tokens(r: Runner, mid: str, ctx: dict) -> Result:
    msgs = [{"role": "user", "content": "回复 ok。"}]
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, max_tokens=r.h.max_tokens)
    except Exception as e:
        return Result("param.max_tokens", FAIL, time.time() - t0,
                      f"max_tokens={r.h.max_tokens} 被拒：{_err_text(e, 110)}", {"http": _http_status(e)})
    content, _, fr, _ = r.unpack(resp)
    return Result("param.max_tokens", PASS, time.time() - t0,
                  f"max_tokens={r.h.max_tokens} 被接受", {"content": _clip(content, 60)})


def probe_reasoning(r: Runner, mid: str, ctx: dict) -> Result:
    if r.seen_reasoning:
        return Result("resp.reasoning_content", INFO, 0.0,
                      "响应含 reasoning_content（harness 出向经 prepare_messages 剥除）", {})
    return Result("resp.reasoning_content", PASS, 0.0, "未见 reasoning_content 字段", {})


def probe_choice_function(r: Runner, mid: str, ctx: dict) -> Result:
    """OpenAI 风格 function 对象 tool_choice：harness 已裁剪规避，这里确认「规避是否必要」。"""
    tools = r.h.tools.get("analyzer") or []
    if not tools:
        return Result("compat.tool_choice_function", SKIP, 0.0, "无工具 schema", {})
    msgs = [{"role": "user", "content": "提交分析摘要。"}]
    t0 = time.time()
    try:
        resp = r.call(msgs, mid, max_tokens=256, tools=tools,
                      tool_choice={"type": "function", "function": {"name": tools[0]["function"]["name"]}})
    except Exception as e:
        return Result("compat.tool_choice_function", INFO, time.time() - t0,
                      f"不支持 function 对象（harness 已用裁剪+required 规避）：{_err_text(e, 100)}",
                      {"http": _http_status(e)})
    _, tcs, _, _ = r.unpack(resp)
    return Result("compat.tool_choice_function", PASS, time.time() - t0,
                  f"支持（tool_calls={[t['name'] for t in tcs]}）", {})


# --quick 只跑这 7 项（探针名 → 函数）
QUICK = {probe_server, probe_conn_reuse, probe_catalog, probe_listed, probe_plain,
         probe_json_schema, probe_tools_auto, probe_required_single}

PROBES: List[Callable[[Runner, str, dict], Result]] = [
    probe_server, probe_conn_reuse, probe_catalog, probe_listed,
    probe_plain, probe_json_schema, probe_json_object,
    probe_tools_auto, probe_required_single, probe_roundtrip, probe_parallel,
    probe_judge, probe_thinking_off, probe_max_tokens,
    probe_reasoning, probe_choice_function,
]


# ---------------------------------------------------------------- 主流程
def run_model(h: Harness, mid: str, args) -> Dict[str, Any]:
    r = Runner(h, args.timeout, args.verbose)
    ctx: Dict[str, Any] = {}
    print(f"\n{_c('═══', 'B')} {_c(mid, 'B')}  @ {h.base_url}")
    results: List[Result] = []
    for p in PROBES:
        if args.quick and p not in QUICK:
            continue
        try:
            res = p(r, mid, ctx)
        except Exception as e:                       # 探针自身崩溃不算模型不兼容
            res = Result(p.__name__, WARN, 0.0, f"探针异常：{_clip(traceback.format_exc(), 200)}", {})
        results.append(res)
        tag = _c(f"[{res.status:4}]", res.status)
        lat = f"{res.latency:6.2f}s" if res.latency else "       "
        print(f"  {tag} {lat}  {res.name:<28} {res.detail}")
        if args.verbose and res.evidence:
            for k, v in list(res.evidence.items())[:4]:
                print(f"        └ {k}: {_clip(v, 160)}")

    n_fail = sum(1 for x in results if x.status == FAIL)
    n_warn = sum(1 for x in results if x.status == WARN)
    blocker_fail = [x.name for x in results if x.status == FAIL and x.name in BLOCKERS]
    if blocker_fail:
        verdict = "INCOMPATIBLE"
    elif n_fail:
        verdict = "DEGRADED"
    elif n_warn:
        verdict = "DEGRADED"
    else:
        verdict = "READY"
    color = {"READY": "PASS", "DEGRADED": "WARN", "INCOMPATIBLE": "FAIL"}[verdict]
    print(f"  {_c('── verdict: ' + verdict, color)} "
          f"(fail={n_fail}, warn={n_warn}, 硬门槛失败={blocker_fail or '无'})")
    for x in results:
        if x.status in (FAIL, WARN) and x.name in HINTS:
            print(f"     {_c('→', 'WARN')} {x.name}: {HINTS[x.name]}")
    return {"model": mid, "verdict": verdict, "fail": n_fail, "warn": n_warn,
            "blocker_fail": blocker_fail, "base_url": h.base_url,
            "results": [{"name": x.name, "status": x.status, "latency": round(x.latency, 3),
                         "detail": x.detail, "evidence": x.evidence} for x in results]}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="swe_agent × 模型 兼容性预检（跑 e2e 之前先跑它）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[],
                    help="指定模型 id（可重复）；默认测 harness 当前角色模型")
    ap.add_argument("--auto", action="store_true", help="测 /v1/models 里 server 上报的全部模型")
    ap.add_argument("--base-url", default=None, help="覆盖后端（默认 LMSTUDIO_BASE_URL）")
    ap.add_argument("--api-key", default=None, help="覆盖 api key")
    ap.add_argument("--timeout", type=float, default=60.0, help="单次请求超时秒数（默认 60）")
    ap.add_argument("--quick", action="store_true", help="只跑核心 7 项")
    ap.add_argument("--json", dest="json_path", default=None, help="报告落盘路径（默认不写盘）")
    ap.add_argument("--verbose", action="store_true", help="打印模型回复摘录")
    args = ap.parse_args()

    h = load_harness(args.base_url, args.api_key)
    print(_c("swe_agent 兼容性预检 pre-test", "B"))
    print(f"  后端        : {h.base_url}")
    print(f"  契约来源    : {'swe_agent（真实 schema / 工具）' if h.ok else '内置 fallback（导入失败：%s）' % h.err}")
    print(f"  角色→模型   : {h.role_models or '(未解析到)'}")
    print(f"  工具集      : {{" + ", ".join(f'{k}: {len(v)}' for k, v in h.tools.items()) + "}")
    print(f"  max_tokens  : {h.max_tokens}   parallel_tool_calls={h.parallel}   timeout={args.timeout}s")

    mids: List[str] = list(args.model)
    if args.auto or not mids:
        try:
            from openai import OpenAI
            ids = [getattr(m, "id", "") for m in
                   (OpenAI(base_url=h.base_url, api_key=h.api_key, timeout=min(args.timeout, 20),
                           max_retries=0).models.list().data or [])]
            ids = [i for i in ids if i]
        except Exception as e:
            ids = []
            if args.auto:
                print(_c(f"  /v1/models 拉取失败：{_err_text(e)}", "FAIL"))
        if args.auto:
            mids = mids + [i for i in ids if i not in mids]
        elif not mids:
            # 默认：harness 角色模型优先；角色模型不在 server 列表时改用 server 实际加载的模型
            role_ids = list(dict.fromkeys(h.role_models.values()))
            mids = role_ids or ids
    if not mids:
        print(_c("  没有可测的模型：用 --model <id> 指定，或先启动后端。", "FAIL"))
        return 1

    reports = [run_model(h, m, args) for m in mids]

    print(f"\n{_c('═══ 汇总', 'B')}")
    for rep in reports:
        color = {"READY": "PASS", "DEGRADED": "WARN", "INCOMPATIBLE": "FAIL"}[rep["verdict"]]
        print(f"  {_c(rep['verdict'], color):<24} {rep['model']:<42} "
              f"fail={rep['fail']} warn={rep['warn']}")

    if args.json_path:
        try:
            with open(args.json_path, "w", encoding="utf-8") as f:
                json.dump({"ts": time.time(), "base_url": h.base_url, "reports": reports},
                          f, ensure_ascii=False, indent=2)
            print(f"\n报告已写入：{args.json_path}")
        except Exception as e:
            print(_c(f"报告写入失败：{e}", "FAIL"))

    return 1 if any(r["blocker_fail"] or r["fail"] for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
