#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/llm_lmstudio.py —— LM Studio 本地 Executor（OpenAI 兼容协议）

LM Studio 在 http://localhost:1234/v1 暴露 OpenAI 兼容接口。本模块用官方
OpenAI SDK（不手搓 requests）调用它，走原生 function calling：

- 把自描述工具 schema（ToolRegistry.glm_tools()）作为 tools 传入；
- 强制 tool_choice=agent_action，让 1.2B 小模型稳定产出结构化动作，
  彻底规避其手写 JSON 时常见的三引号/未转义换行/缺键等畸形；
- 解析返回的 tool_calls.arguments 为动作 dict，转成 JSON 字符串交给
  调度层 parse_action 正常执行（与 GLM executor 路径一致）。
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import (
    LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, LMSTUDIO_MODEL,
    LMSTUDIO_MAX_TOKENS, STATS,
)
from . import config as C  # 运行期读取开关（如 MODEL_LOAD_UNLOAD），不能用导入期快照
from swe_agent.log import logger

# 单次 HTTP 调用硬超时（秒）：本地 7B 长生成可能缓慢甚至死循环，SDK 默认 600s
# 会让整个 loop 假死 10 分钟。超时→异常→返回空串→调用方重试/提示，loop 保持活性。
LMSTUDIO_TIMEOUT = float(os.environ.get("LMSTUDIO_TIMEOUT", "240"))

# 请求落盘开关（默认关，零副作用）：LMSTUDIO_DUMP=1 时把每次 LM Studio 请求的
# 完整 messages + tools + max_tokens 追加写到 logs/lmstudio_requests.jsonl（请求+回复同一文件，
# 不加第二个变量），并在调用失败/超时（error 标记）时一并记录——事后可逐字还原「导致 None 的输入」
# 去对照 server 日志。不改变任何运行路径，仅追加写盘。
LMSTUDIO_DUMP = os.environ.get("LMSTUDIO_DUMP", "0") == "1"
_DUMP_PATH = os.environ.get("LMSTUDIO_DUMP_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "lmstudio_requests.jsonl"
)
os.makedirs(os.path.dirname(_DUMP_PATH), exist_ok=True)


def _dump_request(tag: str, payload: dict, extra: Optional[dict] = None) -> None:
    if not LMSTUDIO_DUMP:
        return
    try:
        msgs = payload.get("messages", [])
        rec = {
            "ts": round(time.time(), 3),
            "tag": tag,
            "model": payload.get("model"),
            "max_tokens": payload.get("max_tokens"),
            "n_messages": len(msgs),
            "total_chars": sum(len(str(m.get("content", ""))) for m in msgs),
            "tools": [t.get("function", {}).get("name") for t in payload.get("tools", [])],
            "messages": msgs,
        }
        if extra is not None:
            rec.update(extra)
        with open(_DUMP_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _dump_response(tag: str, payload: dict, resp) -> None:
    """成功返回后把 server 回复追加写到同一个 dump 文件（与 request 按 ts 配对成 request↔response 对）。

    仅只读访问 resp（已解析的 ChatCompletion 对象），不读流、不改对象；整函数 try/except 包裹，
    下游解析（tool_calls / content）完全不受影响。软空响应（200 但无 tool_calls 无 content）也照常落，
    这是诊断「模型拒答/畸形 tool_call」的关键。
    """
    if not LMSTUDIO_DUMP:
        return
    try:
        choice = resp.choices[0]
        msg = choice.message
        tcs = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = tc.function
            tcs.append({"name": fn.name, "arguments": (fn.arguments or "")[:4000]})
        usage = getattr(resp, "usage", None)
        rec = {
            "ts": round(time.time(), 3),
            "tag": tag,
            "model": payload.get("model"),
            "n_messages": len(payload.get("messages", [])),
            "total_chars": sum(len(str(m.get("content", ""))) for m in payload.get("messages", [])),
            "tools": [t.get("function", {}).get("name") for t in payload.get("tools", [])],
            "finish_reason": getattr(choice, "finish_reason", None),
            "tool_calls": tcs,
            "content": (msg.content or "")[:4000],
            "usage": usage and {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            },
        }
        with open(_DUMP_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # max_retries=0：关掉 OpenAI SDK 的默认自动重试（3.1.0 默认 max_retries=2）。
        # 否则 qwen-7b 长生成一旦触发 240s 读超时，SDK 会在「LM Studio 端前一个请求还在
        # 生成」时就偷偷重发第二请求 → 服务端呈现 queue 1（伪并发，与 harness 顺序调用无关）。
        # 重试统一收口到 models._with_retry 的同步退避（MODEL_BACKOFF=2s），不会叠加并发。
        _client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY,
                         timeout=LMSTUDIO_TIMEOUT, max_retries=0)
    return _client


def _reload_lmstudio_model(model: str) -> bool:
    """根因③修复：harness 在角色/任务间 unload 本地模型，下一请求落到 unloaded 窗口
    → LM Studio 返 400 'Model unloaded.'。这里复用 model_swap.load_model 把模型重新
    加载（含异步加载轮询 + 已加载幂等跳过），使后续请求不再 400。

    全局开关 SWE_MODEL_LOAD_UNLOAD=0 时【不做任何补救加载】：本模式下模型由人工常驻，
    harness 一律不碰 load 接口；若模型真不在，就让请求 fail-loud 暴露问题。
    """
    if not C.MODEL_LOAD_UNLOAD:
        logger.info('%s', f'[lmstudio] MODEL_LOAD_UNLOAD=0，跳过自动 reload({model})；请确认该模型已在 LM Studio 常驻加载。')
        return False
    try:
        from . import model_swap
        model_swap.load_model(model)
        return True
    except Exception as e:
        logger.info('%s', f'[lmstudio] reload 失败：{e}')
        return False


def lmstudio_chat_messages(messages: List[Dict[str, str]], model: str = LMSTUDIO_MODEL,
                           temperature: float = 0.3,
                           max_tokens: int = LMSTUDIO_MAX_TOKENS,
                           tools: Optional[List[Dict]] = None,
                           response_format: Optional[Dict[str, Any]] = None,
                           tool_choice: Optional[Any] = None,
                           thinking_off: bool = False,
                           return_meta: bool = False) -> Any:
    """LM Studio 充当 Executor：接收完整 messages，原生 function calling。

    默认返回 action 列表的 JSON 字符串（同 glm_chat_messages 约定），失败返回空串。
    model 由调用方（models.py 按角色解析）传入，默认 LMSTUDIO_MODEL。
    return_meta=True 时返回结构化 dict（供原生 toolcall 协议使用）：
      {"type":"toolcalls","actions":[...],"tool_calls":[{"id","name","arguments"}]}
      {"type":"content","content":"..."}（模型未走工具）
      None（调用失败 / 空响应）
    """
    # 收敛到模型可承受的单轮生成上限（1.2B 上下文通常较小）。
    max_tokens = min(int(max_tokens or LMSTUDIO_MAX_TOKENS), LMSTUDIO_MAX_TOKENS)
    client = _get_client()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # 是否允许模型一轮下发多个 tool_call（由 BUILD 常量收口）。
        # 旧值硬编码 False = 主动要求服务端每轮只给一个工具调用：强模型（Ling 等）一轮
        # 下发多个读操作时，多余调用会被服务端/客户端协议层砍掉——这是「模型支持多工具
        # 但 harness 用不上」的传输层根因。harness 侧已支持一轮全部执行（agent._apply_toolcall）。
        "parallel_tool_calls": bool(C.PARALLEL_TOOL_CALLS),
    }
    # LM Studio 扩展字段经 extra_body 透传（OpenAI SDK 不接受 chat_template_kwargs 作顶层参数）。
    # thinking_off=True 时关闭思考链（如 Ling），避免思考吃光 token 预算导致空输出。
    extra_body: Dict[str, Any] = {}
    if thinking_off:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        payload["tools"] = tools
        # LM Studio 只接受字符串 tool_choice（none/auto/required），不支持 OpenAI 的
        # {"type":"function",...} 强制指定单个函数。调用方传 function 级强制（如 finish_analysis）
        # 时，用「裁剪 tools 到该函数 + required」模拟强制单函数语义——这是 2026-09-03 实测验证的
        # 正确修法：required+单工具真强制模型调该函数（lfm2.5-2.6b 实测 calls=['finish_analysis']），
        # 而原降级 auto 不可靠（ling 曾在 auto+全集下自选 read_file 不交卷，导致 analyzer 旧 fallback 失败）。
        # 注意 qwen2.5.1-coder-7b 在 required 下曾有「简单轮次返 stop+空 tool_calls」怪癖，但
        # 裁剪到单工具后 required 强制调它，实测本地模型正常；此处尊重调用方强制意图，不再静默丢。
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fn_name = tool_choice["function"]["name"]
            single = [t for t in tools if t.get("function", {}).get("name") == fn_name]
            if single:
                payload["tools"] = single
                payload["tool_choice"] = "required"
            else:
                # 调用方指定了不存在的函数名：降级 auto（避免 400，保留原行为）
                payload["tool_choice"] = "auto"
        elif tool_choice is not None and isinstance(tool_choice, str):
            payload["tool_choice"] = tool_choice
        else:
            payload["tool_choice"] = "auto"
    elif response_format is not None:
        # 与 tools 互斥：结构化输出（json_object / json_schema）仅在无工具时下发，
        # 避免 GLM/LM Studio 在「调工具」与「返回 JSON」间摇摆或返回空。
        payload["response_format"] = response_format
    _dump_request("request", payload)
    try:
        resp = client.chat.completions.create(**payload, extra_body=extra_body)
    except Exception as e:
        err = str(e)
        # 根因③：harness 在角色/任务间 unload 本地模型，下一请求落到 unloaded 窗口
        # → LM Studio 返 400（信息形如 "No models loaded" 或旧版 "Model unloaded"）。
        # 重载模型并重试一次，避免退化为 None。recovered=True 时 resp 已被刷新，
        # 跳出 except 走到下方统一解析；否则记日志并返回空/None（上层退避重试/升级模型）。
        recovered = False
        if "No models loaded" in err or "Model unloaded" in err:
            if _reload_lmstudio_model(model):
                try:
                    resp = client.chat.completions.create(**payload, extra_body=extra_body)
                    recovered = True
                except Exception as e2:
                    err = f"reload-ok-retry-failed: {e2}"
            else:
                err = f"reload-failed: {err}"
        if not recovered:
            logger.info('%s', f'[lmstudio] 调用失败：{err}')
            _dump_request("error", payload, {"error": err})
            return None if return_meta else ""
    msg = resp.choices[0].message
    _dump_response("response", payload, resp)
    # 思维链：LM Studio 在 OpenAI 兼容结构里放 reasoning_content（output-only 字段，
    # 回发模型会触发 400 —— 出向由 ContextManager.prepare_messages 剥除）。
    # 仅 REPL 渲染用，UNATTEND 经剥除后净影响=0。
    reasoning = getattr(msg, "reasoning_content", None)
    # 原生工具调用优先：直接取 tool_calls 的 arguments（LM Studio 已正确转义）
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        args_list = []
        meta = []
        for tc in tool_calls:
            fn = tc.function
            raw_args = fn.arguments or "{}"
            try:
                obj = json.loads(raw_args)
                if isinstance(obj, dict):
                    # per-action 工具：把 function.name 注入为 action，下游 parse_actions /
                    # dispatch 仍按 action 字段路由（单一契约，无需改解析层）。
                    obj["action"] = fn.name
                    args_list.append(obj)
                    meta.append({"id": tc.id, "name": fn.name, "arguments": raw_args})
            except Exception:
                continue
        if args_list:
            STATS["calls"] += 1
            if return_meta:
                return {"type": "toolcalls", "actions": args_list, "tool_calls": meta,
                        "reasoning": reasoning, "content": (msg.content or "").strip()}
            return json.dumps(args_list, ensure_ascii=False)
    # 兜底：content 路径（模型未走工具调用时，把文本交给 parse_action 解析）
    content = (msg.content or "").strip()
    if content:
        STATS["calls"] += 1
        if return_meta:
            return {"type": "content", "content": content, "reasoning": reasoning}
        return content
    # 软空响应：server 正常返回 200，但模型既没走工具调用也没产出文本（本地模型常因畸形
    # tool call 被 LM Studio 丢弃）。与硬崩溃（上方 except 已返回 None）区分，返回 {"type":"empty"}，
    # 交由上层（executor loop → 外层 loop 的 pytest/tester 闸门）裁定，不触发退避重试。
    return {"type": "empty", "reasoning": reasoning} if return_meta else ""
