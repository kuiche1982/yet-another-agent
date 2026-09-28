#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/llm_glm.py —— 远程 zhipu（GLM）Provider 的传输层（Transport）

本模块只负责「怎么调用 zhipu 的 OpenAI 兼容接口」，不绑定任何 agent 角色：
- glm_chat：system+user → content（Planner 用，结构化 JSON 走 content 提取）；
- glm_chat_messages：完整 messages + 可选 tools → 原生 function calling（Executor 用）；
- _glm_extract_answer / _extract_json_object：传输无关的内容清洗 / JSON 提取辅助。

模型 id 由调用方（models.py 按角色解析）通过 model= 参数传入，默认 GLM_MODEL。
角色编排逻辑（planner/故障路由）已移至 roles.py，不在本文件。
"""

import os
import re
import time
import json
import requests
from typing import Dict, Any, List, Optional

from .config import (
    GLM_API_TOKEN, GLM_BASE_URL, GLM_MODEL,
    GLM_MAX_TOKENS, GLM_TIMEOUT, STATS,
)
from swe_agent.log import logger


_GLM_STATS_KEYS = ("glm_calls", "glm_tokens")

# zhipu 调用默认【直连】，绕过本机 sandbox 代理（127.0.0.1:50606，即 WorkBuddy 的
# sandbox-c 进程）。实测该代理会接手所有出站 LLM 请求：要么换 key 中转、要么缓存/拦截，
# 导致 zhipu 服务端用 .env 的 GLM_API_TOKEN 看不到任何消耗记录，且偶发 read timeout。
# 设 GLM_USE_PROXY=1 可恢复走系统代理（兼容必须走代理才能出网的环境）。
_GLM_PROXIES = ({"http": None, "https": None}
                if os.environ.get("GLM_USE_PROXY") != "1" else None)


_GLM_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_GLM_ANSWER_OPEN_RE = re.compile(r"<answer>(.*)$", re.DOTALL)


def _glm_extract_answer(content: str) -> str:
    """从 GLM 返回的 content 里提取最终答案。

    实测 glm-4.1v-thinking-flashx 的 chat 接口【不】分离 reasoning_content：
    content = 思考过程 + <answer>答案</answer>（有时尾部还有零星说明文字）。
    提取优先级：
      1) <answer>...</answer> 完整块；
      2) 未闭合的 <answer>（输出被 max_tokens 截断时）：取 <answer> 之后到结尾；
      3) <|begin_of_box|>...<|end_of_box|>（部分 GLM 版本的答案框）；
      4) 原文（无标签时视为纯答案）。
    """
    if not content:
        return ""
    m = _GLM_ANSWER_RE.search(content)
    if m:
        return m.group(1).strip()
    m = _GLM_ANSWER_OPEN_RE.search(content)
    if m:
        return m.group(1).strip()
    m = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return content.strip()


def glm_chat(system: str, user: str, model: str = GLM_MODEL, temperature: float = 0.3,
             max_tokens: int = GLM_MAX_TOKENS, response_format: Optional[Dict[str, Any]] = None,
             thinking: Optional[Dict[str, Any]] = None) -> str:
    """调用远程 zhipu（OpenAI 兼容协议）。返回 content 文本；失败返回空串（由调用方降级）。

    response_format：结构化输出约束（如 {"type":"json_object"}）。GLM 不支持 json_schema，
    仅 json_object 生效；传 None 则普通文本输出。
    thinking：思维链开关，如 {"type":"disabled"}。若该模型不支持此参数（zhipu 文档称仅 4.5+
    支持），API 会以 4xx 拒绝 —— 本函数自动去掉 thinking 重试一次，不会让调用方拿到空串。
    """
    if not GLM_API_TOKEN:
        return ""
    drop_thinking = False
    for attempt in range(3):
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            if thinking is not None and not drop_thinking:
                payload["thinking"] = thinking
            resp = requests.post(
                f"{GLM_BASE_URL}/chat/completions",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GLM_API_TOKEN}",
                },
                timeout=GLM_TIMEOUT,
                proxies=_GLM_PROXIES,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            STATS["glm_calls"] += 1
            STATS["glm_tokens"] += (usage.get("prompt_tokens", 0) or 0) + \
                                   (usage.get("completion_tokens", 0) or 0)
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            content = (msg.get("content") or "").strip()
            # 思考模型：该端点不分离 reasoning_content，思考混在 content 里且答案
            # 包在 <answer>…</answer> 标签中 —— 提取纯答案，思考过程不进入输出。
            answer = _glm_extract_answer(content)
            if answer:
                return answer
            last_err_txt = "empty content"
        except Exception as e:
            last_err_txt = str(e)
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                if thinking is not None and not drop_thinking:
                    # 该模型可能不支持 thinking.type（如 4.1 系）——去掉后重试一次
                    drop_thinking = True
                    last_err_txt = f"thinking 参数被 API 拒绝（{status}），去掉 thinking 重试"
                    continue
                logger.info('%s', f'[glm] 客户端错误（{status}），不重试：{e}')
                return ""
            wait = min(2 * (2 ** attempt), 30)
            logger.info('%s', f'[glm] 调用失败（{e}），{wait:.1f}s 后重试……')
            time.sleep(wait)
    logger.info('%s', f'[glm] 调用最终失败：{last_err_txt}')
    return ""


def glm_chat_messages(messages: List[Dict[str, str]], model: str = GLM_MODEL,
                      temperature: float = 0.3, max_tokens: int = GLM_MAX_TOKENS,
                      tools: Optional[List[Dict]] = None,
                      thinking: Optional[Dict[str, Any]] = None,
                      tool_choice: Optional[Dict[str, Any]] = None,
                      return_meta: bool = False) -> Any:
    """zhipu 充当 Executor：直接接收完整 messages 列表（含 system/user/assistant 多轮对话）。
    - 传入 tools 时优先走原生 function calling：返回 tool_calls 的 arguments 组成的 JSON 数组字符串，
      由 harness 的 parse_action 正常解析（此时无需手写 JSON，规避 GLM 的非法 JSON 形态）。
    - 未传 tools 时退回「content + 提取 <answer> 答案」路径。
    thinking：思维链开关，如 {"type":"disabled"}。若模型不支持（4.1 系可能 4xx 拒绝），自动去掉重试。
    失败返回空串（由 chat() 降级处理）。
    return_meta=True 时返回结构化 dict（供原生 toolcall 协议使用，见 lmstudio_chat_messages）。"""
    if not GLM_API_TOKEN:
        return None if return_meta else ""
    drop_thinking = False
    for attempt in range(3):
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                payload["tools"] = tools
                # 强制 GLM 必须调用工具，禁止它退化成手写文本 JSON（思考模型常把多行
                # 代码包成 ''' 导致解析出 0 字节 / 无限循环）。没有 tool_calls 就重试。
                # tool_choice 显式传入时（如 Analyzer 升级强制 finish_analysis）优先采用，
                # 否则默认 "required" 强制一次工具调用（schema 内已是 per-action 函数集，
                # 由提示词引导模型选对动作）。
                payload["tool_choice"] = tool_choice if tool_choice is not None \
                    else "required"
            if thinking is not None and not drop_thinking:
                payload["thinking"] = thinking
            resp = requests.post(
                f"{GLM_BASE_URL}/chat/completions",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GLM_API_TOKEN}",
                },
                timeout=GLM_TIMEOUT,
                proxies=_GLM_PROXIES,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            STATS["glm_calls"] += 1
            STATS["glm_tokens"] += (usage.get("prompt_tokens", 0) or 0) + \
                                   (usage.get("completion_tokens", 0) or 0)
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            # 原生工具调用优先：直接取 tool_calls 的 arguments（GLM 已正确转义）
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                args_list = []
                meta = []
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        obj = json.loads(raw_args)
                        if isinstance(obj, dict):
                            # per-action 工具：把 function.name 注入为 action，下游 parse_actions /
                            # dispatch 仍按 action 字段路由（单一契约，无需改解析层）。
                            obj["action"] = fn.get("name")
                            args_list.append(obj)
                            meta.append({"id": tc.get("id"), "name": fn.get("name"),
                                         "arguments": raw_args})
                    except Exception:
                        continue
                if args_list:
                    if return_meta:
                        return {"type": "toolcalls", "actions": args_list, "tool_calls": meta}
                    return json.dumps(args_list, ensure_ascii=False)
            # 退回 content 路径（思考模型答案包在 <answer>…</answer> 中）
            content = (msg.get("content") or "").strip()
            answer = _glm_extract_answer(content)
            if answer:
                if return_meta:
                    return {"type": "content", "content": answer}
                return answer
            last_err_txt = "empty content"
        except Exception as e:
            last_err_txt = str(e)
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                if thinking is not None and not drop_thinking:
                    # 该模型可能不支持 thinking.type（如 4.1 系）——去掉后重试一次
                    drop_thinking = True
                    last_err_txt = f"thinking 参数被 API 拒绝（{status}），去掉 thinking 重试"
                    continue
                logger.info('%s', f'[glm-exec] 客户端错误（{status}），不重试：{e}')
                return None if return_meta else ""
            wait = min(2 * (2 ** attempt), 30)
            logger.info('%s', f'[glm-exec] 调用失败（{e}），{wait:.1f}s 后重试……')
            time.sleep(wait)
    logger.info('%s', f'[glm-exec] 调用最终失败：{last_err_txt}')
    # return_meta 模式下「无可用 toolcall/content 响应」视为软空响应（区别于硬崩溃客户端错误
    # 在上方已 return None），交由上层裁定；非 return_meta（chat_text）仍返回空串走重试。
    return {"type": "empty"} if return_meta else ""


def _repair_json(s: str) -> str:
    """把弱模型产出的「字符串值里含未转义双引号」的非法 JSON 修复为合法 JSON。

    弱模型（如 qwen-7b）常在 content / old_string 等字段里写三引号文档串或散落
    字面量引号，但没按 JSON 规范转义成反斜杠引号。标准 json.loads 会直接报错，而
    宽松正则兜底会在第一个引号处截断字符串——导致 write_file 只写进文件头（实测只
    落 46 字节，整个函数体被吞掉）。本函数在字符串值内部把「非字符串终结符的引号」
    转义，既不丢内容，也不破坏合法的字符串边界。

    判定规则：处于字符串值内部时遇到引号，向后跳过空白看下一个字符——
    - 是 '}' / ']' / ',' 或到串尾 → 这是合法的字符串终结符，保留；
    - 否则 → 视为字符串值内部的字面量引号，转义成反斜杠引号。
    （已转义的引号由 esc 状态保留，不会被二次转义。）
    """
    if not s:
        return s
    out: List[str] = []
    in_str = False
    esc = False
    n = len(s)
    # 字符串终结符：引号后紧跟这些字符之一（跳过空白）即视为合法的字符串边界。
    # - ':'  → 键名闭合（"key":）
    # - ','  → 值闭合后接下一项（"val",）
    # - '}' / ']' → 值闭合后接结构闭合
    # 其余情况（字母 / 空白后的代码 / 另一个引号）→ 视为字符串值内部的字面量引号，转义。
    _TERM = ":,}]"
    # 字符串值内部的控制字符（弱模型直接输出真实换行/制表符而非 JSON 转义）也必须转义，
    # 否则 json.loads 报 Invalid control character。
    _CTRL = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    i = 0
    while i < n:
        ch = s[i]
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            elif ch == "\\":
                out.append(ch)
                esc = True
            elif ch == '"':
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                nxt = s[j] if j < n else ""
                if nxt in _TERM or nxt == "":
                    out.append(ch)          # 合法字符串终结符
                    in_str = False
                else:
                    out.append('\\"')        # 字符串值内部的字面量引号 → 转义
            elif ch in _CTRL:
                out.append(_CTRL[ch])        # 真实控制字符 → JSON 转义
            else:
                out.append(ch)
        else:
            if ch == '"':
                out.append(ch)
                in_str = True
            else:
                out.append(ch)
        i += 1
    return "".join(out)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出里稳健提取第一个完整 JSON 对象（容忍 ```json 围栏、前后杂文）。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    # 定位首个 '{'，做花括号配平扫描（容忍字符串内的花括号）
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    # 容忍字符串值内的未转义引号：修复后重试一次
                    try:
                        obj = json.loads(_repair_json(t[start:i + 1]))
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None
