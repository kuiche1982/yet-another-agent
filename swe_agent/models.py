#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/models.py —— Provider 接入列表 + Model 目录（解耦 model 与 role）

分层（对齐架构设想）：
  1) Provider 接入列表（PROVIDERS）：只描述「怎么连到模型」——传输协议 + 端点 + 鉴权。
  2) Model 目录（MODELS）：每个模型登记自己的特性，核心把「函数调用能力」
     归一化为统一枚举 FCCapability——这一层「拉平」各家 Provider 的差异。
  3) Agent 角色只声明「需要什么能力」，运行时从目录里挑模型。

当前只保留两个标准 OpenAI 兼容 provider：
  - zhipu   ：远程智谱 GLM（OpenAI 兼容）
  - lmstudio：本机 LM Studio（OpenAI 兼容，http://localhost:1234/v1）
两者都走标准 OpenAI tool_calls 协议（由 llm_glm / llm_lmstudio 传输层注入工具 schema）。
旧的 plaintext / 文本 JSON 解析路径（rapid-mlx）已彻底废弃。

模型可随时换：只改一个 env（PLANNER_MODEL / EXECUTOR_MODEL），不碰角色代码。
"""

import json
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from . import config as C
from swe_agent.log import logger


class FCCapability(str, Enum):
    """函数调用能力（归一化枚举）。当前仅 NATIVE_TOOLS：所有 provider 走标准 OpenAI tool_calls。"""
    NATIVE_TOOLS = "native_tools"   # 支持 OpenAI tool_calls，可强制 tool_choice（GLM / LM Studio）


# 弱模型集合：上下文小、承载不了长规则 prompt，走 WEAK_SYSTEM_PROMPT 路径（见 ③）。
# 这些模型经 LM Studio 本地提供，已在 swe_agent/models.py 的 MODELS 注册为 lmstudio provider。
WEAK_EXECUTOR_MODELS = {"lfm2.5-2.6b", "liquid/lfm2.5-1.2b", "Ling-3.0-Tiny"}


def is_weak_executor(model: Optional[str] = None) -> bool:
    """当前 executor 是否为弱模型（上下文小、需精简 prompt）。

    model 为空时取角色默认 EXECUTOR_MODEL；非空时按传入 id 判定。
    用于 build_system_prompt 选 WEAK_SYSTEM_PROMPT、以及 tools 选更紧的写入上限。
    """
    mid = model or role_model_id("executor")
    return bool(mid) and mid in WEAK_EXECUTOR_MODELS


# ======================================================================
# 1) Provider 接入列表（只描述传输）
# ======================================================================
# 仅保留两个标准 OpenAI 兼容 provider：远程 zhipu(GLM) 与本机 lmstudio。
PROVIDERS: Dict[str, dict] = {
    "zhipu": {
        # 远程 OpenAI 兼容（智谱 GLM）
        "transport": "remote_openai",
        "base_url": C.GLM_BASE_URL,
        "api_key": C.GLM_API_TOKEN,
    },
    "lmstudio": {
        # 本机 OpenAI 兼容（LM Studio）
        "transport": "localhost_openai",
        "base_url": C.LMSTUDIO_BASE_URL,
        "api_key": C.LMSTUDIO_API_KEY,
    },
}


# ======================================================================
# 2) Model 目录（登记各自特性，fc 能力归一化）
# ======================================================================
# 每个条目：provider（指向 PROVIDERS 的 key）+ fc（FCCapability）+ 默认生成上限。
# 全部为 NATIVE_TOOLS（标准 OpenAI toolcall）。
MODELS: Dict[str, dict] = {
    # —— 远程 GLM（原生 tool_calls）——
    # glm-4.7：当前默认 zhipu 模型（临时代替 glm-4.5-flash / glm-4.7-flash）；
    #   改名只改 config 里 GLM_MODEL / PLANNER_MODEL 的 env 默认值即可，
    #   本目录与其余模型不绑定任何角色。
    # load_unload=False：远程模型无需本机 VRAM 换入换出，工厂不绑 ModelManager hook（零开销）。
    # context_length：目标模型上下文窗口（token），供 ContextManager.prepare_messages 推导预算；
    #   统一取自 config.MODEL_CONTEXT_LENGTH（默认 96000，env 可覆盖），后续可按模型单独改。
    "glm-4.7":        {"provider": "zhipu", "fc": FCCapability.NATIVE_TOOLS, "max_tokens": C.GLM_MAX_TOKENS, "load_unload": False, "context_length": C.MODEL_CONTEXT_LENGTH},
    "glm-4.7-flash":  {"provider": "zhipu", "fc": FCCapability.NATIVE_TOOLS, "max_tokens": C.GLM_MAX_TOKENS, "load_unload": False, "context_length": C.MODEL_CONTEXT_LENGTH},
    "glm-4.5-flash":  {"provider": "zhipu", "fc": FCCapability.NATIVE_TOOLS, "max_tokens": C.GLM_MAX_TOKENS, "load_unload": False, "context_length": C.MODEL_CONTEXT_LENGTH},
    # —— 本机 LM Studio（liquid/lfm2.5-1.2b，原生 tool_calls）——
    # load_unload=True：本地模型在 loop 执行时 load、执行完 unload（显存只驻留一两个模型）。
    "liquid/lfm2.5-1.2b":           {"provider": "lmstudio", "fc": FCCapability.NATIVE_TOOLS, "max_tokens": C.LMSTUDIO_MAX_TOKENS, "load_unload": True, "context_length": C.MODEL_CONTEXT_LENGTH},
    # —— 本机 LM Studio（lfm2.5-2.6b，原生 tool_calls；合格的对话压缩副驾）——
    "lfm2.5-2.6b":                  {"provider": "lmstudio", "fc": FCCapability.NATIVE_TOOLS, "max_tokens": C.LMSTUDIO_MAX_TOKENS, "load_unload": True, "context_length": C.MODEL_CONTEXT_LENGTH},
    "Ling-3.0-Tiny":                {"provider": "lmstudio", "fc": FCCapability.NATIVE_TOOLS, "max_tokens": C.LMSTUDIO_MAX_TOKENS, "load_unload": True, "context_length": C.MODEL_CONTEXT_LENGTH},
    "Spark-X2.5-1.7B":              {"provider": "lmstudio", "fc": FCCapability.NATIVE_TOOLS, "max_tokens": C.LMSTUDIO_MAX_TOKENS, "load_unload": True, "context_length": C.MODEL_CONTEXT_LENGTH},
    "qwen3.5-4b-mtplx-optimized-speed": {"provider": "lmstudio", "fc": FCCapability.NATIVE_TOOLS, "max_tokens": C.LMSTUDIO_MAX_TOKENS, "load_unload": True, "context_length": C.MODEL_CONTEXT_LENGTH},
    # —— 本机 LM Studio（qwen2.5.1-coder-7b-instruct，原生 tool_calls）——
    "qwen2.5.1-coder-7b-instruct": {"provider": "lmstudio", "fc": FCCapability.NATIVE_TOOLS, "max_tokens": C.LMSTUDIO_MAX_TOKENS, "load_unload": True, "context_length": C.MODEL_CONTEXT_LENGTH},
}


# ======================================================================
# 3) 角色 → 模型（可随时从 Provider 表换）
# ======================================================================
# 角色只声明「需要什么能力」，运行时挑满足能力的模型。
# 默认模型可用 env 覆盖——这就是「模型可换」的开关，不绑定任何角色代码。
# 注意：这里在【调用时】实时读取 config 上的对应属性（不缓存），
# 这样 CLI（main）在运行时改 config.EXECUTOR_MODEL 等能立刻生效。
ROLE_ENV: Dict[str, str] = {
    "planner": "PLANNER_MODEL",
    "analyzer": "ANALYZER_MODEL",
    "executor": "EXECUTOR_MODEL",
}

# 升级子 agent：弱模型（如 liquid/lfm2.5-1.2b）主产出不可用时，升级给此强脑子 agent
# 做复杂判断。仅主模型失败触发，非每轮调用，避免空耗远程配额。env 可覆盖（如 *_FALLBACK_MODEL）。
ROLE_FALLBACK_ENV: Dict[str, str] = {
    "planner": "PLANNER_FALLBACK_MODEL",
    "analyzer": "ANALYZER_FALLBACK_MODEL",
}


def role_fallback(role: str) -> Optional[str]:
    """角色升级目标模型 id；未配置 / 空串返回 None（不升级，优雅降级）。"""
    env_name = ROLE_FALLBACK_ENV.get(role)
    if not env_name:
        return None
    return getattr(C, env_name, None) or None


class ModelUnknown(Exception):
    def __init__(self, role: str, model: str):
        super().__init__(
            f"角色 {role!r} 分配的模型 {model!r} 不在 Model 目录中；"
            f"请在 config 的 ROLE_ENV 对应项或 env 里改成已登记的模型 id。"
        )


def role_model_id(role: str) -> Optional[str]:
    """角色当前分配到的模型 id；空串/未登记=该角色不挂模型（由调用方降级）。"""
    env_name = ROLE_ENV.get(role)
    if not env_name:
        return None
    val = getattr(C, env_name, None)
    return val or None


def role_spec(role: str) -> Optional[dict]:
    """返回角色模型的目录条目（含 provider/fc/max_tokens/load_unload/id）；无模型返回 None。"""
    mid = role_model_id(role)
    if not mid:
        return None
    spec = MODELS.get(mid)
    if spec is None:
        raise ModelUnknown(role, mid)
    return {**spec, "id": mid}


def model_load_unload(model_id: Optional[str]) -> bool:
    """单个模型是否要求 per-loop load/unload。

    收口点（GLOBAL 机制）：所有「要不要换模」的判断都必须走这里，由
    config.MODEL_LOAD_UNLOAD 总开关统一关断（SWE_MODEL_LOAD_UNLOAD=0 → 一律 False）。
    远程模型（zhipu）load_unload=False，本来就不换模。
    """
    if not C.MODEL_LOAD_UNLOAD:
        return False
    return bool(model_id and MODELS.get(model_id, {}).get("load_unload"))


def role_load_unload(role: str) -> bool:
    """角色模型是否要求 per-loop load/unload（仅本机 lmstudio 模型为 True）。"""
    return model_load_unload(role_model_id(role))


def model_context_length(model_id: Optional[str]) -> int:
    """目标模型上下文窗口（token）：ContextManager.prepare_messages 按它推导 RAG/压缩预算。

    收口点（GLOBAL 机制）：统一从 Model 目录解析，未登记模型或目录缺字段时回退
    config.MODEL_CONTEXT_LENGTH（默认 96000，env MODEL_CONTEXT_LENGTH 可覆盖）。
    每模型可在 MODELS[...]["context_length"] 单独声明以覆盖全局默认。
    """
    if not model_id:
        return C.MODEL_CONTEXT_LENGTH
    entry = MODELS.get(model_id)
    if entry:
        return int(entry.get("context_length") or C.MODEL_CONTEXT_LENGTH)
    return C.MODEL_CONTEXT_LENGTH


def role_context_length(role: str) -> int:
    """角色当前模型的上下文窗口（token）；无模型返回全局默认。"""
    return model_context_length(role_model_id(role))


def role_provider(role: str) -> Optional[str]:
    spec = role_spec(role)
    return spec["provider"] if spec else None


def role_fc(role: str) -> Optional[FCCapability]:
    spec = role_spec(role)
    return spec["fc"] if spec else None


# ======================================================================
# 统一调度（按 provider transport 路由到对应传输层）
# ======================================================================
def _with_retry(role: str, fn, empty_ok: bool = False):
    """GLOBAL 重试闸门：本地模型（lmstudio/qwen）偶发崩溃会抛异常或返回空/None，
    统一在此退避重试，避免一次瞬态抖动杀死整个 agent run。

    - 合法「无模型」早退（role_spec 为 None 时函数已提前返回，不走这里）；
    - 这里只包 provider 调用本身，因此只对真实调用失败重试，不影响正常早退语义。
    - empty_ok=False 时，空串 ''（lmstudio 失败约定）也视为失败并重试；
      chat_toolcalls 走 return_meta 返回 None，None 一律视为失败。
    """
    last = None
    for _i in range(C.MODEL_RETRY + 1):
        try:
            last = fn()
        except Exception as e:
            last = None
            logger.info('%s', f'[model] {role} 调用异常（{type(e).__name__}: {e}），重试 {_i + 1}/{C.MODEL_RETRY}…')
        _ok = last is not None and (empty_ok or last != "")
        if _ok:
            return last
        if _i < C.MODEL_RETRY:
            logger.info('%s', f'[model] {role} 第 {_i + 1} 次返回空/None，退避 {C.MODEL_BACKOFF}s 后重试…')
            time.sleep(C.MODEL_BACKOFF)
    return last


# 软空响应哨兵：provider 正常返回 200，但模型既没走工具调用也没产出文本。
# 常见于本地模型（lmstudio/qwen）生成了 LM Studio 无法解析的畸形 tool call，被 server 丢弃。
# 与硬崩溃（异常 / 连接失败 → 返回 None）区分：软空不触发退避重试（同 prompt 重试只会得到同样空响应），
# 交由上层（executor loop → 外层 loop 的 pytest/tester 闸门）裁定。
EMPTY_RESPONSE = {"type": "empty"}


def is_empty(meta) -> bool:
    """判断是否为软空响应（server 正常但模型无 tool call 且无 content）。"""
    return isinstance(meta, dict) and meta.get("type") == "empty"


# ======================================================================
def chat_text(role: str, system: str, user: str,
              temperature: float = 0.3, max_tokens: Optional[int] = None,
              response_format: Optional[Dict[str, Any]] = None,
              model_override: Optional[str] = None,
              thinking_off: bool = False) -> str:
    """Planner 用：system+user → content 文本。

    按角色模型的 provider 路由到对应传输层（无 tools，结构化 JSON 走 content）。
    model_override 非空时忽略角色默认模型，直接对该模型 id 发起调用（供升级子 agent 复用）。
    返回空串表示该角色无模型（调用方应降级到本地自规划 / harness 判定）。
    """
    if model_override:
        mid = model_override
        spec = MODELS.get(mid)
        if spec is None:
            raise ModelUnknown(role, mid)
        spec = {**spec, "id": mid}
    else:
        spec = role_spec(role)
        if not spec:
            return ""
    mid = spec["id"]
    model_name = spec.get("model_name", mid)
    mt = PROVIDERS[spec["provider"]]["transport"]
    mtok = max_tokens or spec["max_tokens"]
    response_format = _adapt_response_format(response_format, mt)
    if mt == "remote_openai":
        from . import llm_glm
        _thinking = {"type": C.GLM_THINKING} if C.GLM_THINKING else None
        return _with_retry(role, lambda: llm_glm.glm_chat(system, user, model=model_name,
                                                         temperature=temperature,
                                                         max_tokens=mtok, response_format=response_format,
                                                         thinking=_thinking), empty_ok=False)
    if mt == "localhost_openai":
        from . import llm_lmstudio
        return _with_retry(role, lambda: llm_lmstudio.lmstudio_chat_messages(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model_name, temperature=temperature, max_tokens=mtok,
            response_format=response_format, thinking_off=thinking_off), empty_ok=False)
    return ""


def _adapt_response_format(response_format, transport: str):
    """按 Provider 传输层归一化结构化输出格式（LM Studio 与 GLM 关键字不同）：

    - LM Studio（localhost_openai）：只接受 json_schema，拒绝 json_object → 保持 json_schema；
    - Zhipu/GLM（remote_openai）：用 json_object（json_schema 可能不被支持）→ json_schema 降级为 json_object。
    这样调用方（如 roles._planner_response_format 返回 json_schema）无需关心升级子 agent
    换了 provider，chat_text 按实际模型再适配一次。
    """
    if not response_format:
        return None
    if transport == "localhost_openai":
        return response_format  # 调用方应已给 json_schema
    if transport == "remote_openai":
        if response_format.get("type") == "json_schema":
            return {"type": "json_object"}
        return response_format
    return response_format


def chat_text_escalating(role: str, system: str, user: str,
                         temperature: float = 0.3, max_tokens: Optional[int] = None,
                         response_format: Optional[Dict[str, Any]] = None,
                         require_json: bool = False,
                         model_override: Optional[str] = None) -> str:
    """弱模型优先，失败升级给 ROLE_FALLBACK 指定的强脑子 agent 做复杂判断。

    - 主模型（如 liquid/lfm2.5-1.2b）先答；
    - require_json=True 时，仅当主产出为空或无法解析为 JSON 才升级（避免「能答但格式松」被误升级）；
    - 无 fallback 配置时直接返回主模型结果（优雅降级，不报错）。
    - model_override 非空时忽略角色默认模型（供升级子 agent / 显式覆盖复用）。
    """
    primary = chat_text(role, system, user, temperature=temperature,
                        max_tokens=max_tokens, response_format=response_format,
                        model_override=model_override)
    if primary and (not require_json or _looks_like_json(primary)):
        return primary
    fb = role_fallback(role)
    if not fb:
        return primary
    logger.info('%s', f'[escalate] {role} 主模型产出不可用，升级子 agent {fb} 做复杂判断。')
    return chat_text(role, system, user, temperature=temperature,
                     max_tokens=max_tokens, response_format=response_format,
                     model_override=fb)


def _looks_like_json(s: Optional[str]) -> bool:
    """粗判：以 { 或 [ 开头的非空串视为可能 JSON（用于升级门控）。"""
    s = (s or "").strip()
    return bool(s) and s[0] in "{["


# ======================================================================
# Judge（结构化输出裁判，2026-09-03；取代 F3 长度阈值 + 启发式判断）
# 用【独立模型】按显式标准评「产出算不算数」，输出 json_schema {result:yes/no/notsure, reason}。
# 独立于角色模型（默认 qwen），避免生产者自判偏差。
# ======================================================================
JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_result",
        "strict": "true",
        "schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string", "enum": ["yes", "no", "notsure"]},
                "reason": {"type": "string"},
            },
            "required": ["result", "reason"],
            "additionalProperties": False,
        },
    },
}


def judge_params(model_id: str) -> Dict[str, Any]:
    """按模型名子串解析 judge 生成参数（BUILD 配置，不硬编码全局）。

    默认 JUDGE_TEMPERATURE / JUDGE_MAX_TOKENS / thinking_off=False；
    命中 JUDGE_MODEL_OVERRIDES 子串（ling/lfm）则覆盖——思考类模型需特殊处理。
    """
    base = {"temperature": C.JUDGE_TEMPERATURE, "max_tokens": C.JUDGE_MAX_TOKENS, "thinking_off": False}
    for key, ov in C.JUDGE_MODEL_OVERRIDES.items():
        if key in (model_id or "").lower():
            base.update(ov)
            break
    return base


def judge(kind: str, content: str, model_override: Optional[str] = None) -> Tuple[str, str]:
    """结构化裁判：kind ∈ {analyzer, tester}。返回 (result, reason)，result∈{yes,no,notsure}。

    - 解析失败 / 模型未登记 / 调用失败 → (notsure, 原因)；调用方据此重试或降级 no。
    - 独立 JUDGE_MODEL（默认 qwen），与生产者模型解耦，避免「生产者自判产出」偏差。
    - 硬 prompt 铁律（config.JUDGE_PROMPTS）已内置「空/占位符/客套话→必须 no」。
    """
    mid = model_override or C.JUDGE_MODEL
    if mid not in MODELS:
        return "notsure", f"judge 模型 {mid!r} 未登记于 Model 目录"
    tmpl = C.JUDGE_PROMPTS.get(kind)
    if not tmpl:
        return "notsure", f"未知 judge 类型 {kind!r}"
    params = judge_params(mid)
    raw = chat_text(role="analyzer", system="", user=tmpl.format(content=content),
                    temperature=params["temperature"], max_tokens=params["max_tokens"],
                    response_format=JUDGE_SCHEMA, model_override=mid,
                    thinking_off=params["thinking_off"])
    if not raw:
        return "notsure", "judge 模型调用失败/空响应"
    # 解析容错：剥 ```json 围栏，抽首个 { 到末个 }（弱模型偶发在 JSON 前后带闲话）
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```", 2)
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s >= 0 and e > s:
        cleaned = cleaned[s:e + 1]
    try:
        val = json.loads(cleaned)
    except Exception:
        return "notsure", f"judge 返回无法解析为 JSON：{raw[:80]!r}"
    res = str(val.get("result") or "").strip().lower()
    reason = str(val.get("reason") or "").strip()
    if res not in ("yes", "no", "notsure"):
        return "notsure", f"judge 返回非法 result={res!r}"
    return res, reason


def chat_text_messages(messages: list, role: str = "planner",
                       temperature: float = 0.3, max_tokens: Optional[int] = None,
                       model_override: Optional[str] = None) -> str:
    """文本补全（摘要 / 压缩用）：完整 messages → content 文本，走 glm/lmstudio 的 content 通道。

    标准 OpenAI 调用，不传 tools（模型直接产出总结文本）。返回空串表示无模型或调用失败。
    供 compact.py 的摘要生成复用，替代原 rapid-mlx plaintext 路径。
    """
    if model_override:
        mid = model_override
        spec = MODELS.get(mid)
        if spec is None:
            raise ModelUnknown(role, mid)
        spec = {**spec, "id": mid}
    else:
        spec = role_spec(role)
        if not spec:
            return ""
    mid = spec["id"]
    model_name = spec.get("model_name", mid)
    mt = PROVIDERS[spec["provider"]]["transport"]
    mtok = max_tokens or spec["max_tokens"]
    if mt == "remote_openai":
        from . import llm_glm
        _thinking = {"type": C.GLM_THINKING} if C.GLM_THINKING else None
        return llm_glm.glm_chat_messages(messages, model=model_name, temperature=temperature,
                                         max_tokens=mtok, thinking=_thinking)
    if mt == "localhost_openai":
        from . import llm_lmstudio
        return llm_lmstudio.lmstudio_chat_messages(messages, model=model_name,
                                                  temperature=temperature, max_tokens=mtok)
    return ""


def chat_messages(role: str = "executor", messages: list = None,
                  temperature: float = 0.3, max_tokens: Optional[int] = None,
                  tools=None, model_override: Optional[str] = None,
                  tool_choice=None) -> str:
    """Executor / Analyzer 用：messages → action JSON 字符串（标准 OpenAI toolcall）。

    按角色模型 provider 路由；tools 恒传（所有模型均为 NATIVE_TOOLS，强制 tool_choice）。
    role 默认 executor；model_override 非空时忽略角色默认模型。
    tool_choice 仅透传给原生工具调用层（用于「强制某工具调用」，如 Analyzer 升级时
    强制 finish_research）。无模型时返回空串（调用方降级）。
    """
    if model_override:
        mid = model_override
        spec = MODELS.get(mid)
        if spec is None:
            return ""
        spec = {**spec, "id": mid}
    else:
        spec = role_spec(role)
        if spec is None:
            return ""
    mid = spec["id"]
    model_name = spec.get("model_name", mid)   # 允许 catalog 条目用 model_name 覆盖实际下发模型名
    mt = PROVIDERS[spec["provider"]]["transport"]
    mtok = max_tokens or spec["max_tokens"]
    use_tools = tools if spec["fc"] == FCCapability.NATIVE_TOOLS else None
    if mt == "remote_openai":
        from . import llm_glm
        _thinking = {"type": C.GLM_THINKING} if C.GLM_THINKING else None
        return _with_retry(role, lambda: llm_glm.glm_chat_messages(messages, model=model_name,
                                                                  temperature=temperature,
                                                                  max_tokens=mtok, tools=use_tools,
                                                                  thinking=_thinking, tool_choice=tool_choice),
                           empty_ok=False)
    if mt == "localhost_openai":
        from . import llm_lmstudio
        return _with_retry(role, lambda: llm_lmstudio.lmstudio_chat_messages(messages, model=model_name,
                                                                             temperature=temperature,
                                                                             max_tokens=mtok, tools=use_tools,
                                                                             tool_choice=tool_choice),
                           empty_ok=False)
    return ""


def chat_toolcalls(role: str = "executor", messages: list = None,
                   temperature: float = 0.3, max_tokens: Optional[int] = None,
                   tools=None, model_override: Optional[str] = None,
                   tool_choice=None) -> Optional[Dict[str, Any]]:
    """原生 toolcall 协议版 chat_messages：返回结构化 dict（供 loop_3 / tester 使用）。

    返回 {"type":"toolcalls","actions":[...],"tool_calls":[{"id","name","arguments"}]}
           {"type":"content","content":"..."}（模型未走工具）
           None（调用失败 / 空响应）
    与 chat_messages 的区别：保留 tool_call_id 与原生 assistant(tool_calls) 结构，
    调用方据此拼出正确的 assistant/tool 交替消息，避免把工具结果拍平进 user 消息
    （那会诱使弱模型退化成「续写散文」而非调用工具）。
    """
    if model_override:
        mid = model_override
        spec = MODELS.get(mid)
        if spec is None:
            return None
        spec = {**spec, "id": mid}
    else:
        spec = role_spec(role)
        if spec is None:
            return None
    mid = spec["id"]
    model_name = spec.get("model_name", mid)
    mt = PROVIDERS[spec["provider"]]["transport"]
    mtok = max_tokens or spec["max_tokens"]
    use_tools = tools if spec["fc"] == FCCapability.NATIVE_TOOLS else None
    if mt == "remote_openai":
        from . import llm_glm
        _thinking = {"type": C.GLM_THINKING} if C.GLM_THINKING else None
        return _with_retry(role, lambda: llm_glm.glm_chat_messages(messages, model=model_name,
                                                                  temperature=temperature,
                                                                  max_tokens=mtok, tools=use_tools,
                                                                  thinking=_thinking, tool_choice=tool_choice,
                                                                  return_meta=True), empty_ok=False)
    if mt == "localhost_openai":
        from . import llm_lmstudio
        return _with_retry(role, lambda: llm_lmstudio.lmstudio_chat_messages(messages, model=model_name,
                                                                             temperature=temperature,
                                                                             max_tokens=mtok, tools=use_tools,
                                                                             tool_choice=tool_choice,
                                                                             return_meta=True), empty_ok=False)
    return None


def describe_config() -> str:
    """人类可读的当前角色→模型→provider 解析结果（调试/日志用）。"""
    lines = ["# 模型/Provider 解析："]
    for role in ("planner", "executor"):
        mid = role_model_id(role)
        if not mid:
            lines.append(f"  - {role}: (无模型，降级本地)")
            continue
        spec = role_spec(role)
        lines.append(f"  - {role}: {mid}  → provider={spec['provider']}, fc={spec['fc'].value}")
    return "\n".join(lines)
