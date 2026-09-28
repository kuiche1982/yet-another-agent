"""LLM 后端（可选扩展点）—— 真模型蒸馏器，验证 contextmgr 在真实模型下的流程。

注意：
- 这是 **可选插件**，默认 contextmgr 走 model-free 的 KeywordDistiller + BM25Embedder，
  不依赖任何 LLM（单测全 model-free/tool-free）。
- 本模块仅在「想用真模型生成 L1/L2/L3」时启用，例如 lfm2.5-2.6b（nothinking）。
- 接口严格对齐 Distiller ABC，因此可无缝替换 store.distill 的蒸馏器。
- 关闭思考：lfm 实测 `enable_thinking: false` 顶层字段即可（reasoning_len=0）；
  若换 Ling 后端需改 chat_template_kwargs（见 docs/contextmgr_dev.md）。
"""

from __future__ import annotations

import json
import re
import time

from openai import OpenAI

from .distill import Distiller
from .store import FragmentStore
from .tokenize import estimate_tokens
from .types import Fragment, L1Structured, L2Visual, L3Index, Source

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"

# 蒸馏输出契约：让模型以 json_schema 返回结构化 L1 + L2(mermaid) + L3 索引
_DISTILL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_distill",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "entities": {"type": "array", "items": {"type": "string"}},
                "mermaid": {"type": "string"},
                "l3_label": {"type": "string"},
                "l3_category": {
                    "type": "string",
                    "enum": ["concept", "howto", "api", "bug", "design", "fact"],
                },
            },
            "required": [
                "summary",
                "key_points",
                "entities",
                "mermaid",
                "l3_label",
                "l3_category",
            ],
        },
    },
}


class LLMDistiller(Distiller):
    """用真模型（默认 lfm2.5-2.6b nothinking）生成 L1/L2/L3。

    - 仅对文本型片段（Library/Session）走 LLM；代码源仍用 CodeDistiller（AST，model-free）。
    - 失败兜底：LLM 不可用/超时/解析失败时回退到 KeywordDistiller，保证链路不崩。
    """

    def __init__(
        self,
        model: str = "lfm2.5-2.6b",
        base_url: str = LMSTUDIO_BASE_URL,
        api_key: str = LMSTUDIO_API_KEY,
        timeout: float = 180.0,
        fallback: Distiller | None = None,
    ) -> None:
        self.model = model
        self._fallback = fallback or _KeywordFallback()
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def distill(self, frag: Fragment) -> tuple[L1Structured, L2Visual, L3Index]:
        # 代码源不走 LLM（AST 已足够且更准）
        if frag.source is Source.CODE:
            return self._fallback.distill(frag)
        try:
            return self._llm_distill(frag)
        except Exception:  # noqa: BLE001 - 任何失败都兜底，绝不炸链路
            return self._fallback.distill(frag)

    def _llm_distill(self, frag: Fragment) -> tuple[L1Structured, L2Visual, L3Index]:
        sysmsg = (
            "你是记忆蒸馏器。给定一段文本，输出 JSON：\n"
            "summary=一句话摘要；key_points=3-5 个要点；entities=命名实体；\n"
            "mermaid=用 flowchart/classDiagram/mindmap 画一张合法 mermaid（禁止用 end 作节点id、"
            "禁止嵌套方括号）；l3_label=一行可检索索引（<=16 词，含关键实体）；"
            "l3_category=concept/howto/api/bug/design/fact 之一。\n"
            "只输出 JSON。"
        )
        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sysmsg},
                {"role": "user", "content": frag.text},
            ],
            temperature=0,
            max_tokens=1500,
            response_format=_DISTILL_SCHEMA,
            extra_body={"enable_thinking": False},
        )
        dt = time.time() - t0
        content = resp.choices[0].message.content or "{}"
        try:
            d = json.loads(content)
        except json.JSONDecodeError:
            d = {}
        mermaid = d.get("mermaid", "") or ""
        mermaid = _strip_mermaid_fence(mermaid)
        l1 = L1Structured(
            fid=frag.fid,
            summary=d.get("summary", "") or frag.text[:80],
            key_points=d.get("key_points", []) or [frag.text[:80]],
            entities=d.get("entities", []),
        )
        l2 = L2Visual(fid=frag.fid, diagram=mermaid, tokens=estimate_tokens(mermaid))
        l3 = L3Index(
            fid=frag.fid,
            source=frag.source,
            priority=frag.priority,
            label=(d.get("l3_label", "") or l1.summary)[:160],
            category=d.get("l3_category", "fact"),
            l0_pointer=frag.l0_pointer(),
            tokens=estimate_tokens((d.get("l3_label", "") or "")),
        )
        l3.engine_meta = {"distiller": "llm", "model": self.model, "seconds": round(dt, 2)}
        return l1, l2, l3


class _KeywordFallback(Distiller):
    """LLMDistiller 的内部兜底：按源路由到正确的 model-free 蒸馏器。

    - 代码源 -> CodeDistiller（AST，merged_with_l0=True）
    - 文本源 -> KeywordDistiller
    """

    def distill(self, frag: Fragment):
        if frag.source is Source.CODE:
            from .distill import CodeDistiller

            return CodeDistiller().distill(frag)
        from .distill import KeywordDistiller

        return KeywordDistiller().distill(frag)


def _strip_mermaid_fence(text: str) -> str:
    m = re.search(r"```(?:mermaid)?\s*(.*?)```", text or "", re.S)
    return (m.group(1) if m else text).strip()
