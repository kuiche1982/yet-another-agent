#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""websearch 插件的工具定义：封装 Zhipu Web Search API 的 web_search 动作。

作为插件自带 tools/ 模块，被 load_plugins 扫描导入时通过 @tool 装饰器
向 ToolRegistry 注册 web_search 动作（即「加载插件 = 注册工具」）。

后端能力：
- zhipu_web_search：Zhipu 网络搜索（意图增强 / 多引擎 / 结构化结果）
- _web_search_duckduckgo：DuckDuckGo 轻量搜索，作为 Zhipu 无 token / 配额耗尽 / 出错时的回退

参考文档：https://docs.bigmodel.cn/api-reference/工具-api/网络搜索
端点：    POST {GLM_BASE_URL}/web_search
鉴权：    Authorization: Bearer <GLM_API_TOKEN>
"""

import re
import urllib.parse

import requests

from swe_agent.config import (
    GLM_API_TOKEN,
    GLM_BASE_URL,
    WEB_SEARCH_ENGINE,
    WEB_SEARCH_COUNT,
    WEB_SEARCH_CONTENT_SIZE,
    WEB_SEARCH_RECENCY,
    WEB_SEARCH_INTENT,
    WEB_SEARCH_TIMEOUT,
    WEB_SEARCH_FALLBACK,
)
from swe_agent.registry import tool

# search_query 上限（接口约束）；超过则截断，避免 400。
_QUERY_MAX_LEN = 70

# 合法枚举（仅用于校验与友好报错）。
_ENGINES = {"search_std", "search_pro", "search_pro_sogou", "search_pro_quark"}
_RECENCY = {"oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"}
_CONTENT_SIZE = {"medium", "high"}


def _norm_engine(v):
    v = (v or WEB_SEARCH_ENGINE).strip()
    return v if v in _ENGINES else "search_std"


def _norm_recency(v):
    v = (v or WEB_SEARCH_RECENCY).strip()
    return v if v in _RECENCY else "noLimit"


def _norm_content_size(v):
    v = (v or WEB_SEARCH_CONTENT_SIZE).strip()
    return v if v in _CONTENT_SIZE else "medium"


def _norm_count(v):
    try:
        c = int(v if v is not None else WEB_SEARCH_COUNT)
    except (TypeError, ValueError):
        c = WEB_SEARCH_COUNT
    if c < 1:
        c = 1
    if c > 50:
        c = 50
    return c


def zhipu_web_search(
    query: str,
    *,
    engine: str = None,
    count: int = None,
    recency: str = None,
    content_size: str = None,
    intent: bool = None,
    domain_filter: str = None,
) -> str:
    """调用 Zhipu Web Search API，返回紧凑可读的搜索结果文本。

    成功 -> 多行「[序号] 标题 / 链接 / 摘要 / 来源 / 发布时间」文本；
    失败/未配置/配额耗尽 -> 以 "web_search_error:" 开头的错误串（便于上层回退）。
    """
    if not GLM_API_TOKEN:
        return "web_search_error: 未配置 GLM_API_TOKEN，无法使用 Zhipu 网络搜索"

    q = (query or "").strip()
    if not q:
        return "web_search_error: query 为空"
    if len(q) > _QUERY_MAX_LEN:
        q = q[:_QUERY_MAX_LEN]

    payload = {
        "search_query": q,
        "search_engine": _norm_engine(engine),
        "search_intent": bool(WEB_SEARCH_INTENT if intent is None else intent),
        "count": _norm_count(count),
        "search_recency_filter": _norm_recency(recency),
        "content_size": _norm_content_size(content_size),
    }
    if domain_filter:
        payload["search_domain_filter"] = domain_filter

    url = f"{GLM_BASE_URL}/web_search"
    headers = {
        "Authorization": f"Bearer {GLM_API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=WEB_SEARCH_TIMEOUT)
    except Exception as e:  # 网络层异常（含连接超时）
        return f"web_search_error: 请求 Zhipu 失败（{e}）"

    # 配额耗尽 / 鉴权失败 / 限流通常表现为非 200（401/429/403 等），统一回退
    if resp.status_code != 200:
        try:
            err = resp.json().get("error", {})
            code = err.get("code", "")
            msg = err.get("message", resp.text[:200])
            return f"web_search_error: HTTP {resp.status_code} [{code}] {msg}"
        except Exception:
            return f"web_search_error: HTTP {resp.status_code} {resp.text[:200]}"

    try:
        data = resp.json()
    except Exception as e:
        return f"web_search_error: 响应 JSON 解析失败（{e}）"

    results = data.get("search_result") or []
    if not results:
        return "web_search: 无结果（可能查询过短/被拦截，或搜索引擎暂无可用数据）"

    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        link = (r.get("link") or "").strip()
        content = (r.get("content") or "").strip()
        media = (r.get("media") or "").strip()
        pub = (r.get("publish_date") or "").strip()
        block = f"[{i}] {title}"
        if media:
            block += f"  （来源：{media}）"
        if link:
            block += f"\n    {link}"
        if content:
            block += f"\n    {content}"
        if pub:
            block += f"\n    发布时间：{pub}"
        lines.append(block)

    header = f"# Zhipu 网络搜索：{q}（共 {len(results)} 条）"
    return header + "\n" + "\n\n".join(lines)


def _web_search_duckduckgo(query: str) -> str:
    """DuckDuckGo HTML 轻量搜索（best-effort，作为 Zhipu 的离线/无 token/配额耗尽回退）。"""
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        blocks = re.findall(
            r'class="result__a"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
            r.text, re.DOTALL,
        )
        out = []
        for title, snippet in blocks[:6]:
            t = re.sub(r"<[^>]+>", "", title).strip()
            s = re.sub(r"<[^>]+>", "", snippet).strip()
            if t:
                out.append(f"- {t}\n  {s}")
        if not out:
            return "web_search: 无结果或解析失败（可能需联网/被拦截）"
        return "\n".join(out)
    except Exception as e:
        return f"web_search_error: {e}"


@tool(
    name="web_search",
    description="联网检索（优先 Zhipu 网络搜索 API：意图增强/多引擎，返回标题、链接、摘要与来源；"
                "未配置 GLM_API_TOKEN、配额耗尽或 Zhipu 出错时自动回退 DuckDuckGo 轻量搜索）。",
    category="web",
    schema={"type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]},
    dangerous=False,
    examples=[
        '{"action":"web_search","query":"python pathlib relative_to example"}',
    ],
    when_to_use="需要联网检索某个技术问题的解法或资料时使用。Zhipu 配额不足时自动改用 DuckDuckGo，无需手动切换。",
)
def web_search(query: str) -> str:
    """联网搜索：优先 Zhipu Web Search API，失败/未配置/配额耗尽时回退 DuckDuckGo。"""
    if not query:
        return "web_search_error: query 为空"
    # 优先 Zhipu 网络搜索（意图增强 + 结构化结果）
    try:
        out = zhipu_web_search(query)
        if not out.startswith("web_search_error"):
            return out
        # Zhipu 返回错误（含配额耗尽/鉴权失败/限流）→ 按开关回退 DuckDuckGo
        if WEB_SEARCH_FALLBACK:
            print(f"[web_search] Zhipu 不可用（{out}），回退 DuckDuckGo")
            return _web_search_duckduckgo(query)
        return out
    except Exception as e:
        if WEB_SEARCH_FALLBACK:
            print(f"[web_search] Zhipu 异常（{e}），回退 DuckDuckGo")
            return _web_search_duckduckgo(query)
        return f"web_search_error: {e}"
