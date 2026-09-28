#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""web_fetch 插件的工具定义：封装 Zhipu 网页阅读（Web Page Reading）API。

作为插件自带 tools/ 模块，被 load_plugins 扫描导入时通过 @tool 装饰器
向 ToolRegistry 注册 web_fetch 动作（即「加载插件 = 注册工具」）。

后端能力：
- zhipu_reader：Zhipu 网页阅读 API（POST {GLM_BASE_URL}/reader），把 URL 解析为 markdown/text；
- _fallback_http_fetch：Zhipu 无 token / 配额耗尽 / 出错时的轻量兜底（直接 HTTP GET + 去标签）。

参考文档：https://docs.bigmodel.cn/api-reference/工具-api/网页阅读
端点：    POST {GLM_BASE_URL}/reader
鉴权：    Authorization: Bearer <GLM_API_TOKEN>
"""

import re

import requests

from swe_agent.config import GLM_API_TOKEN, GLM_BASE_URL
from swe_agent.registry import tool

# 单次返回正文截断上限（避免撑爆上下文）
_CONTENT_MAX = 12000


def zhipu_reader(
    url: str,
    *,
    return_format: str = "markdown",
    timeout: int = 20,
    no_cache: bool = False,
    retain_images: bool = True,
    with_images_summary: bool = False,
    with_links_summary: bool = False,
) -> str:
    """调用 Zhipu 网页阅读 API，返回紧凑可读的网页正文。

    成功 -> 「# 标题 / 来源 / 描述 + 正文」文本；
    失败/未配置/配额耗尽 -> 以 "web_fetch_error:" 开头的错误串（便于上层回退）。
    """
    if not GLM_API_TOKEN:
        return "web_fetch_error: 未配置 GLM_API_TOKEN，无法使用 Zhipu 网页阅读"

    q = (url or "").strip()
    if not q:
        return "web_fetch_error: url 为空"

    try:
        timeout = int(timeout or 20)
    except (TypeError, ValueError):
        timeout = 20

    payload = {
        "url": q,
        "return_format": (return_format or "markdown").strip(),
        "timeout": timeout,
        "no_cache": bool(no_cache),
        "retain_images": bool(retain_images),
        "no_gfm": False,
        "keep_img_data_url": False,
        "with_images_summary": bool(with_images_summary),
        "with_links_summary": bool(with_links_summary),
    }

    api = f"{GLM_BASE_URL}/reader"
    headers = {
        "Authorization": f"Bearer {GLM_API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(api, json=payload, headers=headers, timeout=min(timeout + 10, 60))
    except Exception as e:  # 网络层异常（含连接超时）
        return f"web_fetch_error: 请求 Zhipu 失败（{e}）"

    if resp.status_code != 200:
        try:
            err = resp.json().get("error", {})
            code = err.get("code", "")
            msg = err.get("message", resp.text[:200])
            return f"web_fetch_error: HTTP {resp.status_code} [{code}] {msg}"
        except Exception:
            return f"web_fetch_error: HTTP {resp.status_code} {resp.text[:200]}"

    try:
        data = resp.json()
    except Exception as e:
        return f"web_fetch_error: 响应 JSON 解析失败（{e}）"

    rr = data.get("reader_result") or {}
    content = (rr.get("content") or "").strip()
    title = (rr.get("title") or "").strip()
    desc = (rr.get("description") or "").strip()
    src = (rr.get("url") or q).strip()

    if not content:
        return "web_fetch: 无内容（页面可能为空、需登录，或被反爬拦截）"

    capped = len(content) > _CONTENT_MAX
    body = content[:_CONTENT_MAX] + ("\n…（正文已截断至 12000 字）" if capped else "")
    header = f"# 网页阅读：{title or src}\n来源：{src}"
    if desc:
        header += f"\n描述：{desc}"
    return header + "\n\n" + body


def _fallback_http_fetch(url: str) -> str:
    """Zhipu 不可用时的轻量兜底：直接 GET + 去标签抽取正文（best-effort，结果较粗糙）。"""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        return f"web_fetch_error: 回退抓取失败（{e}）"

    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    # 去掉 script/style，再去所有标签，折叠空白
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "web_fetch: 回退抓取无可用文本（页面可能需 JS 渲染）"
    capped = len(cleaned) > _CONTENT_MAX
    body = cleaned[:_CONTENT_MAX] + ("\n…（正文已截断至 12000 字）" if capped else "")
    header = f"# 网页抓取（直接 HTTP 兜底）：{title or url}\n来源：{url}"
    return header + "\n\n" + body


@tool(
    name="web_fetch",
    description="抓取并阅读网页正文（优先 Zhipu 网页阅读 API：把 URL 解析为 markdown/text，"
                "支持 return_format/timeout/retain_images 等；Zhipu 未配置、配额耗尽或出错时自动回退直接抓取）。",
    category="web",
    schema={"type": "object",
            "properties": {
                "url": {"type": "string"},
                "return_format": {"type": "string"},
                "timeout": {"type": "integer"},
                "no_cache": {"type": "boolean"},
                "retain_images": {"type": "boolean"},
                "with_images_summary": {"type": "boolean"},
                "with_links_summary": {"type": "boolean"},
            },
            "required": ["url"]},
    dangerous=False,
    examples=[
        '{"action":"web_fetch","url":"https://docs.bigmodel.cn/api-reference/工具-api/网页阅读"}',
    ],
    when_to_use="需要按 URL 抓取并阅读某个具体网页/文档的正文时使用（区别于泛搜的 web_search）。",
)
def web_fetch(
    url: str,
    return_format: str = "markdown",
    timeout: int = 20,
    no_cache: bool = False,
    retain_images: bool = True,
    with_images_summary: bool = False,
    with_links_summary: bool = False,
    **kwargs,
) -> str:
    """抓取并阅读网页：优先 Zhipu 网页阅读 API，失败/未配置/配额耗尽时回退直接抓取。"""
    if not url:
        return "web_fetch_error: url 为空"
    out = zhipu_reader(
        url,
        return_format=return_format,
        timeout=timeout,
        no_cache=no_cache,
        retain_images=retain_images,
        with_images_summary=with_images_summary,
        with_links_summary=with_links_summary,
    )
    if not out.startswith("web_fetch_error"):
        return out
    # Zhipu 返回错误（含配额耗尽/鉴权失败/限流/网络异常）→ 回退直接抓取
    print(f"[web_fetch] Zhipu 不可用（{out}），回退直接抓取")
    return _fallback_http_fetch(url)
