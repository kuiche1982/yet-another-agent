#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ocr 插件的工具定义：封装 Zhipu OCR 服务（图片文字识别）。

作为插件自带 tools/ 模块，被 load_plugins 扫描导入时通过 @tool 装饰器
向 ToolRegistry 注册 ocr 动作（即「加载插件 = 注册工具」）。

后端能力：
- zhipu_ocr：Zhipu OCR 服务（POST {GLM_BASE_URL}/files/ocr），多部件上传图片，
  支持印刷体/手写体、20+ 语言，返回每行文字 + 位置 + 置信度。

参考文档：https://docs.bigmodel.cn/cn/guide/tools/zhipu-ocr
端点：    POST {GLM_BASE_URL}/files/ocr
鉴权：    Authorization: Bearer <GLM_API_TOKEN>
文件限制：PNG/JPG/JPEG/BMP，≤8M
"""

import mimetypes
from pathlib import Path

import requests

from swe_agent.config import GLM_API_TOKEN, GLM_BASE_URL
from swe_agent.registry import tool

# 合法语言类型（tool_type 固定为 hand_write）
_LANG_DEFAULT = "AUTO"
_LANGS = {
    "AUTO", "CHN_ENG", "ENG", "JAP", "KOR", "FRE", "SPA", "POR", "GER", "ITA",
    "RUS", "DAN", "DUT", "MAL", "SWE", "IND", "POL", "ROM", "TUR", "GRE",
    "HUN", "THA", "VIE", "ARA", "HIN",
}


def _norm_lang(v):
    v = (v or _LANG_DEFAULT).strip().upper()
    return v if v in _LANGS else _LANG_DEFAULT


def zhipu_ocr(image, *, language_type: str = "AUTO", probability: bool = False) -> str:
    """调用 Zhipu OCR 服务识别图片文字。

    image: 本地图片路径（str/Path）或图片二进制 bytes。
    成功 -> 多行「[序号] 文字 (位置) [置信度]」文本；
    失败/未配置/配额耗尽 -> 以 "ocr_error:" 开头的错误串。
    """
    if not GLM_API_TOKEN:
        return "ocr_error: 未配置 GLM_API_TOKEN，无法使用 Zhipu OCR"

    # 解析图片来源：bytes 或 路径
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
        fname = "image.png"
        ctype = "image/png"
    else:
        p = Path(image)
        if not p.exists():
            return f"ocr_error: 文件不存在 {image}"
        data = p.read_bytes()
        fname = p.name
        ctype = mimetypes.guess_type(str(p))[0] or "image/png"

    api = f"{GLM_BASE_URL}/files/ocr"
    headers = {"Authorization": f"Bearer {GLM_API_TOKEN}"}
    files = {"file": (fname, data, ctype)}
    form = {"tool_type": "hand_write", "language_type": _norm_lang(language_type)}
    if probability:
        form["probability"] = "true"

    try:
        resp = requests.post(api, headers=headers, files=files, data=form, timeout=60)
    except Exception as e:  # 网络层异常
        return f"ocr_error: 请求 Zhipu 失败（{e}）"

    if resp.status_code != 200:
        try:
            err = resp.json().get("error", {})
            code = err.get("code", "")
            msg = err.get("message", resp.text[:200])
            return f"ocr_error: HTTP {resp.status_code} [{code}] {msg}"
        except Exception:
            return f"ocr_error: HTTP {resp.status_code} {resp.text[:200]}"

    try:
        data_json = resp.json()
    except Exception as e:
        return f"ocr_error: 响应 JSON 解析失败（{e}）"

    results = data_json.get("words_result") or []
    if not results:
        msg = data_json.get("message") or "无识别结果"
        return f"ocr_error: {msg}"

    lines = []
    for i, r in enumerate(results, 1):
        words = (r.get("words") or "").strip()
        loc = r.get("location") or {}
        prob = r.get("probability") or {}
        block = f"[{i}] {words}"
        if loc:
            block += (f"  (位置 x={loc.get('left')},y={loc.get('top')},"
                      f"w={loc.get('width')},h={loc.get('height')})")
        if prob:
            block += f"  [置信度 avg={prob.get('average')}, min={prob.get('min')}]"
        lines.append(block)

    header = f"# Zhipu OCR 识别（共 {len(results)} 行）"
    return header + "\n" + "\n".join(lines)


@tool(
    name="ocr",
    description="对图片做 OCR 文字识别（Zhipu OCR 服务：支持印刷体/手写体、20+ 语言，"
                "返回每行文字与位置/置信度）。输入本地图片路径（PNG/JPG/JPEG/BMP，≤8M）。",
    category="web",
    schema={"type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "language_type": {"type": "string"},
                "probability": {"type": "boolean"},
            },
            "required": ["image_path"]},
    dangerous=False,
    examples=[
        '{"action":"ocr","image_path":"/path/to/receipt.png"}',
    ],
    when_to_use="需要从图片/截图/扫描件中抽取文字时使用。",
)
def ocr(image_path: str, language_type: str = "AUTO", probability: bool = False, **kwargs) -> str:
    """对图片做 OCR：调用 Zhipu OCR 服务识别文字。"""
    if not image_path:
        return "ocr_error: image_path 为空"
    return zhipu_ocr(image_path, language_type=language_type, probability=probability)
