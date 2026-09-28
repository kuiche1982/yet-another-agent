#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/actions.py —— 动作解析与净化（标准 OpenAI toolcall 产物）

传输层（llm_glm / llm_lmstudio）走原生 function calling，已把模型输出规约为
合法 JSON 数组字符串（[ {action:...}, ... ]），本模块只做轻量解析与净化，
不再需要 plaintext 那套「三引号 / XML / 宽松正则」容错（那套已随 rapid-mlx 一并废弃）。
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .config import MAX_ACTIONS_PER_RESPONSE, STATS
from .state import _action_fp, _telemetry


def parse_actions(model_out: str) -> List[Dict[str, Any]]:
    """把原生 toolcall 返回的 JSON 数组字符串解析为动作对象列表。

    传输层已正确转义，正常情况下是干净的 ``[{"action": ...}, ...]``；这里只做
    最小容错（去 ```json 围栏、单对象自动包数组），不做 plaintext 那套正则兜底。
    """
    raw = (model_out or "").strip()
    if not raw:
        return []
    # 去掉可能的 ```json ... ``` 围栏
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    if not raw.startswith("["):
        raw = "[" + raw + "]"
    try:
        objs = json.loads(raw)
    except Exception:
        return []
    if isinstance(objs, dict):
        objs = [objs]
    return [o for o in objs if isinstance(o, dict)]


def _sanitize_response_actions(actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """净化单轮模型响应里的动作：

    1. 同轮去重（防刷屏）：完全相同指纹的动作只保留首次。专治 3B 模型一轮吐出数百个相同
       read_file 的失控生成——折叠后只剩 1 个，不再打满执行预算、也不拖垮模型服务。
    2. 突发封顶：去重后动作数仍超过 MAX_ACTIONS_PER_RESPONSE，则截断至上限（保留有产出的
       多动作输出，如「写实现 + 写测试」，但拦住极端刷屏）。
    3. 只读主导预警：整轮全是 read/glob/grep 且无写/执行/完成，提示动手写代码。

    返回 (净化后的动作列表, 警告文案或 None)。有产出的【不同】动作一律保留，绝不因「多动作」而误砍。
    """
    if not actions:
        return [], None
    warns: List[str] = []

    # 1) 同轮去重：按动作指纹折叠完全相同的重复项（只留首次）
    seen: Dict[str, int] = {}
    deduped: List[Dict[str, Any]] = []
    dup = 0
    for a in actions:
        fp = _action_fp(a)
        seen[fp] = seen.get(fp, 0) + 1
        if seen[fp] == 1:
            deduped.append(a)
        else:
            dup += 1
    if dup > 0:
        warns.append(f"已折叠 {dup} 个完全重复的动作（同指纹被反复下发，只执行一次）。")

    # 2) 突发封顶（去重后）
    if len(deduped) > MAX_ACTIONS_PER_RESPONSE:
        dropped = len(deduped) - MAX_ACTIONS_PER_RESPONSE
        deduped = deduped[:MAX_ACTIONS_PER_RESPONSE]
        warns.append(f"本轮动作数超过上限 {MAX_ACTIONS_PER_RESPONSE}，已截断至前 "
                     f"{MAX_ACTIONS_PER_RESPONSE} 个（剩余 {dropped} 个丢弃以防 token 打满）。")

    # 3) 只读主导预警
    READONLY = {"read_file", "grep", "glob"}
    if deduped and all(a.get("action") in READONLY for a in deduped):
        warns.append("⚠️ 你本轮只做了「只读」操作且高度重复，请立即停止只读循环、动手编写/修复代码："
                     "若工作区是空目录就直接用 write_file 创建所需文件（如实现代码与测试），"
                     "若已有代码就用 edit_file 精准修改，然后用 shell 跑 pytest 验证；"
                     "不要反复 read 同一个文件却不下笔。")

    if warns:
        _telemetry("read_loop_sanitized")
    return deduped, (" ".join(warns) if warns else None)
