"""swe_agent/layers.py —— 跨层遥测。

历史背景：本文件曾把 run_agent 的 PDCA 拆成扁平的 L1/L2/L3 三个函数
（run_l1_propose / run_l2_fence / run_l3_sense）。在吸收 forge 教训后，run_agent 已
重构为**嵌套三层循环**（supervisor._run_nested_loops → build_executor_agent；executor 单步
循环逻辑收敛进 agent._apply_toolcall）。那三个函数已彻底弃用并删除——循环防护（fences）与
对话 buffer 现统一由 agent.RunState.cm（ContextManager）持有，不再散落于此。

因此本文件现仅保留**跨层遥测** helper（emit_event / layer_events / reset_layer_events），
供 supervisor 在嵌套循环边界打点。
"""

import time
from typing import List

from .contracts import LayerEvent
from . import state as _state


# ---- 跨层遥测（补充⑥）----
_LAYER_EVENTS: List[LayerEvent] = []


def emit_event(layer: str, kind: str, detail: str = "") -> None:
    """吐一条带 layer 字段的结构化事件，便于定位『哪层挂了』。"""
    ev = LayerEvent(layer=layer, kind=kind, detail=detail, ts=time.time())
    _LAYER_EVENTS.append(ev)
    _state._telemetry(f"{layer}_{kind}")  # 兼容旧单标签遥测


def layer_events() -> List[LayerEvent]:
    return list(_LAYER_EVENTS)


def reset_layer_events() -> None:
    _LAYER_EVENTS.clear()
