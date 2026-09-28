"""swe_agent/contracts.py —— 三层 loop 的层间契约（数据形状）。

L1（模型+工具+结论）↔ L2（harness+fences+hooks）↔ L3（supervisor 目标检测）
之间的接口必须显式、稳定。任何一层只消费契约，不读另一层的内部状态。

这是「补充①：层间契约显式化」的落地——契约定义一次，各层 prompt 只描述自己那段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Action:
    """L1→L2：模型产出的一个结构化动作。"""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        if not isinstance(d, dict):
            return cls(name="", params={})
        name = d.get("action", "")
        params = {k: v for k, v in d.items() if k != "action"}
        return cls(name=name or "", params=params)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.name, **self.params}


@dataclass
class Observation:
    """L2→L1/L3：一次动作执行后的观察。"""

    result: str
    ok: bool = True

    @property
    def error(self) -> Optional[str]:
        return None if self.ok else self.result


@dataclass
class Subtask:
    """L3→L1：supervisor 派发的子任务。"""

    desc: str
    deliverables: List[str] = field(default_factory=list)

    @classmethod
    def from_task(cls, t: Optional[Dict[str, Any]]) -> "Subtask":
        if not t:
            return cls(desc="")
        return cls(desc=t.get("desc", ""), deliverables=list(t.get("deliverables") or []))


@dataclass
class GoalState:
    """L3：目标达成判定（单杠校验）。"""

    status: str = "no_tests"  # pass | fail | no_tests
    detail: str = ""
    output: str = ""


@dataclass
class LayerEvent:
    """跨层遥测事件（补充⑥）。"""

    layer: str  # L1 | L2 | L3
    kind: str
    detail: str = ""
    ts: float = 0.0
