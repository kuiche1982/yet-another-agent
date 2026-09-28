"""contextmgr —— 类型定义（L0–L3 分级记忆 / ContextManager 核心数据结构）

命名边界（重要）：
- 本模块的 L0/L1/L2/L3 指「记忆物料分级」（raw → 结构化 → 可视化 → 索引），
  与 `docs/harness_design_map.md` 的「认知三层 L1/L2/L3」（设计原则 / 脑图 / 代码真相）
  **是不同轴**，切勿混用。本模块一律用 Source/L0Source/L1Structured/L2Visual/L3Index 显式命名。
- L0 是唯一 Truth of Source；L1/L2/L3 都是从 L0 派生的视图，可丢失、可重建、可渲染失败。
- L3 索引强制携带回 L0 的精确指针 (fid, offset, length)，保证「沿线索向下搜索」可达原文片段。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Source(str, enum.Enum):
    """物料来源。三源统一进同一套 L0–L3 分级与检索。"""

    LIBRARY = "library"   # 导入文档 / 资料库（L0: Source=Library）
    SESSION = "session"   # 当前会话（高优先级 L0: Source=Session）
    CODE = "code"         # 代码（L0: Source=Code，脑图作 Code.L3）


# 优先级：会话 > 资料库 > 代码（同分数下先保会话，再资料库，最后代码）
SOURCE_PRIORITY: dict[Source, int] = {
    Source.SESSION: 3,
    Source.LIBRARY: 2,
    Source.CODE: 1,
}


@dataclass
class Fragment:
    """L0 —— Truth of Source。

    一个 Fragment 就是一段「原始真相」。distill/index/retrieve 最终都回落到 Fragment.text。
    origin 记录它在原始文档里的位置，仅用于可回溯，不参与检索逻辑。
    """

    fid: str
    source: Source
    text: str
    priority: int = 0          # 同 source 内的相对重要性；会话片段默认高
    tokens: int = 0            # token 估算（由 tokenize.estimate_tokens 填）
    origin: str = ""           # 原始文档/文件标识（doc_id / file path 相对名）
    offset: int = 0            # 在原始文档中的字符偏移
    length: int = 0            # 片段长度（字符）
    lineno: int = 0            # 起始行号（1-based；Code 必填，Library 由 ingest_library 记录，Session=0）
    end_lineno: int = 0        # 结束行号（含）；同 lineno 语义
    score: float = 0.0         # 检索期瞬态：与命中 L3Index.score 同源的 BM25 分（不落盘、非持久语义）

    def l0_pointer(self) -> tuple[str, int, int]:
        """回 L0 的精确指针：供 L3 索引引用。"""
        return (self.fid, self.offset, self.length or len(self.text))


@dataclass
class L1Structured:
    """L1 —— 结构化蒸馏（JSON 友好，可机读、可检索）。

    Code 源特例：代码本身即结构，L0 与 L1 合并（merged_with_l0=True），
    不再做文本蒸馏，L2 脑图由 AST 直接生成。
    """

    fid: str
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)  # Code：本 def 内调用的 ast.Name 列表（去重，用于 L2 模块调用图）
    merged_with_l0: bool = False   # 代码源置 True（L0==L1）


@dataclass
class L2Visual:
    """L2 —— mermaid 可视化（agent 自导航用，渲染失败不丢信息）。

    图里**禁止写绝对路径**（守红线：避免模型被诱导 `cd /workspace` 逃逸），
    只画相对结构。
    """

    fid: str
    diagram: str = ""          # mermaid 源码
    tokens: int = 0


@dataclass
class L3Index:
    """L3 —— 一行索引 + 回 L0 精确指针。

    检索入口：query 先匹配 L3.label / L1.key_points，命中后沿 l0_pointer 回落 L0。
    Session.L3 + Library.L3 构成最简略资料分级，可向下搜索。
    """

    fid: str
    label: str                              # 一行标签（短、高信噪）
    l0_pointer: tuple[str, int, int] = ("", 0, 0)  # (fid, offset, length)
    source: Source = Source.LIBRARY
    priority: int = 0
    tokens: int = 0
    category: str = "fact"                  # concept/howto/api/bug/design/fact（L3 检索分类）
    score: float = 0.0                      # 检索时填，非持久语义
    engine_meta: dict = field(default_factory=dict)  # 蒸馏后端元信息（model/seconds 等）
