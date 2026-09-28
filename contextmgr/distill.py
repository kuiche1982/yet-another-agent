"""contextmgr —— 蒸馏（L0 → L1 / L2 / L3）。

Distiller 是可插拔接口：
- KeywordDistiller：model-free 确定性启发式，用于单测与默认降级。
- CodeDistiller：用 stdlib `ast` 静态分析生成 Code.L1/L2/L3（model-free），
  且不写绝对路径，守红线。
- 生产可加 LLMDistiller（弱模型产出结构化蒸馏），见 `contextmgr/llm_backends.py`（可选插件，
  默认不启用，contextmgr 单测全 model-free）。

原则：L1/L2/L3 都是从 L0 派生的视图，可丢失/重建；只有 L0 是 Truth。
"""

from __future__ import annotations

import ast
import re
from abc import ABC, abstractmethod

from .tokenize import estimate_tokens
from .types import Fragment, L1Structured, L2Visual, L3Index, Source


def _sanitize_label(s: str, limit: int = 40) -> str:
    """清洗成 mermaid 安全标签：去引号/换行/控制符，截断。"""
    s = s.replace('"', "'").replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:limit]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s*|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


class Distiller(ABC):
    @abstractmethod
    def distill(self, frag: Fragment) -> tuple[L1Structured, L2Visual, L3Index]:
        ...


class KeywordDistiller(Distiller):
    """通用文本蒸馏：取首句作摘要、关键句作 key_points、大写词作实体。

    model-free、确定性；不追求聪明，只保证管道可跑 + 可单测。
    """

    def distill(self, frag: Fragment) -> tuple[L1Structured, L2Visual, L3Index]:
        sents = _split_sentences(frag.text)
        summary = sents[0] if sents else frag.text[:80]
        # key_points：首句 + 含高信号词的句子（数字/术语/动词性）
        heat = re.compile(r"\d|函数|类|接口|配置|模型|错误|必须|关键|返回|调用|依赖")
        kps: list[str] = []
        for s in sents[:8]:
            if s == summary:
                kps.insert(0, summary)
            elif heat.search(s):
                kps.append(s)
        if not kps:
            kps = sents[:3] or [frag.text[:80]]
        entities = sorted({w for w in re.findall(r"[A-Z][A-Za-z0-9_]+", frag.text)})[:12]

        l1 = L1Structured(
            fid=frag.fid, summary=summary, key_points=kps, entities=entities
        )
        nodes = "\n".join(f'  n{i}["{_sanitize_label(k, 36)}"]' for i, k in enumerate(kps[:8]))
        diagram = f"graph TD\n  root[(\"L0 {_sanitize_label(frag.origin or frag.fid, 20)}\")]\n{nodes}"
        l2 = L2Visual(fid=frag.fid, diagram=diagram)
        label = _sanitize_label((kps[0] if kps else frag.text), 60)
        l3 = L3Index(
            fid=frag.fid,
            label=label,
            l0_pointer=frag.l0_pointer(),
            source=frag.source,
            priority=frag.priority,
            tokens=frag.tokens,
        )
        return l1, l2, l3


class CodeDistiller(Distiller):
    """代码蒸馏：stdlib ast 静态分析（model-free）。

    - L1：签名 + 文件:行号 + docstring 作 key_points；同步抽取 ast.Call 调用名到 `calls`
      （供 L2_overview 模块级调用图聚合，跨片段 def→def 边用）
    - L2：单片段结构脑图（graph TD），节点标注「模块 → 本 def → 本 def 内 ast.Name 调用」；
      **只画相对结构，不写绝对路径**
    - L3：一行索引标签（含文件:行号）
    适用于「整文件片段」；更细粒度由 `split_code_fragments` 先切成每定义一个 L0 片段。
    """

    def distill(self, frag: Fragment) -> tuple[L1Structured, L2Visual, L3Index]:
        try:
            tree = ast.parse(frag.text)
        except SyntaxError:
            # 非合法 Python（片段/注释）：退化为 keyword 蒸馏，避免崩
            return KeywordDistiller().distill(frag)

        defs: list[tuple[str, str]] = []   # (kind, name)  e.g. ("def", "foo")
        methods: dict[str, list[str]] = {}
        lines: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = _func_sig(node)
                defs.append(("def", node.name))
                doc = ast.get_docstring(node)
                lines.append(_format_kp(sig, doc, frag))
            elif isinstance(node, ast.ClassDef):
                sig = f"class {node.name}"
                members = [
                    _func_sig(m)
                    for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                methods[node.name] = members
                defs.append(("class", node.name))
                doc = ast.get_docstring(node)
                lines.append(_format_kp(sig, doc, frag))

        # L1.calls：本片段内 ast.Call 调用的 ast.Name 集合（去重，保持顺序）
        calls = _collect_callees(tree)

        name = frag.origin or frag.fid
        l1 = L1Structured(
            fid=frag.fid,
            summary=f"模块/片段 {name}：{len(defs)} 个顶层定义（L0==L1，代码即结构）",
            key_points=lines or [frag.text[:80]],
            entities=[n for _, n in defs],
            calls=calls,
            merged_with_l0=True,   # 代码源：L0 与 L1 合并，不再冗余文本蒸馏
        )
        # L2 脑图：M[name] -> D0..Dk；D0..Dk ->|"calls"| C0..Cm（本片段内 ast.Name 调用）
        edges = [f'  M["{_sanitize_label(name, 24)}"] --> D{i}["{_sanitize_label(_kind_label(k, n), 28)}"]'
                 for i, (k, n) in enumerate(defs[:10])]
        call_edges = [f'  D0 -->|"calls"| C{k}["{_sanitize_label(c, 20)}"]'
                      for k, c in enumerate(calls[:6])] if defs else []
        body = "\n".join(edges + call_edges)
        diagram = f"graph TD\n{body}" if body else "graph TD\n  M[empty]"
        l2 = L2Visual(fid=frag.fid, diagram=diagram)
        # L3 label 带文件:行号（满足「结构描述 + 位置信息」）
        loc = f"{name}:{frag.lineno}" if frag.lineno else name
        label = f"{loc} ({len(defs)} defs" + (f"; {', '.join(list(methods)[:3])})" if methods else ")")
        l3 = L3Index(
            fid=frag.fid,
            label=_sanitize_label(label, 60),
            l0_pointer=frag.l0_pointer(),
            source=frag.source,
            priority=frag.priority,
            tokens=frag.tokens,
        )
        return l1, l2, l3


def _func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in node.args.args]
    return f"def {node.name}({', '.join(args)})"


def _kind_label(kind: str, name: str) -> str:
    """L2 mermaid 节点文本：def -> 'def foo()', class -> 'class Foo'。"""
    if kind == "class":
        return f"class {name}"
    # def: 加 () 以与被调对象视觉区分
    return f"def {name}()"


def _format_kp(sig: str, doc: str | None, frag: Fragment) -> str:
    """L1 key_point 一行：「sig — doc首行  @ origin:lineno-end_lineno」。

    lineno=0 表示片段没有行号信息（罕见，如 Session/Library），省略位置段。
    """
    loc = f" @ {frag.origin}:{frag.lineno}-{frag.end_lineno}" if frag.lineno else ""
    head = sig + (f" — {doc.splitlines()[0]}" if doc else "")
    return head + loc


def _collect_callees(tree: ast.AST) -> list[str]:
    """本片段内 ast.Call 调用的 ast.Name 集合（去重，保持首次出现顺序）。

    仅采集顶级 ast.Name（不含 ast.Attribute 的 self.foo 等实例方法调用，避免噪声）；
    这些 callee 可能是同模块其他 def，也可能是外部函数 —— 由 persist.write_l2_overview
    按 L1.calls ↔ 跨片段 L1.entities 对齐画出 def→def 边。
    """
    seen: set[str] = set()
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            n = node.func.id
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def split_code_fragments(file_text: str, name: str, fid_prefix: str,
                         provider: "SymbolProvider | None" = None,
                         path: str | None = None) -> list[Fragment]:
    """把一份代码切成「每顶层定义一个 L0 片段」（Source=CODE），支持精确 descend 检索。

    不写绝对路径，origin 用相对文件名。offset/length 用字符位移，供 L3 回指针。
    lineno/end_lineno 从符号表填入，供 L1 携带文件:行号信息。

    provider：符号提供者，默认 `ASTSymbolProvider`（纯 stdlib，只支持 Python）。
    传 `LspSymbolProvider` 可走 LSP `documentSymbol`，支持多语言且行号更准
    （含装饰器）。provider 切不出符号（非 Python / 服务器不可用 / 无顶层定义）
    时统一退化为「整文件一个片段」。
    """
    from .symbols import ASTSymbolProvider, SymbolProvider  # 延迟导入避免循环

    prov: SymbolProvider = provider or ASTSymbolProvider()
    try:
        syms = prov.symbols(file_text, name, path=path)
    except Exception:
        syms = None
    if not syms:
        return [Fragment(fid=f"{fid_prefix}#whole", source=Source.CODE, text=file_text,
                         origin=name, offset=0, length=len(file_text),
                         lineno=1, end_lineno=file_text.count("\n") + 1,
                         tokens=estimate_tokens(file_text), priority=2)]
    out: list[Fragment] = []
    src_lines = file_text.splitlines(keepends=True)
    n_lines = len(src_lines)
    for sym in syms:
        start = max(0, sym.lineno - 1)
        end = min(n_lines, max(sym.end_lineno, sym.lineno))
        if start >= end:
            continue
        snippet = "".join(src_lines[start:end])
        offset = sum(len(l) for l in src_lines[:start])
        fid = f"{fid_prefix}#{sym.name}"
        out.append(Fragment(
            fid=fid, source=Source.CODE, text=snippet,
            origin=name, offset=offset, length=len(snippet),
            lineno=sym.lineno, end_lineno=end,
            tokens=estimate_tokens(snippet), priority=2,
        ))
    if not out:  # 符号都在边界外：整文件作一个片段
        out.append(Fragment(fid=f"{fid_prefix}#whole", source=Source.CODE, text=file_text,
                            origin=name, offset=0, length=len(file_text),
                            lineno=1, end_lineno=file_text.count("\n") + 1,
                            tokens=estimate_tokens(file_text), priority=2))
    return out
