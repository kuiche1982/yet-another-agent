"""contextmgr —— 代码符号切分（SymbolProvider 协议，双路）。

为什么要抽这一层
----------------
原 `split_code_fragments` 直接硬编码 `ast.parse`，有两个硬伤：

1. **只支持 Python**：非 Python（.go / .ts / .rs ...）解析即 SyntaxError，
   整文件退化成一个 L0 片段，连函数级切分都没有。
2. **行号/调用关系靠猜**：`ast.Name` 只能解同模块裸调用，`self.method()`
   与跨模块 import 解不出来（详见 docs/contextmgr_dev.md §6）。

LSP 的 `textDocument/documentSymbol` 是标准能力，语言无关，能同时解决这两点
（gopls 实测连 method 都给；pyright 还给 callHierarchy）。

双路设计（**LSP 不能是唯一路径**）
----------------------------------
- `ASTSymbolProvider`：纯 stdlib，永远可用，**默认**。contextmgr 单测铁律是
  model-free / tool-free，LSP 是外部进程（可能没装、可能慢），绝不能当唯一依赖。
- `LspSymbolProvider`：可选挂载。按文件扩展名选语言服务器；任何一步失败
  （没装 / 启动失败 / 握手超时 / 无该能力）都**静默返回 None**，调用方回落 AST。

contextmgr 不硬依赖 swe_agent：LSP 相关 import 全部在函数内延迟进行。
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, NamedTuple

# 自动安装目标 venv 的 pyright（用户明确要求「按目标代码的 venv 安装」）。
# 只在**已存在**的 venv 里装，绝不新建 venv（避免污染目标仓库）。
# 用 CONTEXTMGR_LSP_AUTOINSTALL=0 关闭。
AUTOINSTALL_ENV = "CONTEXTMGR_LSP_AUTOINSTALL"

_KIND_TO_KIND = {5: "class", 6: "def", 12: "def"}   # LSP SymbolKind -> 简化 kind


@dataclass
class CodeSymbol:
    """一个顶层代码符号（用于把文件切成 L0 片段）。"""

    name: str
    kind: str = "def"            # "def" | "class" | "other"
    lineno: int = 1              # 1-based，闭区间起点
    end_lineno: int = 1          # 1-based，闭区间终点（含）
    meta: dict[str, Any] = field(default_factory=dict)


class SymbolProvider(ABC):
    """符号提供者：把「一份代码」变成顶层符号列表。

    `symbols()` 返回 None 或空列表都表示「切不出来」——调用方退化为整文件一片段。
    """

    name = "base"

    @abstractmethod
    def symbols(self, text: str, name: str, path: str | None = None) -> list[CodeSymbol] | None:
        ...


class ASTSymbolProvider(SymbolProvider):
    """stdlib `ast` 实现（默认）。

    行为与原 `split_code_fragments` 的 AST 分支**逐字段一致**：只取顶层
    FunctionDef/AsyncFunctionDef/ClassDef，lineno 取 `def` 行（不含装饰器）。
    """

    name = "ast"

    def symbols(self, text: str, name: str, path: str | None = None) -> list[CodeSymbol] | None:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return None
        out: list[CodeSymbol] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "def"
            elif isinstance(node, ast.ClassDef):
                kind = "class"
            else:
                continue
            out.append(CodeSymbol(
                name=node.name, kind=kind,
                lineno=node.lineno, end_lineno=(node.end_lineno or node.lineno),
            ))
        return out


class LspHandle(NamedTuple):
    """一个已就绪的 LSP 连接。"""

    key: str        # 缓存键（服务器命令）
    client: Any     # swe_agent.lsp.LSPClient（延迟 import，这里不标类型）
    lang_id: str
    owned: bool     # True=本 provider 创建并负责关闭；False=复用全局 manager


class LspSymbolProvider(SymbolProvider):
    """LSP 实现（可选）：`textDocument/documentSymbol` → 顶层符号。

    - 按扩展名选语言服务器（复用 `swe_agent.lsp._EXT_TO_LANG`）；
    - **Python 优先用目标仓库 venv 里的 pyright**（pylsp 无 callHierarchy，
      且 `plugins/lsp@1.0.0/.lsp.json` 里 python 指向的是本仓库 `.venv/bin/pylsp`，
      分析别人的仓库时是错的）；
    - 任何失败都返回 None，由调用方回落 AST。

    client_factory 可注入，单测用假客户端（不起进程，守住 model-free/tool-free）。
    """

    name = "lsp"

    def __init__(self, client_factory=None, autoinstall: bool | None = None,
                 settle: bool = True, timeout: float = 20.0):
        self._factory = client_factory or lsp_client_for
        self.autoinstall = (_env_autoinstall() if autoinstall is None else autoinstall)
        self.settle = settle          # 是否等语义分析就绪（慢，但结果才对）
        self.timeout = timeout
        self._cache: dict[str, LspHandle] = {}

    def symbols(self, text: str, name: str, path: str | None = None) -> list[CodeSymbol] | None:
        if not path:
            return None      # LSP 必须有真实文件（didOpen 需要 uri）
        try:
            handle = self._handle_for(path)
            if handle is None:
                return None
            client = handle.client
            uri = client.open_document(path, handle.lang_id, text)
            if self.settle:
                client.wait_for_semantics(uri, timeout=self.timeout)
            syms = client.top_level_symbols(uri)
        except Exception:
            return None
        if not syms:
            return None
        out = [
            CodeSymbol(
                name=s.get("name", ""),
                kind=_KIND_TO_KIND.get(s.get("kind", 0), "other"),
                lineno=int(s.get("lineno", 1)),
                end_lineno=int(s.get("end_lineno") or s.get("lineno", 1)),
                meta={"source": "lsp", "kind_name": s.get("kind_name", "")},
            )
            for s in syms
            if s.get("name")
        ]
        return out or None

    def _handle_for(self, path: str) -> LspHandle | None:
        handle = self._factory(path, autoinstall=self.autoinstall)
        if handle is None:
            return None
        if handle.key not in self._cache:
            self._cache[handle.key] = handle
        return self._cache[handle.key]

    def close(self):
        for h in self._cache.values():
            if h.owned:
                try:
                    h.client.close()
                except Exception:
                    pass
        self._cache.clear()


# ======================================================================
# 目标仓库 venv 探测 + pyright 自动安装（延迟 import，保持零硬依赖）
# ======================================================================
def _env_autoinstall() -> bool:
    return os.environ.get(AUTOINSTALL_ENV, "1") not in ("0", "false", "False", "no")


def find_target_venv(path: str, max_depth: int = 6) -> str | None:
    """从文件所在目录向上找目标仓库的 venv，返回其 python 可执行路径。

    只认 `venv/` 与 `.venv/`（uv/pipenv/poetry 默认布局）。找不到返回 None。
    """
    d = os.path.dirname(os.path.abspath(path))
    for _ in range(max_depth):
        for cand in (".venv", "venv"):
            p = os.path.join(d, cand, "bin", "python")
            if os.path.isfile(p):
                return p
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def ensure_pyright(venv_python: str, autoinstall: bool = True,
                   timeout: int = 300) -> str | None:
    """返回目标 venv 里 `pyright-langserver` 的路径；找不到就返回 None（**不加载**）。

    - 已有 → 直接返回；
    - 没有且 autoinstall 且有 uv → 试 `uv pip install --python <venv_py> pyright`；
    - 仍缺失（autoinstall 关闭 / 无 uv / 无网 / 超时 / 装完还是没有）→ 返回 None。

    找不到就不加载，由调用方 / lsp 端口返回「pyright 未安装 / lsp 服务未启动」
    状态，而不是抛异常强迫安装。
    """
    bindir = os.path.dirname(venv_python)
    srv = os.path.join(bindir, "pyright-langserver")
    if os.path.isfile(srv):
        return srv
    if not autoinstall or not shutil.which("uv"):
        return None
    try:
        subprocess.run(
            ["uv", "pip", "install", "--python", venv_python, "pyright"],
            timeout=timeout, capture_output=True, check=False,
        )
    except Exception:
        return None
    return srv if os.path.isfile(srv) else None


def pyright_status(venv_python: str | None = None) -> str:
    """给 lsp 端口用的状态文案：pyright 是否可用。

    - 没传 venv（或找不到）→ "LSP 服务未启动"
    - 传了 venv 但 pyright 缺失 → "pyright 未安装（目标 venv 缺少 pyright-langserver）"
    - 可用 → "ok"
    """
    if not venv_python:
        return "LSP 服务未启动"
    if ensure_pyright(venv_python) is None:
        return f"pyright 未安装（目标 venv 缺少 pyright-langserver：{venv_python}）"
    return "ok"


def lsp_client_for(path: str, autoinstall: bool = True) -> LspHandle | None:
    """默认 client_factory：为 path 选一个语言服务器并返回已握手的连接。

    Python 优先目标 venv 的 pyright（并 push pythonPath 指向该 venv 的解释器，
    否则 pyright 不做语义分析，references/callHierarchy 全空）；
    其他语言走 `swe_agent.lsp` 的插件/环境变量解析。
    """
    try:
        from swe_agent import lsp as _lsp
    except Exception:
        return None
    ext = os.path.splitext(path)[1].lower()
    lang_id = _lsp._EXT_TO_LANG.get(ext)
    if not lang_id:
        return None

    if lang_id == "python":
        venv_py = find_target_venv(path)
        if venv_py:
            srv = ensure_pyright(venv_py, autoinstall=autoinstall)
            if srv:
                try:
                    c = _lsp.LSPClient("pyright", srv, ["--stdio"], "python")
                    c.initialize(_lsp._uri_for(os.path.dirname(os.path.abspath(path))))
                    # 指向目标 venv 的解释器：不设则 pyright 不做语义分析
                    c.did_change_configuration({
                        "python": {"pythonPath": venv_py,
                                   "venvPath": os.path.dirname(os.path.dirname(venv_py))}
                    })
                    return LspHandle(srv, c, "python", True)
                except Exception:
                    pass

    # 回落：插件 .lsp.json / LSP_SERVER_CMD 解析（客户端由全局 manager 缓存复用）
    got = _lsp.get_lsp().get_client(path)
    if not got:
        return None
    client, lang_id2, _uri = got
    return LspHandle("plugin:" + lang_id2, client, lang_id2, False)
