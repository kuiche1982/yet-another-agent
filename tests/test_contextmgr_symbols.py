"""contextmgr/symbols.py —— SymbolProvider 双路（AST / LSP）。

单测铁律：model-free / tool-free。LspSymbolProvider 用假客户端注入，不起任何
真实 LSP 进程；ensure_pyright 的失败分支用纯路径判定 + monkeypatch 触发，不联网。
"""

import ast
import os
import sys

import pytest

from contextmgr.symbols import (
    ASTSymbolProvider,
    CodeSymbol,
    LspHandle,
    LspSymbolProvider,
    SymbolProvider,
    ensure_pyright,
    find_target_venv,
    pyright_status,
)


# ----------------------------------------------------------------------
# ASTSymbolProvider
# ----------------------------------------------------------------------
def test_ast_provider_returns_top_level_defs_with_lines():
    # 行号对照（1-based，闭区间）：
    # 1 import os / 2 (空) / 3 def foo / 4 """doc""" / 5 return / 6 (空) / 7 class Bar ...
    src = (
        "import os\n"
        "\n"
        "def foo(a, b):\n"
        '    """doc"""\n'
        "    return a + b\n"
        "\n"
        "class Bar:\n"
        "    def method(self):\n"
        "        pass\n"
    )
    syms = ASTSymbolProvider().symbols(src, "mod.py")
    assert syms is not None
    names = [s.name for s in syms]
    assert names == ["foo", "Bar"]           # 顶层 def + class，嵌套 method 不计入
    foo = next(s for s in syms if s.name == "foo")
    assert foo.kind == "def"
    assert foo.lineno == 3                    # def 行（不含装饰器）
    assert foo.end_lineno >= 5
    bar = next(s for s in syms if s.name == "Bar")
    assert bar.kind == "class"
    assert bar.lineno == 7


def test_ast_provider_non_python_returns_none():
    # 非合法 Python（如一段 Go 代码）→ 切不出来
    assert ASTSymbolProvider().symbols("package main\nfunc main() {}", "main.go") is None


def test_ast_provider_no_top_level_defs_returns_empty_list():
    assert ASTSymbolProvider().symbols("x = 1\ny = 2\n", "mod.py") == []


# ----------------------------------------------------------------------
# LspSymbolProvider（假客户端，不起进程）
# ----------------------------------------------------------------------
class _FakeClient:
    def __init__(self, syms):
        self._syms = syms
        self.opened = None

    def open_document(self, path, lang_id, text):
        self.opened = (path, lang_id)
        return "file:///fake"

    def wait_for_semantics(self, uri, timeout=20.0):
        return True

    def top_level_symbols(self, uri):
        return self._syms

    def close(self):
        pass


def _fake_factory(syms):
    def factory(path, autoinstall=True):
        c = _FakeClient(syms)
        return LspHandle(key="fake:" + os.path.splitext(path)[1], client=c,
                         lang_id="python", owned=True)
    return factory


_LSP_SYMS = [
    {"name": "main", "kind": 12, "kind_name": "function", "lineno": 1,
     "end_lineno": 5, "depth": 0, "line0": 0, "char0": 4},
    {"name": "Helper", "kind": 5, "kind_name": "class", "lineno": 7,
     "end_lineno": 12, "depth": 0, "line0": 6, "char0": 6},
]


def test_lsp_provider_converts_symbols():
    prov = LspSymbolProvider(client_factory=_fake_factory(_LSP_SYMS))
    out = prov.symbols("def main(): pass\nclass Helper: pass\n", "m.py", path="/tmp/m.py")
    assert out is not None
    assert isinstance(out[0], CodeSymbol)
    names = [s.name for s in out]
    assert names == ["main", "Helper"]
    assert out[0].kind == "def"            # kind 12 -> "def"
    assert out[1].kind == "class"          # kind 5  -> "class"
    assert out[0].lineno == 1 and out[0].end_lineno == 5
    assert out[0].meta.get("source") == "lsp"
    prov.close()


def test_lsp_provider_needs_path():
    prov = LspSymbolProvider(client_factory=_fake_factory(_LSP_SYMS))
    # 没有真实 path，LSP 无法 didOpen → 回落 None（调用方退化为整文件片段）
    assert prov.symbols("def main(): pass\n", "m.py") is None


def test_lsp_provider_factory_none_falls_back():
    prov = LspSymbolProvider(client_factory=lambda path, autoinstall=True: None)
    assert prov.symbols("def main(): pass\n", "m.py", path="/tmp/m.py") is None


def test_lsp_provider_client_error_falls_back():
    def boom_factory(path, autoinstall=True):
        raise RuntimeError("lsp died")
    prov = LspSymbolProvider(client_factory=boom_factory)
    assert prov.symbols("def main(): pass\n", "m.py", path="/tmp/m.py") is None


def test_lsp_provider_empty_symbols_falls_back():
    prov = LspSymbolProvider(client_factory=_fake_factory([]))
    assert prov.symbols("def main(): pass\n", "m.py", path="/tmp/m.py") is None
    prov.close()


# ----------------------------------------------------------------------
# find_target_venv
# ----------------------------------------------------------------------
def test_find_target_venv(tmp_path):
    venv = tmp_path / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("")
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    f = src / "mod.py"
    f.write_text("x = 1\n")
    found = find_target_venv(str(f))
    assert found == str(venv / "python")


def test_find_target_venv_missing(tmp_path, monkeypatch):
    # 隔离环境：任何 venv python 都视为不存在，验证「找不到就返回 None」
    monkeypatch.setattr("contextmgr.symbols.os.path.isfile", lambda p: False)
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n")
    assert find_target_venv(str(f)) is None


# ----------------------------------------------------------------------
# ensure_pyright / pyright_status（不联网、不装包）
# ----------------------------------------------------------------------
def test_ensure_pyright_existing_returns_path():
    # 复用本仓库 .venv（用户已装 pyright）
    here = os.path.dirname(os.path.abspath(__file__))
    repo_venv = os.path.join(here, "..", ".venv", "bin", "python")
    repo_venv = os.path.abspath(repo_venv)
    if os.path.isfile(repo_venv):
        srv = os.path.join(os.path.dirname(repo_venv), "pyright-langserver")
        if os.path.isfile(srv):
            assert ensure_pyright(repo_venv) == srv


def test_ensure_pyright_missing_returns_none():
    # 找不到且不强制：返回 None（不抛、不装），由调用方/端口回落到状态文案
    assert ensure_pyright("/no/such/venv/bin/python", autoinstall=False) is None


def test_ensure_pyright_missing_no_uv_returns_none(monkeypatch):
    monkeypatch.setattr("contextmgr.symbols.shutil.which", lambda x: None)
    assert ensure_pyright("/no/such/venv/bin/python", autoinstall=True) is None


def test_pyright_status_strings():
    assert pyright_status(None) == "LSP 服务未启动"
    assert "pyright 未安装" in pyright_status("/no/such/venv/bin/python")
    # 仓库自带 pyright 时应报 ok
    here = os.path.dirname(os.path.abspath(__file__))
    repo_venv = os.path.abspath(os.path.join(here, "..", ".venv", "bin", "python"))
    if os.path.isfile(os.path.join(os.path.dirname(repo_venv), "pyright-langserver")):
        assert pyright_status(repo_venv) == "ok"
