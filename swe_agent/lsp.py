#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""swe_agent/lsp.py —— LSP（Language Server Protocol）客户端与管理器。

对齐 claude-code 的 LSPTool（tools/LSPTool/LSPTool.ts）与插件 LSP 集成机制
（utils/plugins/lspPluginIntegration.ts）：LSP 服务器本身不内置，完全由插件通过
.lsp.json 或插件清单的 lspServers 字段提供（与 MCP 由 .mcp.json 提供同理）。

本模块是「插件扩展层」中 LSP 部分的真实后端：
- supervisor 把 `lsp` 动作路由到这里；
- write_file / edit_file 的 after 钩子（_auto_lsp_hint）调用 lsp.hint(path) 把
  诊断结果作为**非阻塞提示**回灌给模型（对齐 claude-code：写入文件后自动展示 LSP 诊断）。

传输：stdio + JSON-RPC，Content-Length 分帧（标准 LSP 线缆格式，不同于 MCP 的
逐行 JSON）。
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import swe_agent.config as config

# 扩展名 -> LSP languageId（与 claude-code 的默认映射对齐，按需扩展）
_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascriptreact",
    ".ts": "typescript", ".tsx": "typescriptreact",
    ".go": "go", ".rs": "rust", ".java": "java", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".kt": "kotlin", ".lua": "lua",
    ".sh": "shellscript", ".bash": "shellscript", ".zsh": "shellscript",
    ".json": "json", ".html": "html", ".css": "css", ".scss": "scss",
    ".md": "markdown", ".xml": "xml", ".sql": "sql",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
}


# ======================================================================
# LSP 客户端（stdio + Content-Length 分帧 JSON-RPC）
# ======================================================================
class LSPClient:
    def __init__(self, name, command, args, lang_id):
        self.name = name
        self.lang_id = lang_id
        self._ver = 0
        try:
            self.proc = subprocess.Popen(
                [command, *args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as e:
            raise RuntimeError(f"LSP {name}: 启动失败：{e}")
        self._buf = b""
        self._lock = threading.Lock()
        self._pending = {}
        self._next_id = 1
        self._diag_event = threading.Event()
        self._diagnostics: Dict[str, List[Dict[str, Any]]] = {}
        # initialize 返回的服务器能力（能力探测用：不是所有服务器都支持同一套能力）
        self.server_capabilities: Dict[str, Any] = {}
        # 服务器 → 客户端 请求记录（调试用；不响应会死锁，见 _dispatch）
        self.server_requests: List[Dict[str, Any]] = []
        self._opened: Dict[str, int] = {}   # uri -> version（避免重复 didOpen）
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

    # ---- 线缆：Content-Length: N\r\n\r\n{json} ----
    def _send(self, method, params=None, notification=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if not notification:
            with self._lock:
                msg_id = self._next_id
                self._next_id += 1
            msg["id"] = msg_id
        if params is not None:
            msg["params"] = params
        data = json.dumps(msg).encode("utf-8")
        frame = b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data
        try:
            self.proc.stdin.write(frame)
            self.proc.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"LSP {self.name}: 写入失败：{e}")
        if notification:
            return None
        ev = threading.Event()
        with self._lock:
            self._pending[msg_id] = {"event": ev, "result": None, "error": None}
        if not ev.wait(timeout=60):
            with self._lock:
                self._pending.pop(msg_id, None)
            raise TimeoutError(f"LSP {self.name}: 等待 {method} 响应超时")
        with self._lock:
            item = self._pending.pop(msg_id)
        if item["error"]:
            raise RuntimeError(item["error"])
        return item["result"]

    def _read_loop(self):
        try:
            while True:
                # 读 header，遇到空行结束
                headers: Dict[str, str] = {}
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        return
                    line = line.strip()
                    if line == b"":
                        break
                    if b":" in line:
                        k, v = line.split(b":", 1)
                        headers[k.strip().lower().decode()] = v.strip().decode()
                length = int(headers.get("content-length", "0"))
                if length <= 0:
                    continue
                body = b""
                while len(body) < length:
                    chunk = self.proc.stdout.read(length - len(body))
                    if not chunk:
                        return
                    body += chunk
                try:
                    msg = json.loads(body.decode("utf-8"))
                except Exception:
                    continue
                self._dispatch(msg)
        except Exception:
            pass

    # 服务器 → 客户端 请求的默认应答。
    # 必须回：不回则服务器阻塞（pyright 会在 workspace/configuration 上死锁，
    # 表现为「请求无响应也无报错」）。未知方法回 None（JSON-RPC 允许 null result）。
    _SERVER_REQUEST_REPLIES = {
        "workspace/configuration": [],
        "workspace/applyEdit": {"applied": False},
        "client/registerCapability": None,
        "client/unregisterCapability": None,
        "workspace/semanticTokens/refresh": None,
        "workspace/codeLens/refresh": None,
        "workspace/inlayHint/refresh": None,
        "window/workDoneProgress/create": None,
        "window/showMessageRequest": None,
    }

    def _reply_to_server(self, msg):
        method = msg.get("method")
        result = self._SERVER_REQUEST_REPLIES.get(method)
        self.server_requests.append({"method": method, "params": msg.get("params")})
        out = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
        data = json.dumps(out).encode("utf-8")
        try:
            self.proc.stdin.write(
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
            self.proc.stdin.flush()
        except Exception:
            pass

    def _dispatch(self, msg):
        if "id" in msg and "method" in msg:
            # 服务器 → 客户端 请求（与「响应」的区别：带 method）
            self._reply_to_server(msg)
            return
        if "id" in msg and ("result" in msg or "error" in msg):
            with self._lock:
                item = self._pending.get(msg["id"])
            if item:
                item["error"] = str(msg["error"]) if "error" in msg else None
                item["result"] = msg.get("result")
                item["event"].set()
        elif "method" in msg:
            # 通知：主要是 publishDiagnostics（旧模型）与 window/logMessage
            if msg["method"] == "textDocument/publishDiagnostics":
                params = msg.get("params", {})
                uri = params.get("uri")
                if uri:
                    self._diagnostics[uri] = params.get("diagnostics", [])
                self._diag_event.set()
            # window/logMessage / telemetry 等忽略

    def initialize(self, root_uri):
        res = self._send("initialize", {
            "processId": None,
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "synchronization": {"dynamicRegistration": True},
                    "diagnostic": {"dynamicRegistration": True},
                    "publishDiagnostics": {"relatedInformation": True},
                    # 分层符号（DocumentSymbol[] 而非扁平 SymbolInformation[]）
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                    "callHierarchy": {"dynamicRegistration": True},
                    "definition": {"dynamicRegistration": True},
                    "references": {"dynamicRegistration": True},
                    "hover": {"dynamicRegistration": True},
                },
                "workspace": {"diagnostics": {"refreshSupport": True}},
            },
            "clientInfo": {"name": "litertlm-agent", "version": "0.1"},
        })
        # 能力探测：不是所有服务器都支持同一套（pylsp 无 callHierarchy，pyright/gopls 有）
        self.server_capabilities = (res or {}).get("capabilities", {}) or {}
        self._send("initialized", {}, notification=True)
        return res

    def supports(self, capability: str) -> bool:
        """服务器是否声明了该能力（值可以是 True 或对象，均视为支持）。"""
        return bool(self.server_capabilities.get(capability))

    def did_change_configuration(self, settings: Dict[str, Any]) -> None:
        """推送 workspace/didChangeConfiguration（如 pyright 的 python.pythonPath）。"""
        try:
            self._send("workspace/didChangeConfiguration",
                       {"settings": settings}, notification=True)
        except Exception:
            pass

    # ---- 打开文档（同一个 uri 只 didOpen 一次） ----
    def open_document(self, path: str, lang_id: str | None = None,
                      text: str | None = None) -> str:
        """didOpen 一个文件，返回 uri。重复调用只推一次（省去重复解析）。"""
        uri = _uri_for(path)
        if uri in self._opened:
            return uri
        if text is None:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        ext = os.path.splitext(path)[1].lower()
        self.did_open(uri, lang_id or _EXT_TO_LANG.get(ext, "plaintext"), text)
        self._opened[uri] = self._ver
        return uri

    def did_open(self, uri, lang_id, text):
        self._ver += 1
        self._send("textDocument/didOpen", {
            "textDocument": {
                "uri": uri, "languageId": lang_id,
                "version": self._ver, "text": text,
            }
        }, notification=True)

    def get_diagnostics(self, uri, lang_id, text, timeout=20) -> List[Dict[str, Any]]:
        """didOpen 后请求诊断，返回 diagnostics 列表（pull 或 push 均可）。"""
        self._diagnostics.pop(uri, None)
        self._diag_event.clear()
        self.did_open(uri, lang_id, text)
        # 优先 pull 模型（LSP 3.17 textDocument/diagnostic）
        try:
            res = self._send("textDocument/diagnostic", {
                "textDocument": {"uri": uri},
                "identifier": None,
            })
            if isinstance(res, dict) and "items" in res:
                return res["items"]
        except Exception:
            pass
        # 回退：等待服务器 publishDiagnostics 推送
        self._diag_event.wait(timeout=timeout)
        return self._diagnostics.get(uri, [])

    # ---- 语义能力：符号 / 引用 / 悬停 / 调用层级 ----
    # 位置参数统一为 line0 / char0：**0-based**（LSP 原生）。
    # 注意：定位符号要用 selectionRange（名字所在位置），不是 range（含 def/class 关键字）——
    # 用 range.start 去请求 hover/references/callHierarchy 会全部返回空，看着像服务器不支持。
    def document_symbols(self, uri: str) -> List[Dict[str, Any]]:
        """textDocument/documentSymbol → 扁平化符号列表（含 depth）。

        每项：{name, kind, kind_name, lineno, end_lineno, depth, line0, char0}
        lineno/end_lineno 为 **1-based 闭区间**（与 ast 的 lineno 对齐）；
        line0/char0 来自 selectionRange.start，可直接喂给 references/callHierarchy。
        """
        if not self.supports("documentSymbolProvider"):
            return []
        res = self._send("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
        out: List[Dict[str, Any]] = []
        _flatten_document_symbols(res, out)
        return out

    def top_level_symbols(self, uri: str) -> List[Dict[str, Any]]:
        """只取顶层符号（depth==0）：类方法等嵌套符号不参与文件切分。"""
        return [s for s in self.document_symbols(uri) if s.get("depth") == 0]

    def hover(self, uri: str, line0: int, char0: int) -> str:
        if not self.supports("hoverProvider"):
            return ""
        res = self._send("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line0, "character": char0},
        })
        return _hover_text(res)

    def references(self, uri: str, line0: int, char0: int,
                   include_declaration: bool = True) -> List[Dict[str, Any]]:
        """textDocument/references → [{uri, line, character}]（line 为 1-based）。"""
        if not self.supports("referencesProvider"):
            return []
        res = self._send("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": line0, "character": char0},
            "context": {"includeDeclaration": include_declaration},
        })
        out = []
        for loc in res or []:
            if not isinstance(loc, dict):
                continue
            start = (loc.get("range") or {}).get("start") or {}
            out.append({
                "uri": loc.get("uri", ""),
                "line": start.get("line", 0) + 1,
                "character": start.get("character", 0) + 1,
            })
        return out

    def prepare_call_hierarchy(self, uri: str, line0: int, char0: int) -> List[Dict[str, Any]]:
        if not self.supports("callHierarchyProvider"):
            return []
        res = self._send("textDocument/prepareCallHierarchy", {
            "textDocument": {"uri": uri},
            "position": {"line": line0, "character": char0},
        })
        return [i for i in (res or []) if isinstance(i, dict)]

    def outgoing_calls(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """该函数调用了谁 → [{name, uri, line, character}]。"""
        res = self._send("callHierarchy/outgoingCalls", {"item": item})
        out = []
        for c in res or []:
            to = c.get("to") or {}
            start = ((to.get("range") or {}).get("start")) or {}
            out.append({"name": to.get("name", ""), "uri": to.get("uri", ""),
                        "line": start.get("line", 0) + 1,
                        "character": start.get("character", 0) + 1})
        return out

    def incoming_calls(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """谁调用了这个函数 → [{name, uri, line, character}]。"""
        res = self._send("callHierarchy/incomingCalls", {"item": item})
        out = []
        for c in res or []:
            frm = c.get("from") or {}
            start = ((frm.get("range") or {}).get("start")) or {}
            out.append({"name": frm.get("name", ""), "uri": frm.get("uri", ""),
                        "line": start.get("line", 0) + 1,
                        "character": start.get("character", 0) + 1})
        return out

    def wait_for_semantics(self, uri: str, timeout: float = 20.0,
                           poll: float = 1.0) -> bool:
        """等语义分析就绪：轮询首个顶层符号的 hover 直到非空。

        pyright 首启需要解析整个环境（冷启动可达分钟级）；不做这个等待，
        后续 references/callHierarchy 会拿到空结果，容易被误判成「服务器不支持」。
        """
        if not self.supports("hoverProvider"):
            return True
        deadline = time.time() + timeout
        syms = self.top_level_symbols(uri)
        target = next((s for s in syms if s.get("kind") in (12, 6, 5)), None)
        if target is None:
            return False
        while time.time() < deadline:
            try:
                if self.hover(uri, target["line0"], target["char0"]):
                    return True
            except Exception:
                return False
            time.sleep(poll)
        return False

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass


def _resolve_cmd(command: str) -> str:
    """把命令路径规范化，使其不受 harness 当前工作目录影响：

    - 含 '/' 的相对路径（如 `.venv/bin/pylsp`）按 ``config.REPO_ROOT`` 展开；
    - '~' 展开为用户目录；
    - 纯命令名（如 ``gopls``）原样返回，交给 PATH 解析。
    """
    if not command:
        return command
    if "/" in command and not command.startswith("/"):
        p = (config.REPO_ROOT / command).resolve()
        if p.exists():
            return str(p)
    return os.path.expanduser(command)


def _uri_for(path: str) -> str:
    # realpath：macOS 上 /var 是 /private/var 的符号链接，URI 不一致会让
    # 服务器认为文件不在 workspace 内而跳过语义分析。
    return Path(os.path.realpath(path)).as_uri()


def _root_uri() -> str:
    try:
        return config.REPO_ROOT.resolve().as_uri()
    except Exception:
        return Path.cwd().resolve().as_uri()


# LSP SymbolKind -> 可读名（只列 contextmgr / 诊断会用到的几类）
_KIND_NAME = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class",
    6: "method", 7: "property", 8: "field", 9: "constructor", 10: "enum",
    11: "interface", 12: "function", 13: "variable", 14: "constant",
    23: "struct", 24: "event", 25: "operator", 26: "typeParameter",
}


def _flatten_document_symbols(nodes, out, depth=0):
    """递归展开 DocumentSymbol[]（含 children）或 SymbolInformation[]。"""
    for n in nodes or []:
        if not isinstance(n, dict):
            continue
        loc = n.get("location") or {}
        rng = loc.get("range") or n.get("range") or {}
        sel = n.get("selectionRange") or rng
        start = rng.get("start") or {}
        end = rng.get("end") or {}
        sel_start = sel.get("start") or {}
        kind = n.get("kind", 0)
        out.append({
            "name": n.get("name", ""),
            "kind": kind,
            "kind_name": _KIND_NAME.get(kind, str(kind)),
            # 1-based 闭区间，与 ast.lineno / end_lineno 对齐
            "lineno": start.get("line", 0) + 1,
            "end_lineno": end.get("line", 0) + 1,
            "depth": depth,
            # 0-based，LSP 原生；来自 selectionRange（名字位置，非 def 关键字）
            "line0": sel_start.get("line", 0),
            "char0": sel_start.get("character", 0),
        })
        _flatten_document_symbols(n.get("children"), out, depth + 1)
    return out


def _hover_text(res) -> str:
    """hover 结果取文本：支持 MarkedString | MarkedString[] | MarkupContent。"""
    if not res:
        return ""
    contents = res.get("contents") if isinstance(res, dict) else res
    if isinstance(contents, dict):
        return str(contents.get("value", ""))
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        parts = []
        for c in contents:
            if isinstance(c, dict):
                parts.append(str(c.get("value", "")))
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(p for p in parts if p)
    return ""


def _fmt_diag(diags: List[Dict[str, Any]]) -> str:
    if not diags:
        return ""
    lines = ["\n[LSP 诊断]"]
    for d in diags:
        rng = d.get("range", {}).get("start", {})
        ln = rng.get("line")
        line = (ln + 1) if isinstance(ln, int) else "?"
        sev = {1: "E", 2: "W", 3: "I", 4: "H"}.get(d.get("severity", 3), "I")
        msg = d.get("message", "")
        src = d.get("source", "")
        lines.append(f"  {sev} L{line}: {msg}" + (f" ({src})" if src else ""))
    return "\n".join(lines)


# ======================================================================
# LSP 管理器：按 languageId 复用客户端（持久化，避免每次写入都启动服务器）
# ======================================================================
class LSPManager:
    def __init__(self):
        self._clients: Dict[str, LSPClient] = {}

    def _resolve(self, path: str):
        """返回 (command, args, lang_id) 或 None。

        优先级：插件声明的该语言服务器（PLUGIN_LSP_SERVERS[languageId]）
                > 环境变量 LSP_SERVER_CMD（统一服务器，不限定语言）。
        """
        ext = os.path.splitext(path)[1].lower()
        lang_id = _EXT_TO_LANG.get(ext)
        if lang_id and lang_id in PLUGIN_LSP_SERVERS:
            spec = PLUGIN_LSP_SERVERS[lang_id]
            cmd = spec.get("command")
            if cmd:
                return _resolve_cmd(cmd), spec.get("args", []), lang_id
        env_cmd = os.environ.get("LSP_SERVER_CMD")
        if env_cmd:
            return env_cmd, [], (lang_id or "unknown")
        return None

    def hint(self, path: str) -> str:
        """返回该文件的 LSP 诊断提示串；无服务器/无诊断/任何异常时返回空串
        （与旧 stub 'return \"\"' 行为一致，且绝不会拖垮 write/edit 钩子）。"""
        try:
            resolved = self._resolve(path)
            if not resolved or not resolved[0]:
                return ""
            command, args, lang_id = resolved
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        client = self._clients.get(lang_id)
        created = False
        if client is None:
            try:
                client = LSPClient(f"lsp-{lang_id}", command, args, lang_id)
                client.initialize(_root_uri())
                self._clients[lang_id] = client
                created = True
            except Exception:
                # 启动/握手失败：不要缓存坏客户端，返回空提示
                return ""
        try:
            diags = client.get_diagnostics(_uri_for(path), lang_id, text)
            return _fmt_diag(diags)
        except Exception:
            # 客户端可能已死：丢弃以便下次重建
            if not created:
                try:
                    client.close()
                except Exception:
                    pass
                self._clients.pop(lang_id, None)
            return ""

    def get_client(self, path: str, root_uri: Optional[str] = None):
        """返回 (client, lang_id, uri)；无可用服务器 / 启动握手失败返回 None。

        与 hint() 共享客户端缓存：同一 languageId 复用一个服务器进程（起进程很贵）。
        root_uri 只在首次创建该语言的客户端时生效（服务器 root 不可变）。
        """
        try:
            resolved = self._resolve(path)
            if not resolved or not resolved[0]:
                return None
            command, args, lang_id = resolved
        except Exception:
            return None
        client = self._clients.get(lang_id)
        if client is None:
            try:
                client = LSPClient(f"lsp-{lang_id}", command, args, lang_id)
                client.initialize(root_uri or _root_uri())
                self._clients[lang_id] = client
            except Exception:
                self._clients.pop(lang_id, None)
                return None
        return client, lang_id, _uri_for(path)

    def close_all(self):
        for c in self._clients.values():
            try:
                c.close()
            except Exception:
                pass
        self._clients.clear()

    def is_configured(self) -> bool:
        """是否有任何 LSP 服务器可用：插件声明的服务器（PLUGIN_LSP_SERVERS）或
        LSP_SERVER_CMD 环境变量。供传感器/钩子判断「LSP 插件是否就绪」。"""
        return bool(PLUGIN_LSP_SERVERS) or bool(os.environ.get("LSP_SERVER_CMD"))

    def available_for(self, path: str) -> bool:
        """该文件扩展名是否有对应语言服务器（插件或环境变量）。"""
        return self._resolve(path) is not None


# ======================================================================
# 模块级单例 + 懒连接
# ======================================================================
LSP: Optional[LSPManager] = None
# 插件层 .lsp.json / lspServers 解析出的服务器（由 plugins.load_plugins 填充）
# 结构：{ languageId: {"command":..., "args":[...], ...}, ... }
PLUGIN_LSP_SERVERS: Dict[str, Any] = {}


def get_lsp() -> LSPManager:
    """懒加载 LSPManager 单例（始终返回一个管理器；无服务器时 hint 返回空串）。"""
    global LSP
    if LSP is None:
        LSP = LSPManager()
    return LSP


def lsp_diagnostics(path: str) -> str:
    """`lsp` 动作入口：返回该文件的 LSP 诊断。"""
    res = get_lsp().hint(path)
    return res or "lsp: 无诊断（文件通过，或未配置对应语言的 LSP 服务器）"
