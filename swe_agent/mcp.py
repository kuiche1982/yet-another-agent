#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""swe_agent/mcp.py —— MCP（Model Context Protocol）客户端与管理器。

忠实移植自最初 demo.py 的 MCP 子系统（class MCPClient / MCPHttpClient /
MCPManager + resources 支持 + mcp_list_resources / mcp_read_resource）。

本模块是「插件扩展层」中 MCP 部分的真实后端：supervisor 把 `mcp_tool` /
`list_mcp_resources` / `read_mcp_resource` 三个动作路由到这里。配置读取
REPO_ROOT/mcp.json（与原始 demo.py 同款格式），支持 stdio 与 Streamable-HTTP
两种服务器，以及 ${VAR} 环境变量展开。

调用方应通过 get_mcp() 取得（懒连接）MCPManager 实例，而非直接引用模块级 MCP。
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import swe_agent.config as config
from swe_agent.log import logger

try:
    import requests
except ImportError:  # HTTP MCP 仅在有 requests 时可用；stdio MCP 不受影响
    requests = None


# ======================================================================
# stdio MCP 客户端（最小实现：initialize / tools/list / tools/call / resources/*）
# ======================================================================
class MCPClient:
    def __init__(self, name, command, args):
        self.name = name
        self.proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            start_new_session=True,
        )
        self._lock = threading.Lock()
        self._pending = {}
        self._next_id = 1
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        self.tools = []

    def _send(self, method, params=None, notification=False):
        with self._lock:
            msg_id = self._next_id
            self._next_id += 1
        msg = {"jsonrpc": "2.0", "method": method}
        if not notification:
            msg["id"] = msg_id
        if params is not None:
            msg["params"] = params
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"MCP {self.name}: 写入失败：{e}")
        if notification:
            return None
        ev = threading.Event()
        self._pending[msg_id] = {"event": ev, "result": None, "error": None}
        if not ev.wait(timeout=60):
            self._pending.pop(msg_id, None)
            raise TimeoutError(f"MCP {self.name}: 等待 {method} 响应超时")
        item = self._pending.pop(msg_id)
        if item["error"]:
            raise RuntimeError(item["error"])
        return item["result"]

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                # 只处理「有 id 且带 result/error」的响应；notification 忽略
                if "id" in msg and ("result" in msg or "error" in msg):
                    item = self._pending.get(msg["id"])
                    if item:
                        item["error"] = str(msg["error"]) if "error" in msg else None
                        item["result"] = msg.get("result")
                        item["event"].set()
        except Exception:
            pass

    def initialize(self):
        res = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "litertlm-agent", "version": "0.1"},
        })
        self._send("notifications/initialized", {}, notification=True)
        self.server_info = (res or {}).get("serverInfo", {})
        return res

    def list_tools(self):
        res = self._send("tools/list", {})
        self.tools = (res or {}).get("tools", [])
        return self.tools

    def call_tool(self, tool, arguments):
        res = self._send("tools/call", {"name": tool, "arguments": arguments or {}})
        content = res.get("content", []) if isinstance(res, dict) else []
        texts = [c.get("text", "") for c in content
                 if isinstance(c, dict) and c.get("type") == "text"]
        out = "\n".join(texts)
        is_err = res.get("isError", False) if isinstance(res, dict) else False
        return (f"mcp_error: {out}" if is_err else out) or "(无输出)"

    # ---- 资源支持（对应 claude-code 的 ListMcpResources / ReadMcpResource） ----
    def list_resources(self):
        """resources/list；服务器不支持时抛异常。"""
        res = self._send("resources/list", {})
        return (res or {}).get("resources", []) if isinstance(res, dict) else []

    def read_resource(self, uri):
        res = self._send("resources/read", {"uri": uri})
        if not isinstance(res, dict):
            return f"mcp_error: 无法读取 {uri}"
        contents = res.get("contents", [])
        texts = []
        for c in contents:
            if not isinstance(c, dict):
                continue
            if c.get("text") is not None:
                texts.append(c["text"])
            elif c.get("blob"):
                texts.append(f"(二进制资源，base64 长度 {len(c['blob'])})")
        return "\n".join(texts) or f"mcp_error: 资源 {uri} 无文本内容"

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass


def _expand_env_vars(obj):
    """递归展开字符串中的 ${VAR} 环境变量占位符（对齐 claude 插件的 mcp.json 语法）。"""
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}",
                      lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, list):
        return [_expand_env_vars(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    return obj


def normalize_mcp_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把两种 mcp.json 格式统一成 {name: spec}：
    - 裸格式：{"playwright": {"command": ...}}（claude 插件 .mcp.json 常见）
    - 包装格式：{"mcpServers": {...}} 或本项目旧格式 {"servers": {...}}
    """
    if not isinstance(cfg, dict):
        return {}
    for key in ("mcpServers", "servers"):
        if isinstance(cfg.get(key), dict):
            return cfg[key]
    # 裸格式：值必须是含 command/url 的 dict
    out = {}
    for name, spec in cfg.items():
        if isinstance(spec, dict) and ("command" in spec or "url" in spec):
            out[name] = spec
    return out


# ======================================================================
# Streamable-HTTP MCP 客户端（最小实现，用 JSON-RPC POST）
# ======================================================================
class MCPHttpClient:
    """Streamable-HTTP MCP 客户端（最小实现，用 JSON-RPC POST）。
    用于 claude 插件 .mcp.json 里 "type": "http" 的服务器（如 github）。"""

    def __init__(self, name, url, headers=None):
        if requests is None:
            raise RuntimeError("需要 requests 才能使用 HTTP MCP 客户端（pip install requests）")
        self.name = name
        self.url = url
        self.headers = {"Accept": "application/json, text/event-stream"}
        for k, v in (headers or {}).items():
            self.headers[k] = v
        self.next_id = 1
        self.tools = []

    def _send(self, method, params):
        rid = self.next_id
        self.next_id += 1
        payload = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        r = requests.post(self.url, json=payload, headers=self.headers, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        # 响应可能是 JSON 或 SSE 流（data: {...} 行）
        ctype = r.headers.get("Content-Type", "")
        if "text/event-stream" in ctype:
            for line in r.text.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    try:
                        return json.loads(line[5:].strip())
                    except Exception:
                        continue
            raise RuntimeError("SSE 响应中无 JSON 数据")
        return r.json().get("result")

    def initialize(self):
        res = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "demo-agent", "version": "1.0"},
        })
        # streamable http 需要 initialized 通知
        try:
            requests.post(self.url, timeout=15, headers=self.headers,
                          json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            pass
        return res or {}

    def list_tools(self):
        res = self._send("tools/list", {})
        self.tools = (res or {}).get("tools", [])
        return self.tools

    def call_tool(self, tool, arguments):
        res = self._send("tools/call", {"name": tool, "arguments": arguments or {}})
        content = res.get("content", []) if isinstance(res, dict) else []
        texts = [c.get("text", "") for c in content
                 if isinstance(c, dict) and c.get("type") == "text"]
        out = "\n".join(texts)
        is_err = res.get("isError", False) if isinstance(res, dict) else False
        return (f"mcp_error: {out}" if is_err else out) or "(无输出)"

    def close(self):
        pass


# ======================================================================
# MCP 管理器：连接多个服务器、维护工具索引、路由调用
# ======================================================================
class MCPManager:
    def __init__(self):
        self.clients = {}
        self.tools_index = {}   # (server, tool) -> spec

    def connect_all(self, config_path: Path, extra_servers: Dict[str, Any] = None):
        """extra_servers：插件 .mcp.json 解析出的额外服务器（已归一为 {name: spec}）。"""
        specs: Dict[str, Any] = {}
        if config_path is not None and Path(config_path).exists():
            try:
                cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
                specs.update(normalize_mcp_config(cfg))
            except Exception as e:
                logger.info('%s', f'[mcp] 配置解析失败：{e}')
        elif config_path is not None:
            logger.info('%s', f'[mcp] 未找到配置文件 {config_path}，跳过本地 MCP。')
        for name, spec in (extra_servers or {}).items():
            specs.setdefault(name, spec)
        specs = _expand_env_vars(specs)
        # 注意：venv 解释器在 REPO_ROOT/.venv（不在 swe_agent/ 下）
        venv_py = config.REPO_ROOT / ".venv" / "bin" / "python"
        for name, spec in specs.items():
            spec = spec or {}
            try:
                if spec.get("url") or spec.get("type") == "http":
                    url = spec.get("url") or ""
                    if not url:
                        logger.info('%s', f'[mcp] 服务器 {name} 声明 http 但缺 url，跳过。')
                        continue
                    client = MCPHttpClient(name, url, spec.get("headers", {}))
                else:
                    cmd = spec.get("command")
                    if not cmd:
                        logger.info('%s', f'[mcp] 服务器 {name} 缺少 command，跳过。')
                        continue
                    # "python"/"python3" 优先用项目 venv，否则用当前解释器
                    if cmd in ("python", "python3"):
                        cmd = str(venv_py) if venv_py.exists() else sys.executable
                    if cmd == "npx" or cmd == "node":
                        _node = shutil.which(cmd)
                        if _node:
                            cmd = _node
                    client = MCPClient(name, cmd, spec.get("args", []))
                client.initialize()
                tools = client.list_tools()
                self.clients[name] = client
                for t in tools:
                    self.tools_index[(name, t["name"])] = t
                logger.info('%s', f'[mcp] 已连接 {name}：{len(tools)} 个工具 -> ' + ', '.join((t['name'] for t in tools)))
            except Exception as e:
                logger.info('%s', f'[mcp] 连接 {name} 失败：{e}')

    def tool_specs(self):
        return list(self.tools_index.values())

    def call(self, server, tool, arguments):
        client = self.clients.get(server)
        if not client:
            return f"mcp_error: 未连接的服务器 '{server}'"
        if (server, tool) not in self.tools_index:
            avail = ", ".join(f"{s}:{t}" for (s, t) in self.tools_index) or "（无）"
            return f"mcp_error: 服务器 '{server}' 无工具 '{tool}'，可用：{avail}"
        return client.call_tool(tool, arguments)

    def close_all(self):
        for c in self.clients.values():
            c.close()


# ======================================================================
# 模块级单例 + 懒连接
# ======================================================================
MCP: Optional[MCPManager] = None
MCP_CONFIG_PATH = config.REPO_ROOT / "mcp.json"

# 插件层 .mcp.json 解析出的额外服务器（由 plugins.load_plugins 填充）。
# 首次 get_mcp() 时若非空，将一并连接。
PLUGIN_MCP_SERVERS: Dict[str, Any] = {}


def get_mcp() -> Optional[MCPManager]:
    """懒连接：首次调用时按 REPO_ROOT/mcp.json 连接所有服务器；若 PLUGIN_MCP_SERVERS
    非空（插件已加载且启用 MCP），一并连接插件服务器。已连接则直接返回。

    返回 None 仅当配置不存在且 connect_all 未连接任何服务器（理论不会发生，
    因为 mcp.json 始终存在；这里返回 None 仅为防御性）。
    """
    global MCP
    if MCP is None:
        MCP = MCPManager()
        extras = PLUGIN_MCP_SERVERS if PLUGIN_MCP_SERVERS else None
        MCP.connect_all(MCP_CONFIG_PATH, extra_servers=extras)
    return MCP


def mcp_prompt_section() -> str:
    m = get_mcp()
    if m is None or not m.tools_index:
        return ""
    lines = [
        "",
        "28) 调用 MCP 工具（外部服务器提供的工具，移植自 claude-code 的 MCPTool）：",
    ]
    for (server, tool), spec in m.tools_index.items():
        desc = spec.get("description", "")
        lines.append(f"- mcp__{server}__{tool}：{desc}")
    lines.append('调用格式：{"action":"mcp_tool","server":"服务器名","tool":"工具名","arguments":{...}}')
    lines.append("arguments 必须是 JSON 对象（键值对）。")
    return "\n".join(lines)


# ---- MCP 资源（ListMcpResources / ReadMcpResource） ----
def mcp_list_resources() -> str:
    m = get_mcp()
    if m is None:
        return "mcp_error: MCP 未启用"
    out = []
    for srv, client in m.clients.items():
        try:
            res = client.list_resources()
        except Exception as e:
            out.append(f"- {srv}: 列举失败 {e}")
            continue
        if not res:
            continue
        out.append(f"- {srv}:")
        for r in res:
            out.append(f"    • {r.get('uri')}  ({r.get('name', '')})  {r.get('description','')}")
    return "\n".join(out) or "mcp: 无可用资源"


def mcp_read_resource(uri: str) -> str:
    m = get_mcp()
    if m is None:
        return "mcp_error: MCP 未启用"
    for srv, client in m.clients.items():
        try:
            return client.read_resource(uri)
        except Exception:
            continue
    return f"mcp_error: 未找到资源 {uri}（或服务器不支持 resources/read）"
