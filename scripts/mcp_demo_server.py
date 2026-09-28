#!/usr/bin/env python3
"""极简 MCP 演示服务器（stdio + JSON-RPC 2.0，仅依赖标准库）。

实现 MCP 的最小子集：initialize / tools/list / tools/call。
把它当作「外部工具服务器」的示例：demo.py 通过 mcp.json 连接它，
并把它的工具以 mcp__demo__<tool> 的形式提供给 Ling-3.0-tiny 调用。

可用工具：
- add(a, b)  -> 返回 a+b
- now()      -> 返回当前 ISO 时间
"""
import sys
import json
from datetime import datetime


def send(obj: dict):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


TOOLS = [
    {
        "name": "add",
        "description": "两数相加，返回 a + b",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一个数"},
                "b": {"type": "number", "description": "第二个数"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "now",
        "description": "返回当前本地时间的 ISO 8601 字符串",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def call_tool(name: str, arguments: dict):
    try:
        if name == "add":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            return {"content": [{"type": "text", "text": str(a + b)}]}
        if name == "now":
            return {"content": [{"type": "text", "text": datetime.now().isoformat()}]}
        return {"isError": True, "content": [{"type": "text", "text": f"未知工具 {name}"}]}
    except Exception as e:  # pragma: no cover
        return {"isError": True, "content": [{"type": "text", "text": str(e)}]}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        method = msg.get("method")
        mid = msg.get("id")

        if method == "initialize":
            send({
                "jsonrpc": "2.0", "id": mid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "demo", "version": "0.1"},
                },
            })
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = msg.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            send({"jsonrpc": "2.0", "id": mid, "result": call_tool(name, args)})
        else:
            if mid is not None:
                send({"jsonrpc": "2.0", "id": mid, "result": {}})


if __name__ == "__main__":
    main()
