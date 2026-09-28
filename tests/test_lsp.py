#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # -*- coding: utf-8 -*-
    """test_lsp.py —— LSP 子系统 + _auto_lsp_hint 钩子接线验证（离线、不依赖真实 LSP 服务器）。

    验证点：
    1) lsp 动作已注册为真实动作（非插件桩），run 指向 _m_lsp；
    2) _auto_lsp_hint 在未配置 LSP 服务器时返回 "" 且不抛异常（绝不拖垮 write/edit 钩子）；
    3) 写/编辑钩子确实挂了 _hook_write_after（内含 _auto_lsp_hint）；
    4) lsp 动作缺 path 时受控返回；
    5) load_plugins 能正确把插件 .lsp.json 收集进 lsp.PLUGIN_LSP_SERVERS（按 languageId 索引）。
    """

    import json
    import sys
    import tempfile
    from pathlib import Path

    from swe_agent.registry import ToolRegistry, ActionContext
    import swe_agent.supervisor as S
    import swe_agent.lsp as lsp
    from swe_agent.plugins import load_plugins


    def check(name, cond):
        print(("✅" if cond else "❌") + " " + name)
        if not cond:
            assert False


    # 1) lsp 动作已注册且是真实实现（非桩）
    t = ToolRegistry.get("lsp")
    check("lsp 动作已注册", t is not None)
    check("lsp 动作非桩（run 指向 _m_lsp）", t.run.__name__ == "_m_lsp")


    # 2) _auto_lsp_hint 离线安全（无服务器 -> "" 且不抛异常）
    try:
        hint = S._auto_lsp_hint("agent_sandbox/whatever.py")
        check("_auto_lsp_hint 返回 str 且不抛异常", isinstance(hint, str))
        check("_auto_lsp_hint 无服务器时为空串", hint == "")
    except Exception as e:
        check(f"_auto_lsp_hint 不应抛异常（实际：{e}）", False)


    # 3) 写/编辑钩子挂了 _hook_write_after（内含 _auto_lsp_hint）
    wf = ToolRegistry.get("write_file")
    ef = ToolRegistry.get("edit_file")
    check("write_file.after 含 _hook_write_after",
          any(h.__name__ == "_hook_write_after" for h in wf.after))
    check("edit_file.after 含 _hook_edit_after",
          any(h.__name__ == "_hook_edit_after" for h in ef.after))


    # 4) lsp 动作缺 path 受控返回
    res = ToolRegistry.dispatch({"action": "lsp", "path": ""}, ActionContext(messages=[]))
    check("lsp 缺 path 受控返回", res.startswith("lsp_error"))


    # 5) dispatch lsp 动作离线不崩溃（无服务器 -> 友好提示串）
    res2 = ToolRegistry.dispatch({"action": "lsp", "path": "agent_sandbox/none.py"},
                                 ActionContext(messages=[]))
    check("dispatch lsp 动作离线返回提示且不崩溃", isinstance(res2, str) and "lsp" in res2)


    # 6) load_plugins 收集插件 LSP 服务器（按 languageId 索引）
    tmp = tempfile.mkdtemp()
    root = Path(tmp)
    plugin = root / "myplugin"
    plugin.mkdir()
    (plugin / ".lsp.json").write_text(json.dumps([
        {"languageId": "python", "command": "pylsp", "args": ["--help"]},
        {"languageId": "typescript", "command": "typescript-language-server", "args": ["--stdio"]},
    ]))
    (root / "installed_plugins.json").write_text(json.dumps({
        "plugins": {"myplugin@1.0": [{"installPath": str(plugin)}]}
    }))

    # 调用前先清环境，避免 LSP_SERVER_CMD 干扰断言
    import os
    os.environ.pop("LSP_SERVER_CMD", None)

    state = load_plugins(root)
    check("load_plugins 统计含 lsp 键", "lsp" in state)
    check("插件 python LSP 服务器已收集", lsp.PLUGIN_LSP_SERVERS.get("python", {}).get("command") == "pylsp")
    check("插件 typescript LSP 服务器已收集",
          lsp.PLUGIN_LSP_SERVERS.get("typescript", {}).get("command") == "typescript-language-server")

    # 收集后 hint 仍离线安全（命令不存在 -> 启动失败 -> 返回空串，不抛异常）
    try:
        h = lsp.get_lsp().hint("agent_sandbox/x.py")
        check("收集后 hint 仍离线安全（返回 str）", isinstance(h, str))
    except Exception as e:
        check(f"收集后 hint 不应抛异常（实际：{e}）", False)

    print("\n=== test_lsp 全部通过 ===")


if __name__ == "__main__":
    main()


def test_main():
    main()
