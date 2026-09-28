#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # -*- coding: utf-8 -*-
    """验证 lsp 插件配置被正确加载，并且真实服务器能产出诊断。

    运行：.venv/bin/python verify_lsp_config.py
    """
    import os
    import sys
    import tempfile
    import traceback

    import swe_agent.config as C
    import swe_agent.plugins as P
    import swe_agent.lsp as L

    fails = []


    def check(name, cond, extra=""):
        status = "✅" if cond else "❌"
        print(f"{status} {name}" + (f"  -> {extra}" if extra else ""))
        if not cond:
            fails.append(name)


    # ---------- 1) 加载插件 ----------
    state = P.load_plugins(C.PLUGINS_ROOT, enable_mcp=False)
    print("\n[PLUGIN_STATE.lsp] =", state.get("lsp"))
    print("[PLUGIN_LSP_SERVERS] =", {k: v.get("command") for k, v in L.PLUGIN_LSP_SERVERS.items()})

    # ---------- 2) 期望的语言服务器都已声明 ----------
    expected = {"python", "go", "rust", "c", "cpp", "typescript", "javascript"}
    present = set(L.PLUGIN_LSP_SERVERS.keys())
    check("python LSP 已声明", "python" in present, L.PLUGIN_LSP_SERVERS.get("python", {}).get("command"))
    check("go LSP 已声明", "go" in present)
    check("rust LSP 已声明", "rust" in present)
    check("c LSP 已声明", "c" in present)
    check("cpp LSP 已声明", "cpp" in present)
    check("typescript LSP 已声明", "typescript" in present)
    check("javascript LSP 已声明", "javascript" in present)
    check("声明集合覆盖期望语言", expected.issubset(present), f"缺: {expected - present}")

    # python 命令应被解析为 venv 内绝对路径
    py_cmd = L.PLUGIN_LSP_SERVERS.get("python", {}).get("command", "")
    check("python 命令解析到 .venv 绝对路径", py_cmd.endswith("bin/pylsp") and os.path.exists(py_cmd), py_cmd)

    # ---------- 3) 真实诊断：写临时 .go 文件（类型错误） ----------
    mgr = L.get_lsp()
    tmpdir = tempfile.mkdtemp(prefix="lsp_verify_")
    go_path = os.path.join(tmpdir, "sample.go")
    with open(go_path, "w") as f:
        f.write("package main\n\nfunc main() {\n\tvar x int = \"hello\"\n}\n")  # 类型不匹配

    go_diag = mgr.hint(go_path)
    print("\n[go 诊断] =>", repr(go_diag[:300]))
    check("gopls 对类型错误产出诊断", bool(go_diag) and ("L" in go_diag), "gopls 未返回诊断" if not go_diag else "")

    # ---------- 4) 真实诊断：写临时 .py 文件（未定义名） ----------
    py_path = os.path.join(tmpdir, "sample.py")
    with open(py_path, "w") as f:
        f.write("def foo():\n    return undefined_name\n")  # pyflakes: undefined name
    py_diag = mgr.hint(py_path)
    print("\n[py 诊断] =>", repr(py_diag[:300]))
    check("pylsp 对未定义名产出诊断", bool(py_diag) and ("L" in py_diag), "pylsp 未返回诊断" if not py_diag else "")

    # ---------- 5) 清理 ----------
    mgr.close_all()
    for p in (go_path, py_path):
        try:
            os.remove(p)
        except Exception:
            pass
    try:
        os.rmdir(tmpdir)
    except Exception:
        pass

    print("\n" + ("全部通过 ✅" if not fails else f"失败 {len(fails)} 项: {fails}"))
    assert not fails, f"{len(fails)} checks failed"


if __name__ == "__main__":
    main()


def test_main():
    main()
