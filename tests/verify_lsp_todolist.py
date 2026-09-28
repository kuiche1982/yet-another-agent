import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""用真实的 golang CLI todolist 验证 harness LSP（gopls）诊断能力。

步骤：
  1. load_plugins() 加载 lsp@1.0.0 插件（声明 go -> gopls）
  2. 对正确版 main.go 调 mgr.hint() -> 应为空（无诊断）
  3. 写一个含类型错误的探针文件，调 mgr.hint() -> 应抓到 gopls compiler 诊断
"""
import os
import sys

import swe_agent.config as C
import swe_agent.plugins as P
import swe_agent.lsp as L


def check(name, cond, detail=""):
    print(("✅ " if cond else "❌ ") + name + (f"  -> {detail}" if detail else ""))


def main():
    P.load_plugins(C.PLUGINS_ROOT, enable_mcp=False)

    proj = os.path.join(C.REPO_ROOT, "examples", "todolist_go")
    main_go = os.path.join(proj, "main.go")

    check("go 服务器已声明",
          "go" in L.PLUGIN_LSP_SERVERS,
          str(L.PLUGIN_LSP_SERVERS.get("go")))
    check("main.go 可被 LSP 接管",
          L.get_lsp().available_for(main_go), main_go)

    mgr = L.get_lsp()

    # ---- 1) 正确版：应无诊断（hint 返回空串）----
    diag_clean = mgr.hint(main_go)
    check("正确版 main.go 无诊断", diag_clean == "",
          diag_clean[:80] if diag_clean else "空")

    # ---- 2) 错误探针：应抓到 gopls 诊断 ----
    probe = os.path.join(proj, "diag_probe.go")
    probe_src = (
        "package main\n\n"
        "func probeBroken() {\n"
        "\tvar x int = \"hello\"   // 故意类型错误\n"
        "\t_ = x\n"
        "}\n"
    )
    with open(probe, "w", encoding="utf-8") as f:
        f.write(probe_src)
    try:
        diag_err = mgr.hint(probe)
        caught = bool(diag_err) and ("cannot use" in diag_err or "mismatch" in diag_err.lower())
        check("错误探针被 gopls 抓到", caught,
              (diag_err[:120].replace("\n", " ") if diag_err else "空"))
    finally:
        os.remove(probe)
        mgr.close_all()

    print("\nLSP 验证完成。")


if __name__ == "__main__":
    main()
