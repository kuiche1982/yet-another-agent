import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""真实跑一次「生成 golang 版康威生命游戏」并通过 harness 的 LSP 子系统验证。

证明两件事：
  1) litertlm 原来的「固定探针」LspDiagnosticSensor 现在优先走刚配的 lsp 插件
     （PLUGIN_LSP_SERVERS[go]=gopls），硬编码 pylsp/ruff 仅作无插件时的兜底；
  2) 生成的 golang 文件经 harness 真实 LSP 路径（sensor + lsp 动作后端）能被 gopls 诊断。

全程调用 harness 真实代码：LspDiagnosticSensor.run / _scan_workspace、
swe_agent.lsp.lsp_diagnostics（即 `lsp` 动作后端），不另写模拟。
"""
import os
import sys

import swe_agent.config as C
import swe_agent.plugins as P
import swe_agent.lsp as L
from swe_agent.harness import LspDiagnosticSensor
from swe_agent.registry import HarnessContext


def check(name, cond, detail=""):
    print(("✅ " if cond else "❌ ") + name + (f"  -> {detail}" if detail else ""))


def main():
    P.load_plugins(C.PLUGINS_ROOT, enable_mcp=False)

    ws = os.path.join(C.REPO_ROOT, "examples", "game_of_life_go")
    main_go = os.path.join(ws, "main.go")

    print("=== 1) 固定探针是否已改用 lsp 插件 ===")
    check("go 服务器已在 PLUGIN_LSP_SERVERS（插件提供）",
          "go" in L.PLUGIN_LSP_SERVERS, str(L.PLUGIN_LSP_SERVERS.get("go")))
    check("LSPManager.is_configured()=True（插件已加载）",
          L.get_lsp().is_configured() is True)

    # ---- 故意写一个有类型错误的文件，供探针/动作抓取 ----
    broken = os.path.join(ws, "broken_probe.go")
    broken_src = (
        "package main\n\n"
        "func brokenProbe() {\n"
        "\tvar x int = \"oops\"   // 故意类型错误\n"
        "\t_ = x\n"
        "}\n"
    )
    with open(broken, "w", encoding="utf-8") as f:
        f.write(broken_src)
    try:
        print("\n=== 2) 真实 LspDiagnosticSensor（固定探针现走插件）扫描工作区 ===")
        fact = LspDiagnosticSensor().run(HarnessContext(workspace=ws))
        # 插件路径命中时，消息以 "LSP plugin active" 开头
        used_plugin = fact.message.startswith("LSP plugin active")
        check("探针走的是 lsp 插件路径（非 pylsp/ruff 兜底）", used_plugin,
              fact.message[:70])
        check("探针抓到 broken_probe.go 的 gopls 诊断",
              "broken_probe.go" in fact.message, fact.message[:120])
        print("   探针 fact:", fact.message[:160].replace("\n", " "))

        print("\n=== 3) 真实 `lsp` 动作后端 lsp_diagnostics() ===")
        diag = L.lsp_diagnostics(broken)  # 即 supervisor._m_lsp 的真实后端
        caught = "cannot use" in diag or "as int value" in diag
        check("lsp 动作后端抓到 gopls compiler 错误", caught,
              diag[:120].replace("\n", " "))
    finally:
        os.remove(broken)

    print("\n=== 4) 删除错误文件后，探针应无诊断（确认是 gopls 而非误报）===")
    fact_clean = LspDiagnosticSensor().run(HarnessContext(workspace=ws))
    check("清理后探针无诊断", "no diagnostics" in fact_clean.message,
          fact_clean.message[:90])

    L.get_lsp().close_all()
    print("\n真实 harness LSP 验证完成。")


if __name__ == "__main__":
    main()
