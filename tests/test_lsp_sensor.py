#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""test_lsp_sensor.py —— 验证 LspDiagnosticSensor 优先使用 LSP 插件（swe_agent.lsp），
当未配置服务器时回退到可用性探测。全部离线、无模型依赖。
"""
import os
import tempfile
from pathlib import Path

import swe_agent.config as config
import swe_agent.lsp as lsp
from swe_agent.harness import LspDiagnosticSensor
from swe_agent.registry import HarnessContext


def _setup_module():
    # 确保重置单例，使 is_configured 重新读取环境变量
    lsp.LSP = None


def _pylsp_exe() -> str:
    return str(config.REPO_ROOT / ".venv" / "bin" / "pylsp")


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


def main():
    _setup_module()
    pylsp = _pylsp_exe()
    have_pylsp = os.path.exists(pylsp)

    # ---- 1) 未配置任何 LSP 服务器：传感器回退到可用性探测 ----
    os.environ.pop("LSP_SERVER_CMD", None)
    lsp.LSP = None
    lsp.PLUGIN_LSP_SERVERS.clear()
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a.py").write_text("x = 1\n", encoding="utf-8")
    fact = LspDiagnosticSensor().run(HarnessContext(workspace=tmp))
    check("未配置时 manager.is_configured()=False", lsp.get_lsp().is_configured() is False)
    check("未配置时传感器回退 note（含 skipped/available）",
          ("skipped" in fact.message or "available" in fact.message))
    check("未配置时 severity=info", fact.severity == "info")

    # ---- 2) 配置 LSP 插件（LSP_SERVER_CMD）：传感器真正跑诊断 ----
    if have_pylsp:
        os.environ["LSP_SERVER_CMD"] = pylsp
        lsp.LSP = None
        bad = tmp / "bad.py"
        bad.write_text("import os\nx = y + 1\n", encoding="utf-8")  # 未定义名 y
        fact2 = LspDiagnosticSensor().run(HarnessContext(workspace=tmp))
        check("配置后 manager.is_configured()=True", lsp.get_lsp().is_configured() is True)
        check("配置后传感器产出 warning（发现诊断）", fact2.severity == "warning")
        check("诊断消息包含坏文件名 bad.py", "bad.py" in fact2.message)
        check("诊断 payload 记录了问题文件",
              fact2.payload.get("files") and "bad.py" in fact2.payload["files"][0])
        check("诊断内容含 pylsp 报的未定义名 y", "y" in fact2.message)
        # 清理缓存客户端
        lsp.get_lsp().close_all()
    else:
        print("[SKIP] 环境无 pylsp，跳过真实诊断断言")

    # ---- 3) 环境变量 LSP_SERVER_CMD = 通用服务器，对任意扩展名都可用 ----
    os.environ["LSP_SERVER_CMD"] = pylsp if have_pylsp else "pylsp"
    lsp.LSP = None
    lsp.PLUGIN_LSP_SERVERS.clear()
    check("env 模式 manager.is_configured()=True", lsp.get_lsp().is_configured() is True)
    check("available_for('x.py')=True（有服务器）", lsp.get_lsp().available_for("x.py") is True)
    check("available_for('x.unknown_ext')=True（env 为通用服务器）",
          lsp.get_lsp().available_for("x.unknown_ext") is True)

    # ---- 4) 仅插件声明 python 服务器：识别 is_configured，且按语言判断 ----
    os.environ.pop("LSP_SERVER_CMD", None)
    lsp.LSP = None
    lsp.PLUGIN_LSP_SERVERS["python"] = {"command": "pylsp", "args": []}
    check("插件声明 python 服务器后 is_configured()=True",
          lsp.get_lsp().is_configured() is True)
    check("插件声明后 available_for('x.py')=True", lsp.get_lsp().available_for("x.py") is True)
    check("插件未声明 other 语言时 available_for('x.unknown_ext')=False",
          lsp.get_lsp().available_for("x.unknown_ext") is False)
    lsp.PLUGIN_LSP_SERVERS.clear()

    print("\nALL LSP-SENSOR CHECKS PASSED")


if __name__ == "__main__":
    main()
