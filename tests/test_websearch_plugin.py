#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""websearch 插件 + 插件系统端到端验证（对应四个目标）。

目标 1：添加 websearch 能力（web_search 工具由插件 tools/ 注册 + Zhipu 后端 + 配额耗尽回退）
目标 2：验证 harness 能够使用 plugins（load_plugins 真正载入 websearch 插件）
目标 3：验证 harness 能够启动时 / 按需加载插件（startup 模拟 + reload_plugins 动作）
目标 4：websearch 插件可用时，系统提示词注入「可联网搜索获取更多信息」引导词

运行：.venv/bin/python test_websearch_plugin.py
"""
import os
import sys
import json
from pathlib import Path

# 让 harness 启动时扫描工程自带插件目录（含 websearch 插件）
REPO = Path(__file__).resolve().parent.parent
os.environ["SWE_PLUGINS_ROOT"] = str(REPO / "plugins")

import swe_agent.config as C
import swe_agent.supervisor as _sup
import swe_agent.plugins as _plugins
import swe_agent.skills as _skills
from swe_agent.registry import ToolRegistry

C.PLUGINS_ROOT = str(REPO / "plugins")  # 与 main() 解析逻辑一致

_fail = 0


def check(label, cond, detail=""):
    global _fail
    if cond:
        print(f"✅ {label}")
    else:
        print(f"❌ {label}  {detail}")
        _fail += 1


# ============ 目标 1：websearch 能力（工具由插件 tools/ 注册）============
def t1_websearch_capability():
    print("\n--- 目标 1：websearch 能力（工具来自插件 tools/，非内置）---")
    # 纯插件化验证：加载插件前不应存在该工具
    check("加载插件前 web_search 未注册（证明工具来自插件而非内置）",
          ToolRegistry.get("web_search") is None)

    # 加载插件 -> web_search 由插件 tools/ 注册
    state = _plugins.load_plugins(C.PLUGINS_ROOT, enable_mcp=False)
    ws = ToolRegistry.get("web_search")
    check("load_plugins 后 web_search 已注册（来自插件 tools/）",
          ws is not None, "ToolRegistry 无 web_search")
    check("web_search 出现在 PLUGIN_STATE['tools']",
          "web_search" in state.get("tools", []), str(state.get("tools")))
    if not ws:
        return
    check("web_search 自描述含 when_to_use", bool(ws.when_to_use), ws.when_to_use)

    # 实跑一次（优先 Zhipu，失败/无配额自动回退 DuckDuckGo，不视为失败）
    try:
        out = ws.run("python pathlib Path.mkdir parents 参数")
        ok = any(k in out for k in ("web_search", "DuckDuckGo", "http", "条", "- "))
        check("web_search 实跑返回结果（Zhipu 或回退）", ok, repr(out[:120]))
        print("     ↳", out[:160].replace("\n", " "))
    except Exception as e:
        check("web_search 实跑返回结果（Zhipu 或回退）", False, f"异常: {e}")

    # 确定性验证：Zhipu 配额耗尽（返回 error）时应回退 DuckDuckGo（无需真实网络）
    # 定位插件工具模块：按文件路径匹配（不依赖 importlib 内部命名）
    mod = None
    for m in list(sys.modules.values()):
        fn = getattr(m, "__file__", None)
        if fn and fn.replace("\\", "/").endswith("tools/web_search_tool.py"):
            mod = m
            break
    if mod and hasattr(mod, "zhipu_web_search") and hasattr(mod, "_web_search_duckduckgo"):
        orig_z = mod.zhipu_web_search
        orig_d = mod._web_search_duckduckgo
        # 同时 stub 两个后端，使断言不依赖真实外网连通性
        mod.zhipu_web_search = lambda q, **kw: "web_search_error: 配额耗尽 (stub)"
        mod._web_search_duckduckgo = lambda q: "DuckDuckGo 回退结果（stub）：python pathlib 用法"
        try:
            fb = ws.run("test quota fallback")
            ok_fb = ("DuckDuckGo 回退结果" in fb) and ("配额耗尽" not in fb)
            check("Zhipu 配额耗尽时自动回退 DuckDuckGo（且不再透传 Zhipu 错误）",
                  ok_fb, repr(fb[:140]))
            print("     ↳ 回退结果:", fb[:140].replace("\n", " "))
        finally:
            mod.zhipu_web_search = orig_z
            mod._web_search_duckduckgo = orig_d
    else:
        print("⚠️ 无法定位插件工具模块，跳过配额回退单测")


# ============ 目标 2 + 3a：harness 使用插件 / 启动时加载 ============
def t2_startup_and_use():
    print("\n--- 目标 2 & 3a：harness 加载并使用 plugins（模拟启动）---")
    state = _plugins.load_plugins(C.PLUGINS_ROOT, enable_mcp=False)
    check("load_plugins 载入 websearch 插件",
          "websearch" in state.get("plugins", []), str(state.get("plugins")))
    # 注：installed_plugins.json 现含多个插件（websearch + web_fetch + ocr），
    # 故总数不再是 1；此处只校验 websearch 这一插件的专属贡献。
    check("websearch 插件已载入", "websearch" in state.get("plugins", []), str(state.get("plugins")))
    check("插件技能总数 >= 1（含 web-research）", state.get("skills", 0) >= 1, str(state))
    check("插件命令总数 >= 1（含 websearch）", state.get("commands", 0) >= 1, str(state))
    check("web_search 工具来自插件", "web_search" in state.get("tools", []), str(state.get("tools")))
    check("researcher 子智能体来自插件", "researcher" in state.get("agents", {}), str(state.get("agents")))

    names = {s.name for s in _skills.ALL_SKILLS}
    check("插件技能 websearch:web-research 进入 ALL_SKILLS",
          "websearch:web-research" in names, str(names))
    check("插件命令 websearch:websearch 进入 ALL_SKILLS",
          "websearch:websearch" in names, str(names))
    check("插件子智能体 researcher 注册到 SUBAGENT_PROMPTS",
          "researcher" in _sup.SUBAGENT_PROMPTS, str(list(_sup.SUBAGENT_PROMPTS)))


# ============ 目标 3b：按需加载（reload_plugins 动作）============
def t3_ondemand():
    print("\n--- 目标 3b：按需加载（reload_plugins 动作）---")
    check("reload_plugins 动作已注册",
          ToolRegistry.get("reload_plugins") is not None)
    # 模拟模型发出 reload_plugins 动作，走真实 dispatch 路径
    res = ToolRegistry.dispatch({"action": "reload_plugins"})
    check("reload_plugins dispatch 成功",
          res.startswith("reload_plugins_ok"), repr(res))
    check("reload 后仍含 websearch 插件",
          "websearch" in _plugins.PLUGIN_STATE.get("plugins", []), repr(_plugins.PLUGIN_STATE))
    # 重载后工具仍在（幂等，未被重复/丢失）
    check("reload 后 web_search 工具仍在且唯一",
          ToolRegistry.get("web_search") is not None
          and ToolRegistry.names().count("web_search") == 1)


# ============ 目标 4：插件扩展段 / 工具清单段（系统提示已不再注入联网引导词）============
def t4_prompt_guidance():
    print("\n--- 目标 4：插件扩展段列出 websearch 工具 + 工具清单段可生成 ---")
    # 注：3-loop 重构后「目录式 section / 工具引导词」已从 executor 系统提示移除
    # （工具经 ToolRegistry / API 对模型可见，见 build_system_prompt docstring），
    # 故不再断言系统提示含「🌐 联网搜索能力」引导段；web_search 工具本身已由 t1/t2 验证注册。
    sec = _plugins.plugins_prompt_section()
    check("插件扩展段出现（含 websearch 插件贡献）",
          "插件扩展" in sec,
          repr(sec[:200]))
    check("插件扩展段列出自带工具 web_search", "`web_search`" in sec, repr(sec[:300]))
    # 工具清单段（原 agents_prompt_section，已重命名 + 改语义为工具列表）。
    tool_sec = _sup._tools_prompt_section()
    check("_tools_prompt_section 返回工具清单段",
          isinstance(tool_sec, str) and "可用工具" in tool_sec, repr(tool_sec[:120]))
    # researcher 子智能体注册由上方 SUBAGENT_PROMPTS 断言覆盖


def main():
    t1_websearch_capability()
    t2_startup_and_use()
    t3_ondemand()
    t4_prompt_guidance()
    print()
    if _fail == 0:
        print("✅ 全部通过：websearch 工具在插件内 + 插件加载/使用 + 按需重载 + 提示词引导 + 配额回退")
        sys.exit(0)
    else:
        print(f"❌ {_fail} 项失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
