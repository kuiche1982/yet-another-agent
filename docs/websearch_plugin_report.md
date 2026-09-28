# WebSearch 插件 + 插件系统验证报告

验证时间：2026-08-30 | 模式：离线为主 + 一次 Zhipu 实跑

## 四项目标达成情况

### 目标 1：添加 websearch 能力 ✅
- `swe_agent/web_search.py`：封装 Zhipu Web Search API（`POST {GLM_BASE_URL}/web_search`，复用 `GLM_API_TOKEN`）。
- `tools.py` 的 `web_search` 工具优先走 Zhipu，未配置 token / Zhipu 出错时回退 DuckDuckGo（`WEB_SEARCH_FALLBACK=1`）。
- 实跑确认返回结构化结果（标题/链接/摘要/发布时间）。

### 目标 2：harness 能够使用 plugins ✅
- 新建 claude-code 风格插件包 `plugins/websearch@1.0.0/`：
  - `.claude-plugin/plugin.json`（元数据）
  - `skills/web-research/SKILL.md`（联网检索技能）
  - `commands/websearch.md`（/websearch 命令）
  - `agents/researcher.md`（深度调研子智能体）
  - `plugins/installed_plugins.json` 指向安装路径。
- `load_plugins()` 真正载入该插件并汇入三层后端：技能 `websearch:web-research`、命令 `websearch:websearch` 进入 `ALL_SKILLS`，子智能体 `researcher` 注册到 `SUBAGENT_PROMPTS`。

### 目标 3：启动时 / 按需加载 ✅
- **启动时加载**：`supervisor.main()` 现在在 startup 调用 `_plugins.load_plugins(C.PLUGINS_ROOT, enable_mcp=C.ENABLE_PLUGIN_MCP)`（受 `ENABLE_PLUGINS` 开关控制）。默认插件根 `REPO_ROOT/plugins`，可用 `SWE_PLUGINS_ROOT` 覆盖。
- **按需加载**：新增 `reload_plugins` 工具动作（走真实 `ToolRegistry.dispatch`），模型可发 `{"action":"reload_plugins"}` 热加载，无需重启。

### 目标 4：提示词引导词 ✅
- `build_system_prompt()` 新增 `_websearch_guidance_section()`：当 `web_search` 工具可用且 `WEB_SEARCH_GUIDANCE=1` 时，注入「🌐 联网搜索能力（web_search）」引导段，明确告知「优先先搜索网络再动手」的适用场景；插件已加载时再提示 `websearch:web-research` 技能与 `researcher` 子智能体。
- 受控：关闭 `WEB_SEARCH_GUIDANCE` 后引导段消失（已验证非硬编码）。

## 验证方式
`test_websearch_plugin.py`（19 项全绿）：
- 目标1：工具注册 + 实跑
- 目标2/3a：模拟启动 `load_plugins` + 三后端汇入断言
- 目标3b：`reload_plugins` dispatch 断言
- 目标4：提示词含引导段 + 插件段 + 开关可控

## 配置项（config.py）
```
ENABLE_PLUGINS=1              # 总开关
PLUGINS_ROOT=REPO/plugins    # 插件根（SWE_PLUGINS_ROOT 可覆盖）
ENABLE_PLUGIN_MCP=0          # 是否连插件 MCP（--enable-plugin-mcp 开启）
WEB_SEARCH_GUIDANCE=1        # 提示词引导词开关
```

## 改动文件
- 新增：`swe_agent/web_search.py`、`plugins/websearch@1.0.0/**`、`plugins/installed_plugins.json`、`test_websearch_plugin.py`
- 修改：`swe_agent/config.py`（4 个新常量）、`swe_agent/supervisor.py`（启动加载 + reload_plugins 工具 + 引导段）、`swe_agent/tools.py`（`reload_plugins` 动作）
