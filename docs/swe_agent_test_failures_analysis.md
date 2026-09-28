# swe_agent 既有测试失败分析（grounded）

> 结论先行：所谓「全量 10 个失败」现状实际只有 **4 个真实失败**（另有 2 个 `test_compact*.py` 只是 `pytest.skip`，非失败）。  
> 4 个全部是**测试引用了已被 3 层 loop 重构删除/重命名的符号**，与 contextmgr / LSP 改造无关，属既有失败，非本次引入。

## 验证方法

- 逐一对 `tests/test_*.py` 跑 `.venv/bin/python tests/<x>.py`（这些是离线 verify 脚本，自带 `main()`）。
- 对失败符号用 `git log -S <symbol> -- swe_agent/` 追溯删除点。
- 确认本次改动只触及 `contextmgr/*` 与 `swe_agent/lsp.py`（仅新增 capabilities/方法），未触碰 supervisor/harness。

## 失败清单（4 个）

| 测试                               | 报错                                                                                                 | 根因                                                                                                                                                   | 删除/重命名 commit                                                 |
| -------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `tests/test_lsp.py`              | `AttributeError: module 'swe_agent.supervisor' has no attribute '_PLUGIN_STUB_ACTIONS'`            | 桩列表 `_PLUGIN_STUB_ACTIONS` 在 3-loop 重构中被移除；该测试仍断言 `"lsp" not in S._PLUGIN_STUB_ACTIONS`                                                              | `e2507be` 3loop refact swe_agent                              |
| `tests/test_plugins.py`          | `AttributeError: ... no attribute 'agents_prompt_section' (Did you mean: '_tools_prompt_section')` | 函数 `agents_prompt_section` 已重命名为 `_tools_prompt_section`                                                                                             | 重命名发生在 3-loop 重构期间（原符号历史见 `git log -S agents_prompt_section`） |
| `tests/test_websearch_plugin.py` | 同上 `agents_prompt_section`                                                                         | 同上（同一旧符号名）                                                                                                                                           | 同上                                                            |
| `tests/test_hidden_tests.py`     | `AttributeError: module 'swe_agent.harness' has no attribute 'materialize_hidden_tests'`           | `materialize_hidden_tests` / `_run_hidden_tests` 在 3 层 loop 重构中整体移除；测试仍调用 `H.materialize_hidden_tests()` / *`H`*`._run_hidden_tests()`（line 64 / 70） | `1b9954b` 3层loop重构                                            |

## 非失败项（澄清误报）

- `tests/test_compact.py`、`tests/test_compact_paths.py`：脚本内 `pytest.skip("需要本地 LLM 服务（127.0.0.1:8000）...")`，属**跳过**非失败；`estimate_tokens` 阈值逻辑已静态通过。

## 建议修法（按风险排序）

1. **`test_plugins.py` / `test_websearch_plugin.py`（低风险，纯测试侧改名）**
   - 把 `S.agents_prompt_section(...)` 改为 `S._tools_prompt_section(...)`。
   - 需确认函数签名/返回值语义一致（当前 `_tools_prompt_section() -> str`，返回 prompt 段落字符串，与测试预期一致）。
   - 顺手把 `supervisor.py:420` 的陈旧注释 `agents_prompt_section` 改成 `_tools_prompt_section`，消除误导。
2. **`test_lsp.py`（中低风险，断言已过时）**
   - 该测试 #1/#2 已验证 `lsp` 是真实动作（`run.__name__ == "_m_lsp"` ✅）。`_PLUGIN_STUB_ACTIONS` 的桩列表概念已不存在，这条断言是死代码。
   - 修法：删除该 `check("lsp 已不在桩列表", ...)` 一行；或若仍想守「lsp 非桩」语义，改为检查当前 stub 机制（如有 `is_plugin_stub` 之类 API）。
3. **`test_hidden_tests.py`（中高风险，需先定位逻辑去向）**
   - `materialize_hidden_tests` / `_run_hidden_tests` 在 `1b9954b` 被整体移除。修法取决于现状：
     - 若隐藏测试能力已迁移到 supervisor / 新模块 → 更新 import 与调用名；
     - 若能力已砍掉 → 该测试应整体下线或改写以匹配新契约。
   - **此条不要盲改**：先 `git show 1b9954b -- swe_agent/harness.py` 看清删除了什么、替代实现在哪，再决定测试是改还是删。

## 与本次 contextmgr/LSP 改造的关系

- 本次改动文件：`contextmgr/symbols.py`（新增 SymbolProvider 双路）、`contextmgr/distill.py`（接 provider）、`contextmgr/manager.py`（接 symbol_provider + `ingest_code_file`）、`swe_agent/lsp.py`（新增 documentSymbol/references/callHierarchy 能力与死锁修复）。
- 4 个失败均不依赖上述任何改动；在改动前已存在（commit 早于本次工作）。属独立技术债，建议单独一轮处理，不要混进 contextmgr 提交。

## 修复记录（2026-09-05，commit 见 `git log`）

经 `git show 1b9954b` 确认：隐藏测试功能被**有意取代**（删 `materialize_hidden_tests`/`_run_hidden_tests`/`_hidden_failure_feedback`，新增 `_run_test_bar` 验证机制）；子智能体目录从 executor 系统提示移出（改 `run_subagent` 程序派发）；工具引导词（含 web_search 联网引导）从系统提示移除（工具经 ToolRegistry/API 对模型可见）。因此 4 个失败分两类处理：

1. **`tests/test_lsp.py`** — 删掉过时 `check("lsp 已不在桩列表", "lsp" not in S._PLUGIN_STUB_ACTIONS)`（“lsp 是真实动作”已由前两条断言覆盖，桩列表概念已不存在）。
2. **`tests/test_plugins.py`** — `agents_prompt_section` 改名为 `_tools_prompt_section`（语义变为工具清单段）；把过时断言改为 `_sup._tools_prompt_section()` 返回含「可用工具」的 str。原「系统提示含子智能体清单 + my-agent」断言删除（行为已移除；子智能体注册仍由上方 `SUBAGENT_PROMPTS` 断言覆盖）。
3. **`tests/test_websearch_plugin.py`** — 同上改名；删掉「系统提示含联网引导词」3 条断言 + `WEB_SEARCH_GUIDANCE` 开关断言（引导词已从系统提示移除，`WEB_SEARCH_GUIDANCE` 现成死配置）。保留 `plugins_prompt_section` 列出 `web_search` 的有效断言。
4. **`tests/test_hidden_tests.py`** — 整个文件删除（被测功能已被 `_run_test_bar` 取代，无替代测试对象）。

修复后：standalone 全量扫描 4 个原失败 + 其余 swe_agent 测试全部 OK；`test_compact*.py` 仍 `pytest.skip`（需本地 LLM 127.0.0.1:8000，非失败）；`tests/test_contextmgr_*.py` 仍 **60 passed**。

> 待办（非测试问题）：`swe_agent/config.py:69` 的 `WEB_SEARCH_GUIDANCE` 已成死代码（无任何读取点），若不再需要联网引导词应一并删除该配置与 `supervisor.py` 相关陈旧注释（`supervisor.py:420` 注释仍写 `agents_prompt_section`）。
