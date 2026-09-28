Good question — let me verify the exact formats the code actually parses (rather than guess) before giving examples. I'll check the plugin LSP collector and the resolver.
Let me read the exact collector implementation and the CLI flag so my examples are precise.
Let me confirm whether `load_plugins()` (which reads `.lsp.json`) is auto-invoked at import or must be called explicitly, so my guidance is accurate.
`load_plugins()` isn't auto-called at import (unlike `reload_skills()`), so I need to verify whether the main runner invokes it — otherwise plugin `.lsp.json` would never load. Let me check.
LSP 服务器有 **3 种配置方式**，代码按「语言专属插件 > 环境变量统一服务器」的优先级解析（`swe_agent/lsp.py:_resolve`）。下面给例子，先说 supported 的 `languageId`：

**已支持的语言**（来自 `_EXT_TO_LANG`）：`python` `.py` · `javascript`/`javascriptreact` `.js/.jsx` · `typescript`/`typescriptreact` `.ts/.tsx` · `go` · `rust` · `java` · `c`/`cpp` · `csharp` · `ruby` · `php` · `swift` · `kotlin` · `lua` · `shellscript` `.sh/.bash/.zsh` · `json`/`html`/`css`/`scss` · `markdown` · `xml` · `sql` · `yaml` · `toml`

任何服务器命令需满足：在 `PATH` 或绝对路径可启动、且用 **stdio + Content-Length 分帧 JSON-RPC** 讲 LSP 协议（pylsp / pyright-langserver / typescript-language-server / gopls / rust-analyzer / ruff-lsp / clangd 等都行）。

---

### 方式 1：环境变量（统一服务器，最简单，无需 load_plugins）

```bash
# 单语言栈场景：所有文件都交给这一个服务器
export LSP_SERVER_CMD="pylsp"
# 带参数
export LSP_SERVER_CMD="python -m pylsp"
# TypeScript 栈
export LSP_SERVER_CMD="pyright-langserver --stdio"
```
或运行时通过 CLI（直接写进 `LSP_SERVER_CMD`）：
```bash
python -m swe_agent "<task>" --lsp-server "pylsp"
```
**优先级最低**（作为兜底通用服务器）。注意：它会对所有扩展名生效——一个 TS 服务器看不懂 `.py`，那种文件就静默无诊断（`hint` 返回空串，不报错）。

### 方式 2：插件 `.lsp.json`（按语言，推荐）

在插件目录 `~/.claude/plugins/<plugin>/` 下放 `.lsp.json`，**按 `languageId` 索引**：
```json
[
  {"languageId": "python",      "command": "pylsp", "args": []},
  {"languageId": "typescript", "command": "typescript-language-server", "args": ["--stdio"]},
  {"languageId": "go",          "command": "gopls", "args": []}
]
```
也接受 `{"lspServers": [...]}` 包裹格式。生效条件：文件扩展名在 `_EXT_TO_LANG` 里映射到该 `languageId`（如 `.py`→`python`）。

### 方式 3：插件清单 `plugin.json` 的 `lspServers`

`~/.claude/plugins/<plugin>/.claude-plugin/plugin.json`：
```json
{
  "name": "my-lang-plugin",
  "lspServers": [
    {"languageId": "python", "command": "/abs/path/to/.venv/bin/pylsp", "args": []}
  ]
}
```

---

### 优先级

`PLUGIN_LSP_SERVERS[languageId]`（方式 2/3）**高于** `LSP_SERVER_CMD`（方式 1）。即：`.py` 若有插件专属 pylsp，就用它；否则回落到环境变量统一服务器。

### 验证配置是否生效

```python
import swe_agent.lsp as lsp
mgr = lsp.get_lsp()
mgr.is_configured()            # True = 有任何服务器可用
mgr.available_for("x.py")      # True = .py 能映射到服务器
mgr.hint("path/to/file.py")    # 真实诊断串；无服务器/无诊断 → 空串（绝不崩）
```
或跑 `python test_lsp_sensor.py`（15 项会覆盖三种配置路径）。

---

### ⚠️ 一个关键前提（务必知道）

**插件式 LSP（方式 2/3）目前不会自动生效**——`load_plugins()` 在 `swe_agent` 里**没有在启动时自动调用**（只有 `skills/` 目录的 `reload_skills()` 是自动的）。所以：
- 方式 1（`--lsp-server` / 环境变量）开箱即用，不受限；
- 方式 2/3 必须在入口显式调用一次 `load_plugins()`，`.lsp.json`/`plugin.json` 才会被读进 `PLUGIN_LSP_SERVERS`。

要我把它接进启动流程（在 `supervisor` 初始化处加一行 `load_plugins()`）吗？这样插件式按语言 LSP 就能和 skills/MCP 一样自动加载。要的话我顺手补上并加一条测试。