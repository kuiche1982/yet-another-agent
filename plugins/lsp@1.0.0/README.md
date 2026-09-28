# lsp 插件（config-only）

本插件**不含任何工具代码**，只负责向 harness 声明一组 LSP 语言服务器。
harness 的 `load_plugins()` 读取 `.lsp.json` 后，会按 `languageId` 填充
`swe_agent.lsp.PLUGIN_LSP_SERVERS`；之后每次写文件 / 调 `lsp` 动作，都会惰性拉起
对应服务器并把诊断回灌给模型。

## 已声明服务器

| languageId | 命令 | 备注 |
|---|---|---|
| python | `.venv/bin/pylsp` | 项目 venv 内已装（python-lsp-server 1.15.0）；相对路径由 harness 按 `REPO_ROOT` 解析 |
| go | `gopls` | 需 `go install golang.org/x/tools/gopls@latest`（本机已装） |
| rust | `rust-analyzer` | 需随 Rust 工具链安装（本机已装） |
| c / cpp | `clangd` | 随 LLVM/Homebrew 安装 |
| typescript / typescriptreact / javascript / javascriptreact | `typescript-language-server --stdio` | 需 `npm i -g typescript-language-server typescript` |

## 扩展方式

- 新增语言：在 `.lsp.json` 加一条 `{ "languageId": "...", "command": "...", "args": [...] }`。
- 全语言统一服务器：`LSP_SERVER_CMD` 环境变量（优先级低于各语言声明）。
- 所有命令解析：含 `/` 的相对路径按 `REPO_ROOT` 展开，`~` 展开为用户目录，纯命令名走 PATH。
