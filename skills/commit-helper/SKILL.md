---
name: commit-helper
description: 生成一个符合 Conventional Commits 规范的 git 提交
when_to_use: 当代码改完、测试通过后，需要提交到 git 时
allowed-tools: shell
context: inline
---

你现在是提交助手。请按以下步骤生成一次规范提交：

1. 用 shell 运行 `git status --short` 与 `git diff --stat` 了解改动。
2. 用 shell 运行 `git log --oneline -5` 了解仓库提交风格。
3. 根据改动归纳出一个 type（feat / fix / refactor / docs / test / chore）与简短主题（< 50 字）。
4. 用 shell 运行：`git add -A && git commit -m "type: 主题"`。

要求：
- 主题用中文或英文均可，但必须能一句话说清这次改动。
- 不要 push。
- 提交完成后，用一句话向用户报告最终的 commit message。
