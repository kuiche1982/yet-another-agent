---
name: web-fetch
description: 网页抓取阅读技能——当需要根据 URL 阅读某页面的正文/文档内容时使用 web_fetch。
when_to_use: 需要抓取并阅读某个具体网页/文档的正文时使用。
allowed-tools: web_fetch, read_file, shell
---
# 网页抓取阅读技能（web-fetch）

你拥有 `web_fetch` 动作（参数 `url`，可选 `return_format`/`timeout` 等）。本技能指导你在「已知目标 URL、要读正文」时使用它，而不是泛泛搜索。

## 何时用
- 已知某个文档/博客/README 的链接，要读其正文；
- 搜索结果里挑出最相关的一条，深入阅读原文；
- 需要把网页内容转成 markdown 喂给后续步骤。

## 怎么用
1. 调用 `{"action":"web_fetch","url":"<目标链接>"}`；
2. 默认返回 markdown 正文（标题/来源/描述 + 正文，正文过长会被截断至 12000 字）；
3. 需要纯文本可传 `"return_format":"text"`；
4. Zhipu 不可用时自动回退直接抓取（结果较粗糙，仅作兜底）。

## 注意
- 抓取结果是辅助，最终以本地代码/测试为准；
- 不要对需要登录或反爬严格的站点强求。
