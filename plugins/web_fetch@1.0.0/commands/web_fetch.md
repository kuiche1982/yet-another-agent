---
name: web_fetch
description: 抓取并阅读给定 URL 的网页正文。
allowed-tools: web_fetch
---
# /web_fetch 命令

对用户提供的链接抓取并阅读正文：

1. 取用户输入作为 `url`；
2. 调用动作 `{"action":"web_fetch","url":"<链接>"}`；
3. 把返回的正文要点整理返回，保留来源链接。

抓取失败时如实告知用户并建议换链接或换 web_search 检索。
