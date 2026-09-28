---
name: ocr
description: 图片 OCR 技能——当需要从图片/截图/扫描件中抽取文字时使用 ocr。
when_to_use: 需要从图片、截图或扫描件中抽取文字内容时使用。
allowed-tools: ocr, read_file, shell
---
# 图片 OCR 技能（ocr）

你拥有 `ocr` 动作（参数 `image_path`，可选 `language_type`/`probability`）。

## 何时用
- 用户发来截图、扫描件、拍照图片，要提取其中文字；
- 需要从图片里读取代码、表格、票据、文档内容。

## 怎么用
1. 确认图片为本地路径（PNG/JPG/JPEG/BMP，≤8M）；
2. 调用 `{"action":"ocr","image_path":"<图片路径>"}`；
3. 默认自动检测语言（language_type=AUTO），已知语言时指定可提升准确率；
4. 需要置信度时传 `"probability":true`。

## 注意
- 识别结果是辅助，重要信息需人工核对；
- 手写体建议深色墨迹、浅色背景以提升准确率。
