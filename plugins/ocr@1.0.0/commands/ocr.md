---
name: ocr
description: 对给定图片路径做 OCR 文字识别。
allowed-tools: ocr
---
# /ocr 命令

对用户提供的图片做 OCR 识别：

1. 取用户输入作为 `image_path`（本地图片路径，PNG/JPG/JPEG/BMP，≤8M）；
2. 调用动作 `{"action":"ocr","image_path":"<图片路径>"}`；
3. 把识别出的文字逐行返回。

识别失败时如实告知用户并建议换图片或检查格式/大小。
