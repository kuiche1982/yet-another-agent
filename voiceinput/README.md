# voiceinput — MacBook 语音输入 → whisper.cpp 转写（中英夹杂）

目标：用 whisper.cpp **small** 模型识别中英文夹杂语音，输出结构化文本，供无人值守链路（如 swe_agent）消费。

## 目录

- `capture.sh` — ffmpeg 录音 / 截取 / 转 16k wav 工具
  - `./capture.sh list` 列出麦克风；`rec <秒> [out.wav]` 录音；`cut <输入> <起始> <时长> <输出>` 截取；`to16k <输入> <输出>` 转换
- `transcribe.sh` — small 模型转写，默认 `-l zh` + 简体中文提示词，输出 txt + 完整 JSON
  - `TR_LANG=auto` 切自动检测；`TR_MODEL=/path` 换模型；`TR_PROMPT=` 改提示词
- `models/ggml-small.bin` — 官方 ggml small 模型（465MB，来自 HF ggerganov/whisper.cpp，经 hf-mirror 下载）
- `audio/` — 测试音频（say TTS 生成，Tingting/Eddy 中文语音）
- `out/` — 转写结果：`<名>.txt`（纯文本）+ `<名>.json`（分段/时间戳/token 置信度）

## 安装配置（Setup）

前置依赖：Homebrew、`ffmpeg`、`say`（macOS 自带）。

```bash
# 1. 安装 whisper.cpp（本机 1.9.4；Apple Silicon 默认启用 Metal + FlashAttention）
brew install whisper-cpp

# 2. 建目录
mkdir -p voiceinput/{models,audio,out}

# 3. 下载官方 ggml small 模型（465MB）
#    huggingface.co 直连超时，本机走 hf-mirror：
curl -L -o voiceinput/models/ggml-small.bin \
  "https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
#    直连可用时也可用官方脚本：sh models/download-ggml-model.sh small

# 4. 脚本授权
chmod +x voiceinput/capture.sh voiceinput/transcribe.sh

# 5. 验证
whisper-cli --help | head -5        # 首次运行会编译 Metal 库（约 18s，之后有缓存）
whisper-bench -m voiceinput/models/ggml-small.bin   # 可选：跑性能基准
```

测试音频（可选，macOS TTS 生成中英夹杂样本）：

```bash
say -v Tingting -o t.aiff "明天早上 ten o'clock 开 standup meeting，先 review 一下代码。"
ffmpeg -y -i t.aiff -ar 16000 -ac 1 -c:a pcm_s16le t.wav
```

## 实测结果（M2，whisper.cpp 1.9.4，Metal + FlashAttention，8 线程）

| 文件 | 时长 | 转写耗时 | 倍速 | 平均置信度 | 备注 |
|---|---|---|---|---|---|
| t1 | 12.4s | 0.91s | ~13.6x | 0.82 | 中文为主，英文术语夹杂，识别良好 |
| t2 | 11.0s | 0.82s | ~13.5x | 0.81 | 同上；`i/o 的` 误听为 `i or the` |
| t3v2 | 11.5s | 0.83s | ~13.9x | 0.86 | 中英比例接近，结果最准 |
| t1_cut | 6.0s | 0.66s | ~9x | 0.75 | 截取片段脱离上下文，准确率下降 |

参考转写（t3v2）：`明天早上10:00开stand up meeting,先把代码review一遍,然后看看这个issue的status怎么样,ok的话就close掉。`

## 关键结论

1. **语言选择**：中文主导语音用 `-l zh` 比 `auto` 稳；短句且英文占比高时 `auto` 可能误判成英文，整段中文被英文化（实测 t3 快语速版）。
2. **提示词**：`--prompt "以下是简体中文的普通话。"` 稳定简体输出并提升准确率（否则偶发繁体）。
3. **截取**：`cut` 截出的片段失去上文，准确率明显下降；建议整句转写或配合 VAD。
4. **性能**：M2 上 small + Metal 达 13 倍实时，完全满足无人值守。
5. **提精度**：升级 medium 模型（磁盘 1.5G、约 2-3 倍慢，仍超实时）、调 `-bs 8`、或控制语速。

## 无人值守接线

录音 → 转写 → 写任务文件 → watcher（launchd/目录监听）→ `uv run --no-sync python -m swe_agent "$(cat 任务文件)"` → 结果通知。
