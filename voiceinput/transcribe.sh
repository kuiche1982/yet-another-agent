#!/usr/bin/env bash
# voiceinput/transcribe.sh — whisper.cpp small 模型转写（中英夹杂，高准确率配置）
# 用法：
#   ./transcribe.sh <音频1.wav> [音频2.wav ...]
#   ./transcribe.sh audio/t1.wav audio/t2.wav   # 逐个转写
#   TR_LANG=zh ./transcribe.sh audio/t1.wav      # 指定语言（默认 auto 自动检测）
# 输出：out/<音频名>.txt（纯文本） + out/<音频名>.json（结构化，含分段/时间戳/置信度）
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${WHISPER_MODEL:-models/ggml-small.bin}"
WHISPER_BIN="${WHISPER_BIN:-/opt/homebrew/bin/whisper-cli}"
LANG_SET="${TR_LANG:-auto}"
TR_PROMPT="${TR_PROMPT:-以下是简体中文的普通话。}"
THREADS="${THREADS:-8}"
OUTDIR="out"
mkdir -p "$OUTDIR"

[ -f "$MODEL" ] || { echo "缺少模型: $MODEL（先运行 models/download-model.sh）"; exit 1; }
[ -x "$WHISPER_BIN" ] || { echo "缺少 whisper-cli: $WHISPER_BIN"; exit 1; }
[ "$#" -ge 1 ] || { echo "用法: $0 <音频1.wav> [...]"; exit 1; }

for f in "$@"; do
  [ -f "$f" ] || { echo "跳过（不存在）: $f"; continue; }
  base="$(basename "$f" .wav)"
  echo "==> 转写 $f (lang=$LANG_SET, threads=$THREADS)"
  "$WHISPER_BIN" -m "$MODEL" -f "$f" \
    -l "$LANG_SET" -t "$THREADS" -fa \
    --prompt "$TR_PROMPT" \
    -otxt -ojf \
    -of "$OUTDIR/$base" 2>&1 | grep -E "whisper_print_timings|error" | sed 's/^/    /'
  echo "    => $OUTDIR/$base.txt / $OUTDIR/$base.json"
done
