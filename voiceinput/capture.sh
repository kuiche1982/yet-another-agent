#!/usr/bin/env bash
# voiceinput/capture.sh — ffmpeg 录音 / 截取 / 转 16k wav 工具
# 用法：
#   ./capture.sh list                         列出可用录音设备
#   ./capture.sh rec <秒数> [输出.wav]         录制麦克风（默认 30 秒，16k 单声道）
#   ./capture.sh cut <输入> <起始> <时长> <输出.wav>   截取音频片段（起始/时长支持 秒 或 00:00:05）
#   ./capture.sh to16k <输入> <输出.wav>      任意音频转 16kHz 单声道 wav
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  list)
    ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1
    ;;
  rec)
    SEC="${2:-30}"
    OUT="${3:-audio/mic.wav}"
    echo "录制 ${SEC}s 到 ${OUT}（可用 Ctrl+C 提前结束）..."
    ffmpeg -y -hide_banner -loglevel warning \
      -f avfoundation -i ":0" -t "$SEC" -ar 16000 -ac 1 -c:a pcm_s16le "$OUT"
    echo "完成: $OUT"
    ;;
  cut)
    [ "$#" -ge 5 ] || { echo "用法: cut <输入> <起始> <时长> <输出.wav>"; exit 1; }
    IN="$2"; START="$3"; DUR="$4"; OUT="$5"
    ffmpeg -y -hide_banner -loglevel warning \
      -ss "$START" -t "$DUR" -i "$IN" -ar 16000 -ac 1 -c:a pcm_s16le "$OUT"
    echo "截取完成: ${OUT}（${START} 起 ${DUR} 长）"
    ;;
  to16k)
    [ "$#" -ge 3 ] || { echo "用法: to16k <输入> <输出.wav>"; exit 1; }
    ffmpeg -y -hide_banner -loglevel warning -i "$2" -ar 16000 -ac 1 -c:a pcm_s16le "$3"
    echo "转换完成: $3"
    ;;
  *)
    echo "用法: $0 {list|rec|cut|to16k} ..."
    exit 1
    ;;
esac
