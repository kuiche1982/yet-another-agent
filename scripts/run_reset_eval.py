#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史重置验证驱动器（run_reset_eval.py）
======================================
验证「任务完成时重置 executor 历史」对本地小模型的影响：
对比 ling-3.0-tiny  vs  LFM2.5-1.2B（两者都走本地 rapid-mlx / TEXT_JSON，公平）。
每格用 CELL_MAX_ITER 控制轮数；重置开关由 config.EXECUTOR_RESET_ON_TASK_DONE 控制
（默认开）。重点观测：每轮 input 字符数、reset_events（任务完成触发次数）、收敛。
"""
import subprocess
import os
import time
import socket
from swe_agent.log import logger

REPO = "~/kuiwork/workdir2/litertlm"
os.chdir(REPO)
PY = os.path.join(REPO, ".venv", "bin", "python")
PORT = 8000

# 验证用：压低轮数，缩短单次耗时（仍足够触发多次任务完成→重置）
os.environ["CELL_MAX_ITER"] = os.environ.get("CELL_MAX_ITER", "12")

# ling / lfm(1.2B) 属同档极弱模型：多为「规划出多任务但执行不出动作/任务不完成」，
# 用于观察「弱模型下 reset 不触发、input 单任务内滚动增长」的基线；
# qwen-2.5-coder-7b 较强，能真正完成任务 → 触发 reset，演示 input 回落。三者同走 rapid-mlx，公平。
CELLS = [
    ("ling", "new"),
    ("lfm", "new"),
    ("qwen", "new"),
]


def kill_stale_server():
    subprocess.run("pkill -9 -f '[r]apid-mlx serve' 2>/dev/null || true", shell=True)
    subprocess.run("pkill -9 -f '[r]apid-mlx' 2>/dev/null || true", shell=True)
    for _ in range(10):
        s = socket.socket()
        if s.connect_ex(("127.0.0.1", PORT)) != 0:
            s.close()
            return
        s.close()
        time.sleep(3)
    logger.info('%s', '[eval] 警告：端口 8000 等待后仍被占用！')


def run_cell(model, task):
    kill_stale_server()
    log = os.path.join(REPO, f"reset_eval_{model}_{task}.log")
    logger.info('%s', f"\n===== [{model}/{task}] 开始 {time.strftime('%H:%M:%S')} =====")
    with open(log, "w") as f:
        r = subprocess.run(
            [PY, "-u", "run_compare_one.py", model, task],
            stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
            bufsize=1, cwd=REPO,
        )
    logger.info('%s', f"===== [{model}/{task}] 结束 exit={r.returncode} {time.strftime('%H:%M:%S')} -> {log} =====")
    return r.returncode


def main():
    # 清空旧结果，保证汇总干净
    rp = os.path.join(REPO, "compare_results.jsonl")
    open(rp, "w").close()
    overall_t0 = time.time()
    for model, task in CELLS:
        run_cell(model, task)
    total = round(time.time() - overall_t0, 1)

    logger.info('%s', '\n==================== 重置验证汇总 ====================')
    logger.info('%s', f"总耗时：{total}s | CELL_MAX_ITER={os.environ['CELL_MAX_ITER']}")
    try:
        with open(rp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = __import__("json").loads(line)
                logger.info('%s', f"  {d['model']:5} × {d['task']:6} | {d['verdict']:7} | 轮={d['rounds']:2} 重置={d['reset_events']} input总={d['input_chars_total']} 首轮={d['input_chars_first_round']} 峰值={d['input_chars_max']} 末轮={d['input_chars_last_round']} | {d['duration_s']}s")
    except Exception as e:
        logger.info('%s', f'读取汇总失败：{e}')


if __name__ == "__main__":
    main()
