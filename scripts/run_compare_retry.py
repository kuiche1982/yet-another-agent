#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比重试驱动器（run_compare_retry.py）
====================================
第一版 run_all_compare 因「子进程继承了驱动器的坏 stdin fd」导致后 3 格在
Python 启动期就崩（init_sys_streams: Bad file descriptor）。本驱动器修复：
  - 子进程显式 stdin=subprocess.DEVNULL，彻底隔离驱动器的 fd 0；
  - 每格启动前 pkill rapid-mlx 并**轮询确认 8000 端口空闲**，确保本格加载的是
    正确权重（避免 ensure_server 复用端口上已有的错误权重服务）。
  - 不截断 compare_results.jsonl（ling/new 的有效结果予以保留）。

只跑之前崩溃的 3 格：ling×bugfix / qwen×new / qwen×bugfix。
"""
import subprocess
import sys
import time
import os
import socket
from swe_agent.log import logger

REPO = "~/kuiwork/workdir2/litertlm"
os.chdir(REPO)
PY = os.path.join(REPO, ".venv", "bin", "python")
PORT = 8000

CELLS = [
    ("ling", "new"),
    ("ling", "bugfix"),
    ("qwen", "new"),
    ("qwen", "bugfix"),
]


def kill_stale_server():
    """清掉任何遗留的 rapid-mlx 服务，并等到 8000 端口真正空闲。

    用 [r]apid-mlx 正则技巧：pkill 自身的 shell 命令行里含字面量
    '[r]apid-mlx'，而正则 [r]apid-mlx 只匹配 'rapid-mlx'，因此不会误杀自己。
    """
    subprocess.run("pkill -9 -f '[r]apid-mlx serve' 2>/dev/null || true", shell=True)
    # 也顺手清掉可能残留的普通 rapid-mlx（非 serve 子进程），避免占用端口
    subprocess.run("pkill -9 -f '[r]apid-mlx' 2>/dev/null || true", shell=True)
    # 轮询端口，最多等 30s
    for _ in range(10):
        s = socket.socket()
        if s.connect_ex(("127.0.0.1", PORT)) != 0:
            s.close()
            return
        s.close()
        time.sleep(3)
    logger.info('%s', '[retry] 警告：端口 8000 在等待后仍被占用，本格可能复用错误权重！')


def run_cell(model: str, task: str):
    kill_stale_server()
    log = os.path.join(REPO, "scripts", f"compare_{model}_{task}.log")
    logger.info('%s', f"\n===== [{model}/{task}] 开始 {time.strftime('%H:%M:%S')} =====")
    with open(log, "w") as f:
        r = subprocess.run(
            [PY, "-u", "scripts/run_compare_one.py", model, task],
            stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
            bufsize=1, cwd=REPO,
        )
    logger.info('%s', f"===== [{model}/{task}] 结束 exit={r.returncode} {time.strftime('%H:%M:%S')} -> {log} =====")
    return r.returncode


def main():
    overall_t0 = time.time()
    for model, task in CELLS:
        run_cell(model, task)
    total = round(time.time() - overall_t0, 1)

    logger.info('%s', '\n==================== 重试汇总 ====================')
    logger.info('%s', f'总耗时：{total}s')
    res_path = os.path.join(REPO, "scripts", "compare_results.jsonl")
    try:
        with open(res_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = __import__("json").loads(line)
                logger.info('%s', f"  {d['model']:5} × {d['task']:6} | {d['verdict']:7} | 轮={d['rounds']:2} 执行调用={d['executor_calls']:3} glm调用={d['glm_calls']:2} | {d['duration_s']}s")
    except Exception as e:
        logger.info('%s', f'读取汇总失败：{e}')


if __name__ == "__main__":
    main()
