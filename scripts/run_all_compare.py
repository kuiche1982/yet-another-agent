#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比驱动器（run_all_compare.py）
================================
顺序跑 4 个对比单元（大脑恒为 glm-4.7）：
    ling   × new
    ling   × bugfix
    qwen   × new
    qwen   × bugfix

每个单元是独立子进程（run_compare_one.py），进程间用 pkill 清掉残留的
rapid-mlx 服务，确保下一个单元加载的是正确权重（否则 ensure_server 会复用
端口上已有的错误权重服务）。

每格日志：compare_<model>_<task>.log
汇总结果：compare_results.jsonl
"""
import subprocess
import sys
import time
import os
from swe_agent.log import logger

REPO = "~/kuiwork/workdir2/litertlm"
os.chdir(REPO)

CELLS = [
    ("ling", "new"),
    ("ling", "bugfix"),
    ("qwen", "new"),
    ("qwen", "bugfix"),
]


def kill_stale_server():
    """清掉任何遗留的 rapid-mlx 服务，避免复用错误权重。"""
    subprocess.run("pkill -f 'rapid-mlx serve' 2>/dev/null || true", shell=True)
    time.sleep(3)


def run_cell(model: str, task: str):
    kill_stale_server()
    log = os.path.join(REPO, "scripts", f"compare_{model}_{task}.log")
    logger.info('%s', f"\n===== [{model}/{task}] 开始 {time.strftime('%H:%M:%S')} =====")
    with open(log, "w") as f:
        r = subprocess.run(
            [".venv/bin/python", "scripts/run_compare_one.py", model, task],
            stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, cwd=REPO,
        )
    logger.info('%s', f"===== [{model}/{task}] 结束 exit={r.returncode} {time.strftime('%H:%M:%S')} -> {log} =====")
    return r.returncode


def main():
    # 清空上次的汇总结果
    res_path = os.path.join(REPO, "scripts", "compare_results.jsonl")
    open(res_path, "w").close()

    overall_t0 = time.time()
    for model, task in CELLS:
        run_cell(model, task)
    total = round(time.time() - overall_t0, 1)

    logger.info('%s', '\n==================== 汇总 ====================')
    logger.info('%s', f'总耗时：{total}s')
    logger.info('%s', '各格明细见 compare_results.jsonl：')
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
