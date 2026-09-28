#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_e2e_battery.py —— litertlm SWE-Agent 小 e2e 电池（多任务顺序跑，30 分钟硬超时）。

用法：
    uv run --no-sync python run_e2e_battery.py
    uv run --no-sync python run_e2e_battery.py --tasks conway,fizzbuzz   # 只跑指定

行为：
  - 每个任务：清空 agent_sandbox → 启动 `python -m swe_agent <task>`（独立进程组）
  - 单任务硬超时 TIMEOUT 秒（默认 1800=30 分钟），超时则 killpg 整个进程组，判 timeout
  - 跑完后独立用 `python -m pytest -q` 校验工作区测试（与 harness 单杠标准一致：count>0 且全绿=pass）
  - 结果写入 logs/e2e_battery/summary.json，每个任务日志在 logs/e2e_battery/NN_name.log

环境变量（可按需覆盖）：
  SWE_MAX_ATTEMPTS / SWE_MAX_ROUNDS / SWE_MAX_STEPS / SWE_MODEL_RETRY / SWE_MODEL_BACKOFF
模型：**全部角色位（planner/analyzer/executor/tester/compact）的目标模型统一从
`swe_agent/config.py` 的 BUILD 常量读取**，本脚本不硬编码任何模型名（见 `_harness_target_models`）。
任务开始前 `reset_models()` 【强制】确认目标模型就位（与 SWE_MODEL_LOAD_UNLOAD 无关——该开关只管 per-loop 换模）：
缺失则加载、无关模型则卸载、无法就位则直接 fail-loud 退出，杜绝 LM Studio 用驻留模型静默顶替。
若要显式指定，用环境变量：PLANNER_MODEL / ANALYZER_MODEL / EXECUTOR_MODEL /
FORGE_TESTER_MODEL（TESTER_MODEL）/ SIDECAR_COMPRESS_MODEL。
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from swe_agent.log import logger

REPO = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO / "agent_sandbox"   # 隔离根：每个电池实例每个任务一个独立子目录
LOGDIR = REPO / "logs" / "e2e_battery"
TIMEOUT = 30 * 60  # 单任务硬超时（秒）

# 默认任务集（经典小题目，agent 自写实现+测试并跑通）
# 规范统一对齐 harness 契约（config.py:303）：测试放工作区【根目录】、用 `from src.x import` 导入，
# 不建 tests/ 子目录、也不依赖 conftest.py 绕路径——直接消除「代码在 src/ 但测试 from xxx 顶层导入」的导入坑。
DEFAULT_TASKS = [
    ("conway",
     "在 当前目录 用 Python 实现康威生命游戏：src/life.py 提供 init_grid(w,h)、step(grid) "
     "（标准 B3/S23 规则演化 Moore 邻居）、以及 main() CLI（if __name__=='__main__': main() 演化若干代）。"
     "测试文件放在工作区【根目录】命名为 test_life.py（不要建 tests/ 子目录），用 "
     "`from src.life import init_grid, step, main` 导入。写 3-5 个 pytest 函数覆盖：空网格演化后不变、"
     "单个存活细胞在空邻域会死亡、滑翔机(glider)演化 4 代后整体向右下平移 1 格且形状不变"
     "（glider 恒为 5 个活细胞组成的实心块，不是对角线：初始置于左上角 "
     "(0,1)(1,2)(2,0)(2,1)(2,2)，在 5x5 网格上演化 4 代后为 (1,2)(2,3)(3,1)(3,2)(3,3)；"
     "测试须断言 step 演化 4 代后的存活坐标集合等于该期望集合——glider 周期 4、不会消失）。"
     "确保 pytest 全部通过。"),
    ("fizzbuzz",
     "在 当前目录 用 Python 实现 fizzbuzz：src/fizzbuzz.py 提供 fizzbuzz(n) 返回 1..n 的列表"
     "（3 的倍数→'Fizz'，5→'Buzz'，15→'FizzBuzz'，其余为数字字符串），以及 main() CLI"
     "（if __name__=='__main__': main()）。测试文件放在工作区【根目录】命名为 test_fizzbuzz.py"
     "（不要建 tests/ 子目录），用 `from src.fizzbuzz import fizzbuzz, main` 导入。写 3-5 个 pytest 函数"
     "覆盖 n=15 的完整输出、边界 n=1、以及 n=0 返回空列表。确保 pytest 全部通过。"),
    ("guess",
     "在 当前目录 用 Python 实现数字猜谜：src/guess.py 提供 make_target(seed)——【函数内部】用 "
     "random.seed(seed) 然后返回 random.randint(1,100) 作为目标（seed 是【参数】，不要放到模块顶层）；"
     "guess(target, x) 返回比较提示：x==target→'hit'、target>x（目标偏大）→'low'、target<x（目标偏小）→'high'；"
     "main() CLI（if __name__=='__main__': main() 用固定种子生成目标以便测试）。测试文件放在工作区【根目录】"
     "命名为 test_guess.py（不要建 tests/ 子目录），用 `from src.guess import make_target, guess` 导入。"
     "写 3-5 个 pytest 函数验证：猜中返回 'hit'、target>x 时返回 'low'、target<x 时返回 'high'。"
     "确保 pytest 全部通过。"),
    ("fib",
     "在 当前目录 用 Python 实现斐波那契：src/fib.py 提供 fib(n)（返回前 n 项列表，fib(0)=[]、"
     "fib(1)=[1]、fib(2)=[1,1]、fib(10) 末项 55），以及 main() CLI（if __name__=='__main__': main()）。"
     "测试文件放在工作区【根目录】命名为 test_fib.py（不要建 tests/ 子目录），用 `from src.fib import fib` "
     "导入。写 3-5 个 pytest 函数验证 fib(0)/fib(1)/fib(10) 等已知值。确保 pytest 全部通过。"),
]


def _harness_target_models() -> set:
    """harness 实际会用到的全部模型——单点取自 `swe_agent/config.py` 的 BUILD 常量。

    覆盖 5 个角色位：
      planner → PLANNER_MODEL
      analyzer → ANALYZER_MODEL
      executor → EXECUTOR_MODEL
      tester   → TESTER_MODEL（"off" = 关闭独立验收，不加载）
      compact  → SIDECAR_COMPRESS_MODEL（compact.py:102 的压缩后端）

    治理背景（2026-09-02）：旧实现在本脚本硬编码「load qwen + unload 2.6b」，而 config.py
    早已把全部角色默认切到 lfm2.5-2.6b —— 两边脱节，导致每次 e2e 白白加载一个 7b 模型占显存、
    且把主模型卸掉再由 harness 重新 load 回来（日志可见 load/unload 抖动）。
    模型名必须只在 BUILD 层定义一处，驱动脚本只消费。
    """
    from swe_agent import config as C
    raw = [
        C.PLANNER_MODEL,
        C.ANALYZER_MODEL,
        C.EXECUTOR_MODEL,
        C.TESTER_MODEL,
        C.SIDECAR_COMPRESS_MODEL,
    ]
    # 过滤空值与关闭位（TESTER_MODEL="off"/"none"/"false" 表示关闭独立验收，无需加载）
    off = {"off", "none", "false"}
    return {str(m).strip() for m in raw
            if m and str(m).strip() and str(m).strip().lower() not in off}


def reset_models():
    """任务开始前强制确认 harness 各角色位的目标模型【存在且已加载且为唯一响应者】（防御错模/残留态）。

    本函数与 SWE_MODEL_LOAD_UNLOAD 开关【解耦】：它只负责电池运行前的「模型就位」校验，
    不碰 per-loop 的 load/unload 编排（那由 supervisor/compact 经 models.role_load_unload
    受 SWE_MODEL_LOAD_UNLOAD 控制，只管 loop2/loop3/compact 内是否自己换模）。无论该开关如何，
    电池都必须保证目标模型就位——否则 LM Studio 会用其它驻留模型（如 lfm2.5）在请求的
    model 字段未命中时静默顶替，导致「号 ling 跑 lfm2.5」的错模（2026-09-03 实测）。

    行为（一次性、非 per-loop，不引入换模抖动）：
      1) 收集目标模型（_harness_target_models，来自 config BUILD 常量/env）；
      2) 卸掉所有【非目标、非升级 fallback】的驻留模型——彻底杜绝无关驻留模型顶替；
         （旧 2026-09-02 抖动是「卸了目标模型又被 harness 重新 load 回来」，本步只卸非目标，
          故 harness 不会重新 load，无抖动；fallback 模型受保护不卸，保留升级子 agent 能力）
      3) 补齐加载缺失的目标模型；
      4) 二次校验：任一目标模型仍不在 loaded 列表 → 直接 fail-loud 退出（sys.exit），
         绝不让错误模型静默顶替跑完整场 e2e。
    """
    from swe_agent import config as C
    targets = _harness_target_models()
    if not targets:
        logger.info('%s', '[reset_models] BUILD 层未解析出任何目标模型（全部关闭？），跳过换模')
        return
    # 受保护（不卸载）集合：目标模型 + 可能的升级 fallback（避免误卸升级子 agent 所需模型）
    protected = set(targets)
    for envn in ("PLANNER_FALLBACK_MODEL", "ANALYZER_FALLBACK_MODEL"):
        v = getattr(C, envn, None)
        if v and str(v).strip() and str(v).strip().lower() not in ("off", "none", "false"):
            protected.add(str(v).strip())
    logger.info('%s', f'[reset_models] 目标模型（取自 config.py）= {sorted(targets)}')
    logger.info('%s', f'[reset_models] 受保护（不卸载）模型 = {sorted(protected)}')
    import urllib.request, urllib.error
    base = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234").rstrip("/")
    def _api(method, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    # 拿 LM Studio 当前驻留列表；连不上则无法确认就位 → fail-loud 退出
    try:
        loaded = {m["id"] for m in _api("GET", "/v1/models").get("data", [])}
    except Exception as e:
        logger.info('%s', f'[reset_models] ⚠️ 无法连接 LM Studio（{e}），目标模型就位校验失败，直接退出。')
        sys.exit(2)
    # 1) 卸非目标/非 fallback 驻留模型（一次性，杜绝错模顶替）
    for name in sorted(loaded - protected):
        try:
            _api("POST", "/api/v1/models/unload", {"instance_id": name})
            logger.info('%s', f'[reset_models] 已卸载非目标模型 {name}')
        except Exception as e:
            logger.info('%s', f'[reset_models] 卸载非目标模型 {name} 失败（{e}），继续。')
    # 2) 补齐加载缺失目标模型
    loaded = {m["id"] for m in _api("GET", "/v1/models").get("data", [])}
    for name in sorted(targets - loaded):
        try:
            _api("POST", "/api/v1/models/load",
                 {"model": name, "context_length": 32768,
                  "flash_attention": True, "echo_load_config": True})
            logger.info('%s', f'[reset_models] 已加载 {name}')
        except Exception as e:
            logger.info('%s', f'[reset_models] 加载 {name} 失败：{e}')
    # 3) 二次校验：任一目标仍未加载 → fail-loud 退出
    loaded = {m["id"] for m in _api("GET", "/v1/models").get("data", [])}
    still_missing = sorted(targets - loaded)
    if still_missing:
        logger.info('%s', f'[reset_models] ⚠️ 以下目标模型未能就位：{still_missing}；直接退出，避免错模顶替。请手动在 LM Studio 加载后重试。')
        sys.exit(2)
    logger.info('%s', f'[reset_models] 目标模型全部就位：{sorted(targets)}')


def clear_workspace(ws: Path):
    """清空并重建指定隔离工作区（每任务独立目录，杜绝跨任务文件泄漏）。"""
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)


def verify_workspace(task_ws: Path) -> dict:
    """独立 pytest 校验（与 harness 单杠标准一致）。返回 {rc, passed, summary, ws_used}。

    健壮性：agent 实际落盘目录可能因 SWE_WORKSPACE 偶发未生效而回退到默认
    WORKSPACE_ROOT（config.WORKSPACE 的默认值 = REPO/agent_sandbox），此时文件写在
    根目录而非隔离嵌套 dir → 仅查 task_ws 会误报 'no test files' / passed=false。
    故同时检查 task_ws（隔离 dir）与 WORKSPACE_ROOT（默认回退），优先用真正含测试文件的
    目录；命中回退目录时附告警（隔离已失效，便于发现 env 透传异常）。"""
    candidates = [task_ws, WORKSPACE_ROOT]
    used = None
    for c in candidates:
        if any(c.rglob("test_*.py")) or any(c.rglob("*_test.py")):
            used = c
            break
    if used is None:
        return {"rc": None, "passed": False,
                "summary": f"no test files in workspace (checked: {task_ws}, {WORKSPACE_ROOT})",
                "ws_used": None}
    try:
        proc = subprocess.run(
            ["uv", "run", "--no-sync", "python", "-m", "pytest", "-q"],
            cwd=str(used), capture_output=True, text=True, timeout=120)
        out = (proc.stdout + proc.stderr)[-2500:]
        note = "" if used == task_ws else (
            f"[warn] agent 实际落盘于默认 WORKSPACE_ROOT（{WORKSPACE_ROOT}），"
            f"SWE_WORKSPACE 可能未生效，每任务隔离失效。\n")
        return {"rc": proc.returncode, "passed": proc.returncode == 0,
                "summary": (note + out.strip()).strip(),
                "ws_used": str(used)}
    except Exception as e:
        return {"rc": None, "passed": False, "summary": f"verify error: {e}", "ws_used": str(used)}


def _loaded_models() -> str:
    try:
        import urllib.request
        base = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234").rstrip("/")
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        return ",".join(m.get("id", "?") for m in d.get("data", []))
    except Exception as e:
        return f"(探测失败:{e})"


def run_one(idx: int, name: str, task: str, env: dict, run_id: str) -> dict:
    # 每任务独立隔离目录：agent_sandbox/<run_id>/<name>
    # 即使旧子进程残留/电池重启，也不会污染其它任务的 workspace（根治跨任务文件泄漏）。
    task_ws = WORKSPACE_ROOT / run_id / name
    clear_workspace(task_ws)
    reset_models()
    logpath = LOGDIR / f"{idx:02d}_{name}.log"
    statuspath = LOGDIR / f"{idx:02d}_{name}.status"
    logger.info('%s', f"\n===== [{idx:02d}] {name} | 开始 {time.strftime('%H:%M:%S')} | ws={task_ws} | log={logpath} =====")
    # 子进程行缓冲：harness 输出实时落盘，便于中途诊断
    child_env = dict(env)
    child_env["PYTHONUNBUFFERED"] = "1"
    # 把隔离目录透传给 harness（config.WORKSPACE 在 import 时读取），并关闭外层重复 reset
    child_env["SWE_WORKSPACE"] = str(task_ws)
    child_env["SWE_FRESH_WORKSPACE"] = "1"
    cmd = ["uv", "run", "--no-sync", "python", "-m", "swe_agent", task]
    t0 = time.time()

    # 看门狗：每 30s 写一次心跳（已用时 / 当前已加载模型），方便发现静默卡死
    stop_watch = False
    def _watchdog():
        while not stop_watch:
            try:
                with open(statuspath, "w") as sf:
                    sf.write(f"elapsed={time.time()-t0:.0f}s models=[{_loaded_models()}]\n")
            except Exception:
                pass
            time.sleep(30)

    with open(logpath, "w", buffering=1) as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(REPO), env=child_env, preexec_fn=os.setsid)
        wd = threading.Thread(target=_watchdog, daemon=True)
        wd.start()
        timed_out = False
        try:
            rc = proc.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
            rc = -9
        stop_watch = True
        dur = time.time() - t0

    status = "timeout" if timed_out else ("returned" if rc == 0 else f"exit{rc}")
    verify = verify_workspace(task_ws)
    passed = verify["passed"]

    rec = {
        "idx": idx, "name": name, "status": status,
        "passed": passed, "duration_s": round(dur, 1),
        "harness_rc": rc, "pytest_rc": verify["rc"],
        "pytest_summary": verify["summary"], "log": str(logpath),
        "task": task,
    }
    logger.info('%s', f"  -> status={status} passed={passed} dur={rec['duration_s']}s pytest_rc={verify['rc']}")
    return rec


def _count_running_instances() -> list:
    """统计【除自己之外】正在跑的本脚本实例 pid 列表（无参数依赖，纯进程探测）。

    精确排除三类干扰进程，只数「真正的解释器进程在跑本脚本」：
      - `sh -c ... eval '... python run_e2e_battery.py'` 包装层 → 进程名是 zsh，不是 python
      - `uv run --no-sync python run_e2e_battery.py` 包装层 → 进程名是 uv
      - 电池驱动 spawn 的 harness 子进程（`python -m swe_agent`）→ 命令行不含本脚本名
    做法：取 `pgrep -f <脚本名>`（全命令行候选）与 `pgrep -x python3`（按进程名精确匹配）
    的**交集**，再刨掉本进程。

    探测失败（pgrep 不可用 / 被沙箱限制）时返回空列表 —— 观测手段缺失不应阻塞正常跑测，
    此时退化为「不拦截」，由 rmtree 前的其它提示兜底。
    """
    script = Path(__file__).resolve().name
    try:
        cand = set(subprocess.run(["pgrep", "-f", script],
                                  capture_output=True, text=True, timeout=5).stdout.split())
        py: set = set()
        for exe in ("python3", "python3.13", "python3.12", "python", "python3.11"):
            py |= set(subprocess.run(["pgrep", "-x", exe],
                                     capture_output=True, text=True, timeout=5).stdout.split())
    except Exception:
        return []
    return sorted((cand & py) - {str(os.getpid())})


def _pgrep(pattern: str) -> list:
    """按命令行模式取 pid 列表；探测失败返回空（观测手段缺失不阻塞流程）。"""
    try:
        return subprocess.run(["pgrep", "-f", pattern],
                              capture_output=True, text=True, timeout=5).stdout.split()
    except Exception:
        return []


def _kill_leftover_instances():
    """自动清理所有已存在的电池实例，【连 harness 孙进程一起杀】。

    必须杀两层，缺一不可：
      A. `pgrep -f <本脚本名>` → 覆盖 `zsh -c` 包装层 / `uv` 包装层 / python 驱动层；
      B. `pgrep -f swe_agent`  → 驱动层 spawn 的 harness 子进程（独立进程组）。
    ⚠️ 只杀 A 不杀 B 会留下孤儿继续占 LM Studio 推理资源（2026-09-02 实测：父进程被杀后
    孙进程 97661/98731 仍在跑，导致后续 e2e 被拖慢）——这正是当初「第一个 e2e 跑 25 分钟」
    的次生原因。全部用 SIGKILL，不给残留进程留自旋/清理时间窗口。
    """
    # A 层刻意【只杀 python 驱动层】而非 `pgrep -f <脚本名>` 全量：后者会命中新实例自己的
    # `zsh -c` / `uv` 包装层（它们的 pid 比本进程小、命令行里也含脚本名），会把自己一起杀掉。
    # 杀掉驱动层后，其 uv / zsh 父进程会因子进程退出而自然收尾，无需（也不能）手动杀。
    for label, pids in (("电池驱动层(python)", _count_running_instances()),
                        ("harness 子进程(swe_agent)", [p for p in _pgrep("swe_agent")
                                                       if p != str(os.getpid())])):
        if not pids:
            continue
        logger.info('%s', f"   · 清理 {label}: pid={','.join(pids)}")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass
    time.sleep(2)   # 给内核回收进程的时间，再复查


def _assert_single_instance():
    """并发闸门：同一时刻只允许一个电池实例在跑；检测到旧实例自动清理后继续。

    2026-09-02 事故根因：并发实例有三重破坏 ——
      1. main() 里的 `shutil.rmtree(WORKSPACE_ROOT)` 会让后启动者把先启动者的整个
         `agent_sandbox/`（含其 cwd）连根删除，前者只能在废墟上反复重试空转；
      2. `logs/e2e_battery/` 下 01_<task>.log / .status / summary.json 是固定名，互相覆盖；
      3. 抢同一个 LM Studio 推理后端，请求排队 + MODEL_RETRY 退避，耗时成倍放大。
    故在任何写盘动作（LOGDIR.mkdir / rmtree）之前完成清理，清理不干净则拒绝启动。
    """
    others = _count_running_instances()
    if not others:
        return
    logger.info('%s', f"⚠️ 检测到已有 {len(others)} 个 run_e2e_battery 实例在跑（pid={','.join(others)}） —— 同一时刻只允许一个，自动清理后继续。")
    _kill_leftover_instances()
    rest_battery = _count_running_instances()
    rest_harness = _pgrep("swe_agent")
    if rest_battery or rest_harness:
        logger.info('%s', f"❌ 自动清理未果，仍有残留：电池实例={rest_battery or '无'} harness={rest_harness or '无'}。终止启动（不带着冲突往下跑）。")
        sys.exit(2)
    logger.info('%s', '   ✅ 旧实例已清理干净，继续。')


def main():
    global TIMEOUT
    # 并发闸门必须早于一切写盘动作（LOGDIR.mkdir / rmtree），否则来不及止损。
    _assert_single_instance()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="", help="逗号分隔任务名子集，空=全部")
    ap.add_argument("--timeout", type=int, default=TIMEOUT, help="单任务超时秒")
    args = ap.parse_args()

    TIMEOUT = args.timeout

    tasks = DEFAULT_TASKS
    if args.tasks:
        want = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [(n, t) for (n, t) in DEFAULT_TASKS if n in want]
        if not tasks:
            logger.info('%s', f'无匹配任务：{args.tasks}')
            sys.exit(2)

    LOGDIR.mkdir(parents=True, exist_ok=True)
    # 每个电池实例一个唯一 run_id：即使旧电池子进程残留或本电池被重启，
    # 每任务的隔离目录路径都不同 → 任意重叠都不会互相污染工作区文件。
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    # 清掉上一电池遗留的隔离根（按 run_id 隔离后历史目录不再被复用，直接整体清理避免磁盘堆积）
    if WORKSPACE_ROOT.exists():
        shutil.rmtree(WORKSPACE_ROOT, ignore_errors=True)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info('%s', f'[battery] run_id={run_id} workspace_root={WORKSPACE_ROOT}')
    env = dict(os.environ)
    env.update({
        "SWE_MAX_ATTEMPTS": "3",
        "SWE_MAX_ROUNDS": "3",
        "SWE_MAX_STEPS": "8",
        "SWE_MODEL_RETRY": "4",
        "SWE_MODEL_BACKOFF": "2.0",
        # 无人值守：BUILD 层摘除 ask 工具，tester 失败/模型不可用自动判 fail 续跑，绝不卡 stdin
        "UNATTENDED_MODE": "1",
        # 自动压缩默认开启（compact.py 已接入 model_swap：压缩前 unload qwen → load 2.6b → 还原）
    })

    overall_t0 = time.time()
    results = []
    for i, (name, task) in enumerate(tasks, 1):
        rec = run_one(i, name, task, env, run_id)
        results.append(rec)
        # 增量写 summary，方便中途查看
        _write_summary(results, overall_t0, args)

    total = len(results)
    ok = sum(1 for r in results if r["passed"])
    logger.info('%s', '\n================ 电池汇总 ================')
    logger.info('%s', f'总任务={total}  通过={ok}  完成率={ok / total * 100:.0f}%')
    for r in results:
        logger.info('%s', f"  {r['idx']:02d} {r['name']:10s} {r['status']:8s} passed={r['passed']!s:5s} {r['duration_s']:7.1f}s")
    logger.info('%s', f'总耗时={time.time() - overall_t0:.0f}s')


def _write_summary(results, t0, args):
    total = len(results)
    ok = sum(1 for r in results if r["passed"])
    summary = {
        "total": total, "passed": ok,
        "completion_rate": (ok / total) if total else 0,
        "elapsed_s": round(time.time() - t0, 1),
        "tasks": results,
    }
    (LOGDIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
