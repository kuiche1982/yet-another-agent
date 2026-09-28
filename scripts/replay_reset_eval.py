#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线回放验证：用已有的 session 历史记录，重现「executor 任务完成重置历史」机制的效果。

不跑全量 harness —— 直接从 s-20260830-080857.json 取出每轮的：
  - task 状态（done_list 增长 → 触发 reset）
  - 累积式输入（WITHOUT reset，原 harness 行为）
  - 重置式输入（WITH reset，system + 最近 K 轮 + 更新任务列表，指令置底）
然后逐步把「重置式」上下文发给本地模型，验证信息是否够用（模型仍能给出有效动作）。
"""
import json, os, sys, time, re, requests

REPO = "~/kuiwork/workdir2/litertlm"
SESSION = os.path.join(REPO, "sessions/s-20260830-080857.json")
BASE = "http://127.0.0.1:8000/v1"
KEEP_TURNS = 1  # 与 config.EXECUTOR_RESET_KEEP_TURNS 一致

# ---------- 直接复用 harness 里的真实重置逻辑（不再复刻，验证即验真代码） ----------
sys.path.insert(0, REPO)
import swe_agent.supervisor as _S
from swe_agent.log import logger
_capture_last_turns = _S._capture_last_turns
_rebuild_executor_context = _S._rebuild_executor_context

def _chars(msgs):
    return sum(len(m.get("content", "")) for m in msgs)

# ---------- 解析历史 ----------
sess = json.load(open(SESSION))
msgs = sess["messages"]

# 找出所有「轮次指令」user 消息（以【目标】开头）
directives = [(i, m) for i, m in enumerate(msgs) if m.get("role") == "user" and str(m.get("content", "")).startswith("【目标】")]
logger.info('%s', f"session={sess.get('session_id')}  总轮次指令数={len(directives)}")

def parse_done(ctx):
    """从指令文本里取 done 任务列表。

    以【已完成】行（authoritative done_list）为准，而非【任务列表】里的
    [done]/[pending] 标记 —— 本 session 是「补测修复」续跑，任务列表块标记已过时。
    注意【已完成】行里可能存在重复项，用 set 去重。
    """
    done = set()
    m = re.search(r"【已完成】(.*?)(?:\n|【当前待办】|【本步)", ctx, re.S)
    if m:
        for part in m.group(1).split("，"):
            p = part.strip()
            if p and p != "无":
                done.add(p)
    return done

rounds = []
prev_done = None  # 第 1 轮是基线，没有上一轮可对比，不算 reset
for idx, (i, m) in enumerate(directives, start=1):
    ctx = m["content"]
    done = parse_done(ctx)
    # WITHOUT reset：到这一轮指令为止的累积输入（原 harness 行为，含 auto-compact 摘要）
    without = msgs[:i + 1]
    # WITH reset：重建上下文
    with_ctx = _rebuild_executor_context(msgs[:i + 1], ctx, KEEP_TURNS)
    # reset 触发条件：上一轮有 task 被标记完成（done_list 相对上一轮增长）
    cur_done = set(done)
    reset_fires = (prev_done is not None) and (len(cur_done - prev_done) > 0)
    prev_done = cur_done
    rounds.append({
        "round": idx,
        "directive_index": i,
        "done_count": len(done),
        "reset_fires": reset_fires,
        "without_chars": _chars(without),
        "with_chars": _chars(with_ctx),
        "with_ctx": with_ctx,
        "directive": ctx,
        "original_action": msgs[i + 1]["content"] if (i + 1) < len(msgs) and msgs[i + 1].get("role") == "assistant" else "(无)",
    })

# ---------- 打印对比表 ----------
logger.info('%s', '\n=== 每轮输入字符数对比（WITHOUT reset 累积 vs WITH reset 重置） ===')
logger.info('%s', f"{'轮':>2} | {'done':>4} | {'reset?':>6} | {'WITHOUT':>8} | {'WITH':>7} | {'节省':>7} | {'降幅':>5}")
total_wo = total_wi = 0
for r in rounds:
    saved = r["without_chars"] - r["with_chars"]
    pct = (saved / r["without_chars"] * 100) if r["without_chars"] else 0
    total_wo += r["without_chars"]
    total_wi += r["with_chars"]
    logger.info('%s', f"{r['round']:>2} | {r['done_count']:>4} | {('是' if r['reset_fires'] else '否'):>6} | {r['without_chars']:>8} | {r['with_chars']:>7} | {saved:>+7} | {pct:>4.1f}%")
logger.info('%s', f"{'Σ':>2} | {'':>4} | {'':>6} | {total_wo:>8} | {total_wi:>7} | {total_wo - total_wi:>+7} | {(total_wo - total_wi) / total_wo * 100:>4.1f}%")

# ---------- 逐步发给本地模型（验证 reset 上下文信息够用） ----------
MODEL = "qwen-2.5-coder-7b"
live = []
if os.environ.get("OFFLINE"):
    logger.info('%s', '\n[OFFLINE] 跳过本地模型调用，仅输出离线字符对比。')
else:
    logger.info('%s', '\n=== 逐步把「WITH reset」上下文发给本地 qwen，验证模型仍能给出有效动作 ===')
    for r in rounds:
        ctx_msgs = r["with_ctx"]
        payload = {
            "model": MODEL,
            "messages": [{"role": mm["role"], "content": mm["content"]} for mm in ctx_msgs],
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        t = time.time()
        try:
            resp = requests.post(f"{BASE}/chat/completions", json=payload, timeout=(15, 120))
            data = resp.json()
            out = data["choices"][0]["message"]["content"]
            err = None
        except Exception as e:
            out = ""
            err = f"{type(e).__name__}: {e}"
        dt = time.time() - t
        # 抽取动作类型
        act = "(解析失败)"
        m = re.search(r'"action"\s*:\s*"([^"]+)"', out)
        if m:
            act = m.group(1)
        # 原始这一轮的动作类型
        oact = "(解析失败)"
        mo = re.search(r'"action"\s*:\s*"([^"]+)"', r["original_action"])
        if mo:
            oact = mo.group(1)
        logger.info('%s', f"轮{r['round']} reset={('是' if r['reset_fires'] else '否')} in={r['with_chars']}c 耗时={dt:.1f}s 模型动作={act} | 原动作={oact} " + (f'ERR={err}' if err else ''))
        live.append({
            "round": r["round"], "reset_fires": r["reset_fires"],
            "input_chars": r["with_chars"], "model_action": act,
            "original_action": oact, "latency_s": round(dt, 1), "error": err,
        })

# ---------- 汇总写文件 ----------
summary = {
    "session": sess.get("session_id"),
    "rounds": len(rounds),
    "keep_turns": KEEP_TURNS,
    "total_without_chars": total_wo,
    "total_with_chars": total_wi,
    "total_saved_chars": total_wo - total_wi,
    "total_saved_pct": round((total_wo - total_wi) / total_wo * 100, 1) if total_wo else 0,
    "reset_events": sum(1 for r in rounds if r["reset_fires"]),
    "per_round": [
        {k: v for k, v in r.items() if k not in ("with_ctx", "directive", "original_action")}
        for r in rounds
    ],
    "live_model": live,
}
out_path = os.path.join(REPO, "replay_reset_summary.json")
json.dump(summary, open(out_path, "w"), ensure_ascii=False, indent=2)
logger.info('%s', f'\n已写出汇总: {out_path}')
logger.info('%s', json.dumps({k: v for k, v in summary.items() if k != 'per_round'}, ensure_ascii=False, indent=2))
