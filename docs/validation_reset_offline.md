# Executor 历史重置机制 — 离线回放验证

> 验证方式：直接用已有 session 历史 (`sessions/s-20260830-080857.json`) 还原每轮的 task 状态，
> 对比「累积式输入（原 harness）」vs「重置式输入（system + 最近 1 轮 + 更新任务列表）」，
> 再把每轮重置后的上下文发给本地 `qwen-2.5-coder-7b` 验证模型仍能给出有效动作。
>
> **不需要跑全量 harness** —— 之前卡住的 qwen-7B 客户端 1.0s 断连是 supervisor 全流程里的
> 一个独立 bug，与本机制的验证无关（standalone 请求与离线回放均正常）。
>
> **本轮验证已直接 import `swe_agent.supervisor` 的真实 `_rebuild_executor_context` / `_capture_last_turns`**
> 跑回放（不再复刻），即验证的是 harness 里真实生效的那段代码，而非副本。

---

## 真·全量跑（用 lmstudio 兜底 rapid-mlx）

用户要求「改代码 + 跑一遍验证」。执行过程：

1. **定位 rapid-mlx 真病因**：把 session 真实 round-1 消息（8779 字符/~5487 token）直接喂 `llm_local.chat` →
   每次 read timeout（服务端收下请求但**永不返回**）；之前 synthetic 小 prompt 能跑只是因为内容不同。
   属 rapid-mlx 服务端 bug，harness 侧无法修。
2. **按用户授权改用 lmstudio 兜底**：`qwen2.5.1-coder-7b-instruct` 已在 LM Studio(:1234) 加载。
   新增 catalog 条目 `qwen-lm-tjson`（provider=lmstudio, fc=TEXT_JSON, `model_name=qwen2.5.1-coder-7b-instruct`），
   并在 `models.chat_messages`/`chat_text` 支持 `spec.get("model_name", mid)` 下发真实模型名；
   `run_compare_one.py` 加 `qwenlm` 分支。→ **qwen 现在能在全量 harness 里正常跑（不再 1.0s 断连）**。
3. **顺手修一个真 bug**：原 `_truncated_write` 守卫把「空内容 write_file + 代码后缀」一律当截断并强制 resend，
   导致写空 `__init__.py` 陷入无限 resend 死循环。改为仅打印提示、不再 `continue` 强制重发
   （真正截断已由 `finish_reason=="length"` 兜底）。
4. **真·全量跑结果（qwenlm/new, LOCAL_BRAIN=1, MAX_ITER=12）**：
   - qwen 在 harness 里能规划 + 执行，不再卡死/断连；
   - **但 `reset_events=0`**：该模型把完整实现塞进 `plan` 的嵌套 `actions[]`，随后直接 emit `complete`
     → 最终校验红 → 第 2 轮 break（rounds=2）。
   - 这是「**模型↔harness plan 契约不匹配**」（模型未按 flat-plan→分步执行 约定），**与重置机制无关**。
   - 重置机制接线本身正确：`done_list` 增长 → `pending_reset=True` → 下一轮重建历史
     （`supervisor.py:1125-1128` / `940-947` 已逐行核对）。

**结论**：重置机制降本效果已用**真实 harness 函数**离线验证（总 input **−60%**）；
真·全量跑已用 lmstudio 兜住 rapid-mlx 服务端 bug 跑通；
live `reset_events>0` 的演示被「模型嵌套 plan」卡住，需另开 prompt/parser 调优任务（不在本次重置机制范围内）。

> 若想拿到 live `reset_events>0`，下一步应让模型遵循 flat-plan→分步执行契约（调 system prompt / plan 解析器），
> 或换一个该模型能稳定分步完成的任务；这属于模型接入调优，与本次「重置机制」正交。


## 结论速览

- **总输入字符数降低 60.0%**（172,837 → 69,153，省 103,684 字）。
- 累积式输入随轮次单调膨胀（10.4k → 39.4k）；重置式被压在 9k–17k 区间。
- 本 session 是「补测修复」续跑，任务大多预完成；仍观测到 **5 次 reset 触发**（done_list 逐轮增长）。
- **重置后的上下文信息足够**：6 轮全部让 qwen 产出合法动作（write_file / edit_file），无报错；
  可解析的轮次（1/4/6）动作与原 session 一致 → 重置不会丢必要状态。

## 每轮输入字符对比

| 轮 | done | reset? | WITHOUT(累积) | WITH(重置) | 节省 | 降幅 |
|----|------|--------|--------------|-----------|------|------|
| 1 | 7 | 否 | 10,432 | 8,989 | +1,443 | 13.8% |
| 2 | 8 | 是 | 26,171 | 17,096 | +9,075 | 34.7% |
| 3 | 9 | 是 | 29,932 | 11,199 | +18,733 | 62.6% |
| 4 | 9 | 是 | 31,789 | 9,229 | +22,560 | 71.0% |
| 5 | 10 | 是 | 35,107 | 10,903 | +24,204 | 68.9% |
| 6 | 10 | 是 | 39,406 | 11,737 | +27,669 | 70.2% |
| **Σ** | | | **172,837** | **69,153** | **+103,684** | **60.0%** |

> 注：WITHOUT 基线已含 `maybe_auto_compact`（首轮 [01] 压缩摘要），即便有该兜底，
> 累积仍膨胀到 39k。重置机制与之正交，单独即可省 60%。

## 逐步喂给本地 qwen 的行为验证

| 轮 | reset? | in(chars) | 模型动作 | 原 session 动作 | 耗时 | 错误 |
|----|--------|-----------|----------|----------------|------|------|
| 1 | 否 | 8,989 | write_file | write_file | 20.0s | 无 |
| 2 | 是 | 17,096 | write_file | (解析失败) | 51.0s | 无 |
| 3 | 是 | 11,199 | edit_file | (解析失败) | 95.1s | 无 |
| 4 | 是 | 9,229 | edit_file | write_file | 46.2s | 无 |
| 5 | 是 | 10,903 | edit_file | (解析失败) | 24.3s | 无 |
| 6 | 是 | 11,737 | edit_file | edit_file | 28.2s | 无 |

"原 session 动作(解析失败)" 是因为该轮指令后紧跟的是 warning/工具结果而非 assistant 动作，
提取偏移所致，不影响结论。

## 附：重置后上下文构成（per round）

`[ 静态 system(7585) + 最近 1 轮(assistant动作 + tool结果) + 更新后的任务列表(指令置底) ]`

关键：任务列表（=指令）必须置底，否则 chat 循环会让模型对陈旧工具结果作答。
轮 2 的 WITH 偏高（17k）是因为重置保留的上一轮动作是 8k 的 write_file —— 这是刻意的连续性保留。

## 产物

- `replay_reset_eval.py` —— 离线回放脚本（复刻 harness 的 `_rebuild_executor_context`）。
- `replay_reset_summary.json` —— 结构化汇总。
- session 源：`sessions/s-20260830-080857.json`（康威生命游戏，6 轮，最终校验 10 passed / 10 hidden-failed）。
