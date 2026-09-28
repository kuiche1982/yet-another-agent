# tester 角色：提示词 & 消息快照

> 生成于 2026-09-02，对应 run `20260902_094144_9e8dc8`（fib，全角色 lfm2.5-2.6b）。
> 目的：澄清「tester 到底收到了什么」+ 「调用工具」兜底提示的真实来源。

---

## 一、tester 的 system prompt（`swe_agent/verify.py:33-54`）

```
你是独立验收 agent（tester）。你的唯一职责是【独立裁判】：这份代码不是你写的，
你也绝不能写——你只负责对照「需求」与「验收点」，去查看真正的实现代码与运行结果，判断需求是否被满足。
你绝不当运动员，也不兼当裁判和下场选手。

你拥有只读工具：read_file / glob / grep / shell（shell 仅用于只读检查命令，如 ls/grep/cat 查看文件；绝不用来运行交付的程序或重跑 pytest）。你没有、也不能调用任何写文件/改文件工具。

逐条核对每个验收点的方法：
1. 先按验收点的描述，定位它对应的实现代码：用 grep / read_file 找到相关的类/函数定义，
   读它的真实签名与逻辑，核对是否实现了该验收点要求的行为——
   不是凭印象、也不是信任 executor 的说法，而是看【代码本体】到底写了什么。
2. 运行时/行为类验收点（如 CLI 可运行、算法输出正确、边界不崩）：通过【读代码 + 读测试】判定，
   不靠 shell 运行程序——运行目标程序可能卡在 stdin 输入导致超时，且单杠 pytest 已在 loop 2 由 harness 跑过。
   具体做法：读实现确认逻辑（如 main() 不依赖交互式 input、有 `if __name__=='__main__'` 入口、边界处理到位）；
   读测试文件确认有真实断言覆盖该行为（而非空测试或只 import 不 assert）。
3. 每条验收点都必须有【可核对的证据】：来自实际读到的代码片段，或真实命令输出。
   没有证据就判 pass = 作弊，一律判 fail；禁止「我认为应该没问题」式结论。

提交规则：
4. 逐条核对完后，调用 finish_verify 提交，每条含 verdict(pass/fail) 与 evidence。
5. 探查要聚焦：预算约 = 验收点数 × 2 + 2 轮工具调用，给足「读实现 + 读测试」的余地，
   不要反复空跑 shell 或重复 read 同一文件。若目标文件不存在 / 命令无有效输出，
   直接基于现有证据判对应验收点 fail 并 finish_verify，不要陷入「读不到就一直试」的死循环。
```

工具集硬控（`roles_config.py:62-76` + `verify.py:31`）：`read_file / glob / grep / shell / finish_verify`，
`stop_actions=("finish_verify",)`，纯只读。

---

## 二、发给 tester 的 user message（`verify.py:79-102` `_render_task`）

模板结构（按顺序拼接）：

```
【需求】
<goal>

【实现契约 · 模块与公开签名】（按 path 用 read_file/grep 查看对应实现）：
  - <path>：<public 签名列表>
  - ...

【接口签名 interface（全部公开函数最终签名，与上方 public 逐字一致）】：
  - <sig>
  - ...

【验收点】（逐条核对，必须引用证据）：
  #<id> <point>
    建议核对：<check_hint>
  #<id> <point>
    建议核对：<check_hint>
  ...

请开始验证，全部核对完后调用 finish_verify 提交。
```

### 当前 run 实际渲染样例（goal 来自任务描述；contract 来自 planner 落地的 plan；验收点取自 `.swe_verify.json`）

```
【需求】
实现 fib 函数：返回斐波那契数列前 n 项（list[int]），并提供 CLI 入口。

【实现契约 · 模块与公开签名】（按 path 用 read_file/grep 查看对应实现）：
  - src/fib.py：fib; main

【接口签名 interface（全部公开函数最终签名，与上方 public 逐字一致）】：
  - def fib(n: int) -> list:
  - def main():

【验收点】（逐条核对，必须引用证据）：
  #1 fib(0) 返回空列表 []
    建议核对：assert fib(0) == []
  #2 fib(1) 返回 [1]
    建议核对：assert fib(1) == [1]
  #3 fib(10) 前 10 项末项为 55
    建议核对：assert fib(10)[-1] == 55

请开始验证，全部核对完后调用 finish_verify 提交。
```

---

## 三、「你必须通过【调用工具】来输出动作」真实来源（`agent.py:218`）

**不在 tester 任何提示词里**，是 `_apply_toolcall` 的兜底分支：

```python
# agent.py:215-222
if meta.get("type") != "toolcalls":
    st = self.ctx.cm.guard
    st["no_tool_streak"] += 1
    self.ctx.cm.append("user", "（系统）你必须通过【调用工具】来输出动作，不要只回复散文。请调用一个工具。")
    if st["no_tool_streak"] >= C.LOOP_REPEAT_THRESHOLD:
        print(f"⚠️ agent：连续 {st['no_tool_streak']} 次未调用工具，判定停滞。")
        return "stuck"
    return "continue"
```

触发条件：**模型本轮返回的 meta 不是 toolcalls 类型**——即模型只输出了散文 / reasoning，没发出任何工具调用。
判定 `{"type":"toolcalls", "actions":[...]}` 来自 `models.chat_toolcalls`；lfm2.5-2.6b 偶发只回文本 → 落入此分支 → 注入这句强制提示。

> 注：另有 `agent.py:212` 的「上一轮模型未返回任何内容或工具调用」（空响应分支）和
> `agent.py:272` 的「连续重复相同动作」提示，三者都是 loop-guard 兜底，都不是 tester 提示词。

---

## 四、当前 run 日志印证的现象（`/tmp/e2e_battery/01_fib.log` tester 阶段）

```
34:[agent] 终止动作 finish_verify 命中，本步循环结束。
35:[agent] 终止动作 finish_verify 命中，本步循环结束。
36:⚠️ agent：连续 3 次重复动作（指纹 'finish_verify:'），结束本步循环以换策略。
37:[agent] 终止动作 finish_verify 命中，本步循环结束。
38:[agent] 终止动作 finish_verify 命中，本步循环结束。
39:⚠️ agent：连续 3 次重复动作（指纹 'finish_verify:'），结束本步循环以换策略。
```

读出来的事实：
1. `finish_verify` **确实被正确识别为 stop_action**（agent.py:276 捕获 → all_done），所以 tester 能收尾。
2. 但它在日志里**命中了 5 次**还夹了 2 次「连续 3 次重复动作（指纹 finish_verify:）」——
   说明 `run_tester` / verify_gate 被外层 3-layer loop **反复触发**，且模型本身也会连续重发同一 finish_verify（指纹相同、无参差异）。
3. 用户贴的「模型在 reasoning 里反复说『我已调过 finish_verify，系统却让我调工具』」——对应
   模型偶发只回散文（触发 `agent.py:218` 兜底提示）的那个轮次；它把 harness 的强制提示误读成
   「finish_verify 没被接受」，于是在 reasoning 里反复纠结，而非真正重发带参 tool_call。

根因定性：弱模型（lfm2.5-2.6b）在 tester 阶段 tool_call 产出不稳定 + 外层 loop 对 verify_gate 的重触发，
共同导致「反复 finish_verify / 偶发纯文本触发兜底提示」的观感。**不是提示词写错**，是弱模型行为 + loop 重触发。
