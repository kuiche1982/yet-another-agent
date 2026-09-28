#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ContextManager.prepare_messages —— sliding 前后 × append(user/assistant/tool) 单测。

数据来自**真实** LM Studio 请求快照（tests/fixtures/*.json，由
`scripts/gen_contextmgr_fixtures.py` 从 logs/lmstudio_requests.jsonl 第 345/466 行导出）；
不直接读 sessions/ 与 logs/ —— 二者是可变目录，会被后续 e2e 覆写，不能当测试输入。

覆盖矩阵（sliding 前 / 后 × 三种 append）：
  1. sliding 前：预算充足 → 历史逐条原样，system 唯一居首
  2. sliding 前：只读，不缩 buffer
  3. sliding 后（中间轮被丢）：保留 首轮 + 末轮，中间整组折叠
  4. sliding 后（激进）：只剩最后一轮，末尾 tool 回执保住
  5. sliding 后（F1 极端）：预算 < 末轮体量 → 轮内子单元降级，保住最后一条 user（历史不再被清空）
  6. append user（新用户轮）：不重复注入、RAG 表只在末条
  7. append user（UU 连续两条 user）：两条都保留，末条即当前 input
  8. append assistant：末尾 assistant 保住，不凭空注入 user
  9. append tool：tool 回执与 assistant(tool_calls) 严格配对
 10. append tool（单轮 4 tool_calls）：4 条回执全配、顺序 = 声明顺序
 11. sliding 后 + 末尾 tool：压缩完仍配对（OpenAI 协议铁律）
 12. reasoning_content：出向剥除、buffer 保留

模型无关（model-free / tool-free）：kb 显式传空 RagEngine（避免构造期扫工作区），
compress_backend 传恒等函数（杜绝二级 LFM 压缩后端触网络）。
"""

import json
from pathlib import Path

import pytest

from swe_agent.management import ContextManager
from contextmgr import ContextManager as RagEngine
from contextmgr import estimate_tokens as est

FIXTURES = Path(__file__).parent / "fixtures"
CONWAY = "lmstudio_req_345_conway.json"        # 48 条：SUATATAU...(31)...UUATATATAT
MULTITOOL = "lmstudio_req_466_multitool.json"  # 41 条：SUATTTT...(含单轮 4 tool_calls)


# ---------------------------------------------------------------- 工具
def _messages(name):
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return [dict(m) for m in data["messages"]]


def _cm(messages):
    """构造 ContextManager：kb 显式传空 RagEngine（否则构造期会扫整个工作区）。

    ⚠️ 压缩后端必须**构造后置 None**，不能写 `ContextManager(compress_backend=None)`：
    构造函数把 None 解释为「回落到默认 LFM compact 通道」，那会触网络。
    而 `compress()` 内部对非 None 的后端是「候选超预算就整包交给后端并直接 return」，
    会**绕过整组裁剪**——所以也不能传恒等函数充数。置 None 才是纯 model-free 路径。
    """
    cm = ContextManager(kb=RagEngine(budget_tokens=4096))
    cm.compress_backend = None
    for m in messages:
        cm.add(dict(m))
    return cm


def _hist_budget(mcl, sys_content):
    """复刻 prepare_messages 的历史预算公式：budget*0.6 - system。"""
    budget = int(mcl * 0.8)
    return max(0, budget - est(sys_content) - int(budget * 0.4))


def _declared_ids(out):
    return [c["id"] for m in out if m.get("tool_calls") for c in m["tool_calls"]]


def _result_ids(out):
    return [m.get("tool_call_id") for m in out if m.get("role") == "tool"]


def _assert_protocol(out):
    """OpenAI 协议不变量 —— 每条测试都必须过。"""
    assert out, "输出不得为空"
    assert out[0]["role"] == "system", "system 必须在最前"
    assert sum(1 for m in out if m["role"] == "system") == 1, "system 不得重复"

    declared, results = _declared_ids(out), _result_ids(out)
    orphan = [r for r in results if r not in declared]
    missing = [d for d in declared if d not in results]
    assert not orphan, f"孤儿 tool 回执（有回执无声明）→ provider 400：{orphan[:3]}"
    assert not missing, f"缺回执的 tool_calls（有声明无结果）→ provider 400：{missing[:3]}"

    # 回执必须排在声明它的 assistant 之后
    decl_at = {}
    for i, m in enumerate(out):
        for c in m.get("tool_calls", []):
            decl_at[c["id"]] = i
    for i, m in enumerate(out):
        if m.get("role") == "tool":
            assert decl_at.get(m.get("tool_call_id"), -1) < i, "tool 回执排到了声明之前"

    # 出向清理：reasoning_content 是 output-only 字段；带 tool_calls 的 assistant content 须为空
    for m in out:
        assert "reasoning_content" not in m, "出向不得携带 reasoning_content"
        if m.get("tool_calls"):
            assert not m.get("content"), "带 tool_calls 的 assistant content 必须为空"


def _retained_indices(buf, out):
    """把 out 每条消息映射回 buf 下标；-1 表示凭空出现。"""
    used, idxs = set(), []
    for m in out:
        hit = -1
        for i, b in enumerate(buf):
            if i in used:
                continue
            same_ids = ([c["id"] for c in b.get("tool_calls", [])]
                        == [c["id"] for c in m.get("tool_calls", [])])
            if (b.get("role") == m.get("role")
                    and (b.get("content") or "") == (m.get("content") or "")
                    and b.get("tool_call_id") == m.get("tool_call_id")
                    and same_ids):
                hit = i
                break
        if hit >= 0:
            used.add(hit)
        idxs.append(hit)
    return idxs


def _assert_subsequence(buf, out):
    """保留下来的消息必须是 buffer 的**保序子序列**（不许重排、不许凭空造）。"""
    idxs = _retained_indices(buf, out)
    assert -1 not in idxs, "输出里出现了 buffer 中不存在的消息"
    body = idxs[1:]  # 跳过 system
    assert body == sorted(body), f"消息顺序被打乱：{body}"
    return idxs


def _assert_tail_suffix(buf, out):
    """保留的消息（除 system）必须是 buffer 的**严格后缀** —— 保最新、丢最旧。

    这条专治「裁剪方向反了」（本该从最旧端砍，却从最新端砍）的回归。
    """
    idxs = _retained_indices(buf, out)
    body = idxs[1:]
    n = len(buf)
    assert body == list(range(n - len(body), n)), (
        f"保留的不是最新的一段（应保尾丢头），实际保留下标：{body}"
    )


def _scan_slide(buf, *, aggressive):
    """扫描出「触发压缩且未清空」的 mcl，返回 (mcl, out)。

    ⚠️ 不能把档位写死成常量：`contextmgr.estimate_tokens` 在装了 tiktoken 的环境走
    cl100k_base、否则回退字符启发式，同一段文本两者可差约 2 倍（本项目 .venv 装了
    tiktoken，裸解释器没有）→ 写死 mcl / 条数的断言换环境必红。故运行时扫描定位：
      aggressive=False → 从大到小，首个触发压缩的档（保留最多：head + tail 形态）
      aggressive=True  → 从小到大，首个不清空的档（保留最少：只剩末轮）
    """
    rng = range(1000, 64001, 500) if aggressive else range(64000, 999, -500)
    for mcl in rng:
        out = _cm(buf).prepare_messages(model_context_length=mcl, user_input="")
        if 1 < len(out) < len(buf):
            return mcl, out
    raise AssertionError("扫不到「既触发压缩又没把历史清空」的档位")


def _scan_slide_cm(buf, *, aggressive):
    """同 `_scan_slide`，但额外返回构造的 ContextManager（便于检查 _truth / recall 数据源）。"""
    rng = range(1000, 64001, 500) if aggressive else range(64000, 999, -500)
    for mcl in rng:
        cm = _cm(buf)
        out = cm.prepare_messages(model_context_length=mcl, user_input="")
        if 1 < len(out) < len(buf):
            return mcl, out, cm
    raise AssertionError("扫不到「既触发压缩又没把历史清空」的档位")


# ---------------------------------------------------------------- sliding 前
def test_sliding_not_triggered_keeps_full_history_verbatim():
    """预算充足 → 不压缩：48 条历史逐条原样，system 唯一居首。"""
    buf = _messages(CONWAY)
    cm = _cm(buf)
    mcl = 32000  # hist_budget=14725 ≥ body 13969 → 不触发
    assert ContextManager._est_msgs(buf[1:]) <= _hist_budget(mcl, buf[0]["content"])

    out = cm.prepare_messages(model_context_length=mcl, user_input="")
    assert len(out) == len(buf)
    for a, b in zip(out, buf):
        assert a == b, f"历史被改写：{a.get('role')} != {b.get('role')}"
    _assert_protocol(out)


def test_sliding_not_triggered_is_readonly():
    """prepare_messages 只读 buffer：调用前后 to_list() 必须逐条相等。"""
    buf = _messages(CONWAY)
    cm = _cm(buf)
    before = [dict(m) for m in cm.to_list()]
    cm.prepare_messages(model_context_length=20000, user_input="")  # 会触发压缩
    assert cm.to_list() == before, "prepare_messages 修改了 buffer（应为只读视图）"


# ---------------------------------------------------------------- sliding 后
def test_sliding_triggered_drops_middle_turn_keeps_head_and_tail():
    """预算收紧 → 丢中间整轮：首轮 + 末轮保住，中间轮（'运行pytest' 起的 31 条）整组消失。

    真实分组：user@1 / user@7 / user@38 / user@39 → 4 组；head=1 + tail=2，
    中间 g1(idx7..37) 被整组丢弃（不拆 turn，保证 tool 协议不断裂）。
    """
    buf = _messages(CONWAY)
    mcl, out = _scan_slide(buf, aggressive=False)

    _assert_protocol(out)
    idxs = _assert_subsequence(buf, out)
    assert len(out) < len(buf)

    users = [m.get("content") for m in out if m["role"] == "user"]
    assert "加载readme." in users, "首轮 user 应保留（head 组）"
    assert "运行pytest" not in users, "中间轮 user 应被整组丢弃"
    assert "为啥edit_file失败了" in users, "末轮 user 应保留（tail 组）"

    assert idxs[-1] == len(buf) - 1, "最新一条消息必须保住"
    body = [m for m in out if m["role"] != "system"]
    assert ContextManager._est_msgs(body) <= _hist_budget(mcl, buf[0]["content"])


def test_sliding_triggered_aggressive_keeps_only_last_turn():
    """🔴 F1 后语义变更：预算极紧（连最后一轮都装不下）→ 轮内子单元降级：保住最后一轮的
    user 问题、丢弃超大的 tail tool 结果（原文仍在 _truth，可经 recall 取回）。

    旧断言「末尾 tool 回执必须保住」在 F1 下不再成立——F1 降级优先级：user 意图 > 超大 tool 输出。
    故改为：极端 aggro 下 out 末条 = 最后一条 user 问题，且被丢弃的 tool 原文仍在 _truth
    （recall 数据源，会话史不丢），协议不变量不破。
    """
    buf = _messages(CONWAY)
    mcl, out, cm = _scan_slide_cm(buf, aggressive=True)

    _assert_protocol(out)                     # assistant+tool 同子单元同进退 → 极端下仍无孤立/缺回执
    assert len(out) >= 2, "极端预算下不应清空成只剩 system"
    assert len(out) < len(buf), "应触发压缩"
    assert out[0]["role"] == "system"

    last_user = next(m for m in reversed(buf) if m["role"] == "user")
    assert out[-1]["role"] == "user", "F1：极端 aggro 保住最后一条 user 问题（意图不丢）"
    assert out[-1]["content"] == last_user["content"]

    # 被丢弃的超大 tail tool 原文必须仍在 _truth（recall 数据源），且不进入发往模型的工作集
    dropped = buf[-1]
    assert dropped["role"] == "tool", "CONWAY 末条应为 tool 结果"
    assert not any((m.get("content") or "") == (dropped.get("content") or "")
                   for m in out), "超大 tool 结果不应进入工作集（应被 F1 丢弃）"
    assert any((m.get("content") or "") == (dropped.get("content") or "")
               for m in cm.truth_list()), "被丢弃的 tool 原文应仍在 _truth（recall 数据源，会话史不丢）"

    _, light = _scan_slide(buf, aggressive=False)
    assert len(out) <= len(light), "越紧的预算应保留越少，而不是越多"
    # 注：此处不校验「out 体量 ≤ hist_budget」——F1 的语义正是「即便 hist_budget 已耗尽，
    # 也至少保住一条 user 问题（宁可略微超出预算也不让 agent 失忆）」，故该极端档下
    # out body 可能 > hist_budget，这是预期行为而非回归。


def test_sliding_triggered_should_not_wipe_whole_history():
    """🔴 F1 已修：极端预算（连最新一轮都装不下）下，整段历史不再被清空成只剩 system。

    轮内子单元降级：从最旧端丢「assistant(+tool) 子单元」、保留 user 子单元，至少保住
    system + 最近一条 user 问题，避免 agent 完全失忆。
    """
    buf = _messages(CONWAY)
    cm = _cm(buf)
    out = cm.prepare_messages(model_context_length=1000, user_input="")
    assert len(out) >= 2, (
        f"预算不足时不应把历史清成只剩 system（agent 失忆），实际 {len(out)} 条"
    )
    assert out[0]["role"] == "system"
    users = [m.get("content") for m in out if m["role"] == "user"]
    assert users, "极端预算下仍应至少保住一条 user 问题（F1 修复后不再清空历史）"


# ---------------------------------------------------------------- append user
def test_append_user_message_new_turn_not_duplicated():
    """append 新 user（真实 idx7 '运行pytest'）→ 末条即该 user，且不重复注入。"""
    buf = _messages(CONWAY)
    new_user = buf[7]                       # 真实存在的下一条 user
    assert new_user["role"] == "user"
    base = buf[:7]                          # 切到 assistant@6 为止
    cm = _cm(base + [new_user])

    out = cm.prepare_messages(model_context_length=96000,
                              user_input=new_user["content"])
    _assert_protocol(out)
    assert out[-1]["role"] == "user"
    assert out[-1]["content"].startswith(new_user["content"])

    users = [m.get("content") for m in out if m["role"] == "user"]
    base_users = [m.get("content") for m in base + [new_user] if m["role"] == "user"]
    assert len(users) == len(base_users), f"user 消息数被改写：{len(users)} != {len(base_users)}"
    assert users.count(new_user["content"]) <= 1, "新 user 被重复塞入"


def test_append_user_message_after_user_uu_pair():
    """append user 紧跟另一条 user（真实 idx38 停滞保护 + idx39 追问）→ 两条都得留。"""
    buf = _messages(CONWAY)
    prev_user, new_user = buf[38], buf[39]
    assert prev_user["role"] == "user" and new_user["role"] == "user"
    cm = _cm(buf[:39] + [new_user])

    out = cm.prepare_messages(model_context_length=96000,
                              user_input=new_user["content"])
    _assert_protocol(out)
    users = [m.get("content") for m in out if m["role"] == "user"]
    assert new_user["content"] in users, "新 append 的 user 丢了"
    assert prev_user["content"] in users, "前一条 user（停滞保护）丢了"
    assert out[-1]["content"].startswith(new_user["content"])


# ---------------------------------------------------------------- append assistant
def test_append_assistant_message_keeps_tail_and_no_user_injection():
    """append assistant（真实 idx6，834 字纯文本）→ 末尾是它，且不凭空注入 user。"""
    buf = _messages(CONWAY)
    new_asst = buf[6]
    assert new_asst["role"] == "assistant" and not new_asst.get("tool_calls")
    base = buf[:6]
    cm = _cm(base + [new_asst])

    last_user = "加载readme."                  # 等价于 agent._last_user_text(buffer)
    out = cm.prepare_messages(model_context_length=96000, user_input=last_user)

    _assert_protocol(out)
    assert out[-1]["role"] == "assistant", "末尾 assistant 被顶掉了"
    assert out[-1]["content"] == new_asst["content"], "assistant 内容被改写"
    users = [m.get("content") for m in out if m["role"] == "user"]
    assert users == [last_user], f"tool/assistant 轮不该再注入 user：{users}"


# ---------------------------------------------------------------- append tool
def test_append_tool_message_paired_with_tool_call():
    """append tool 回执（真实 idx3）→ 与其 assistant(tool_calls) 严格配对，不重复注入 user。"""
    buf = _messages(CONWAY)
    asst, tool_msg = buf[2], buf[3]
    assert asst.get("tool_calls") and tool_msg["role"] == "tool"
    cm = _cm(buf[:3] + [tool_msg])

    out = cm.prepare_messages(model_context_length=96000, user_input="加载readme.")
    _assert_protocol(out)
    assert out[-1]["role"] == "tool" and out[-1]["tool_call_id"] == tool_msg["tool_call_id"]
    assert _declared_ids(out) == _result_ids(out), "tool_call_id 与 tool_calls 未一一对齐"
    assert [m.get("content") for m in out if m["role"] == "user"] == ["加载readme."]


def test_append_tool_message_multitool_all_paired():
    """单轮 4 个 tool_calls → 4 条回执全配，且回执顺序 == 声明顺序。"""
    buf = _messages(MULTITOOL)
    asst = buf[2]
    tools = buf[3:7]
    assert len(asst["tool_calls"]) == 4 and all(t["role"] == "tool" for t in tools)
    cm = _cm(buf[:3] + tools)

    out = cm.prepare_messages(model_context_length=96000, user_input=buf[1]["content"])
    _assert_protocol(out)

    declared = _declared_ids(out)
    results = _result_ids(out)
    assert declared == results, f"多 tool 回执顺序/配对错位：{declared} vs {results}"
    assert len(results) == 4


def test_sliding_after_tool_append_pairs_intact():
    """sliding 之后，末尾 tool 回执仍与 assistant(tool_calls) 严格配对 —— OpenAI 协议铁律。

    用 head+tail 扫描（aggressive=False，保留最多）：tail 组整组保留，故末尾 tool 回执与其
    assistant(tool_calls) 一并保住、一一配对，不因滑动窗口断裂。
    （注：aggressive=True 在 F1 下会轮内子单元降级、把整段 assistant+tool 同进退丢弃，
     那种极端场景改由 test_sliding_triggered_aggressive_keeps_only_last_turn 覆盖。）
    """
    buf = _messages(MULTITOOL)
    mcl, out = _scan_slide(buf, aggressive=False)

    _assert_protocol(out)
    _assert_subsequence(buf, out)
    assert 1 < len(out) < len(buf)
    assert out[-1]["role"] == "tool"
    assert _declared_ids(out) == _result_ids(out)
    assert ContextManager._est_msgs(
        [m for m in out if m["role"] != "system"]
    ) <= _hist_budget(mcl, buf[0]["content"])


def test_compress_drops_oldest_tail_group_not_newest():
    """裁剪方向：tail 有 ≥2 组、预算只够一组时，必须留**最新**一组。

    直接压 contextmgr.compress（prepare_messages 走 COMPRESS 分支时最终调它）——
    真实 fixture 在该分支上是「等价变异体」（扫到的档位不区分），故补一条构造用例封洞。
    """
    from contextmgr.compress import CompressionStrategy, compress

    def turn(i):
        return [{"role": "user", "content": f"u{i} " + "问" * 300},
                {"role": "assistant", "content": f"a{i} " + "答" * 300}]

    body = turn(1) + turn(2) + turn(3) + turn(4)   # 4 组 → head=1 + tail=2 分支
    budget = ContextManager._est_msgs(turn(1)) + 10  # 只够装一组
    # 注意：这里必须传 compress_backend=None —— 非 None 时 compress() 会把整包直接
    # 交给后端 return，压根走不到下面的整组裁剪循环。
    out = compress(list(body), budget, CompressionStrategy.COMPRESS,
                   compress_backend=None)
    texts = " ".join((m.get("content") or "") for m in out)

    assert "u4" in texts, "最新一组被砍掉了（裁剪方向反了：该从最旧端砍）"
    assert "u3" not in texts, "留下的是倒数第二组，不是最新一组"
    assert "u1" not in texts, "head 组不该在这种预算下存活"


# ---------------------------------------------------------------- 出向清理
def test_reasoning_stripped_outgoing_but_buffer_intact():
    """reasoning_content：出向剥除（防 LM Studio 400），buffer 内保留（REPL 渲染要用）。"""
    buf = _messages(CONWAY)
    cm = _cm(buf[:3])
    cm.append("assistant", "先看一下 readme", reasoning_content="我在想该不该读文件")

    out = cm.prepare_messages(model_context_length=96000, user_input="加载readme.")
    assert all("reasoning_content" not in m for m in out), "出向仍带 reasoning_content"
    buffered = [m for m in cm.to_list() if m["role"] == "assistant"]
    assert any(m.get("reasoning_content") for m in buffered), "buffer 里的 reasoning 被误删"
