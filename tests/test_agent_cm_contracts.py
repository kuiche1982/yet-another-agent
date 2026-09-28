#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""swe_agent Agent + ContextManager × contextmgr 集成契约测试（model-free / tool-free）。

用户要求的三件事集中在这里锁死：
  A) swe_agent.management.ContextManager 的 system / user / tool / compress 行为
     —— 全部 model-free：不调 LLM、不跑 LFM 压缩后端（compress_backend=None）。
  B) swe_agent.agent.Agent 类在「假模型 + 假工具」双 fake 下的控制流契约
     —— _step 接电 prepare_messages；_apply_toolcall 写 tool 结果带 tool_call_id；
        stop action / 空响应 / 未调工具 stuck 等分支。
  C) contextmgr 侧新接口：ingest_code_file / retrieve(include_levels=True) /
     save_session + load_session round-trip / build_context / stats。

零副作用：不连 LLM / 网络 / 真实工具执行；contextmgr 用临时目录与内存态。
运行：.venv/bin/python -m pytest tests/test_agent_cm_contracts.py -q --basetemp=.pytest_tmp
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextmgr import ContextManager as RagEngine, Source  # contextmgr 侧的 RagEngine
from swe_agent.agent import Agent, RunState, _last_user_text
from swe_agent.management import ContextManager as SweCM, set_compress_backend
from swe_agent.registry import ToolRegistry
from swe_agent import models as M, roles_config as RC
from swe_agent.config import LOOP_REPEAT_THRESHOLD


# ======================================================================
# 装配：model-free（禁用 LFM 压缩后端）+ 隔离 KB（不扫工作区）
# ======================================================================
def _make_kb() -> RagEngine:
    """隔离 RagEngine：不 ingest 工作区、不连向量模型（BM25，确定性）。"""
    return RagEngine(budget_tokens=4096)


def _make_cm() -> SweCM:
    """swe_agent ContextManager，强制 model-free：backend=None + compress_backend=None，
    且把全局 _COMPRESS_BACKEND 也置空 —— 否则构造器会回落到 LFM 压缩通道（非 model-free）。
    kb 用隔离实例，避免 _get_workspace_kb(C.WORKSPACE) 把整个仓库扫进 KB。
    """
    set_compress_backend(None)
    return SweCM(kb=_make_kb(), backend=None, compress_backend=None)


# ======================================================================
# Part A：swe_agent ContextManager —— system / user / tool / compress
# ======================================================================
def test_cm_system_is_first_and_singleton():
    cm = _make_cm()
    cm.set_system("SYS-1")
    cm.append("user", "u1")
    assert cm.to_list()[0] == {"role": "system", "content": "SYS-1"}
    cm.set_system("SYS-2")  # 原地更新，不新增 system
    out = cm.to_list()
    assert sum(1 for m in out if m["role"] == "system") == 1
    assert out[0]["content"] == "SYS-2"
    # prepare_messages 也只产出 1 个 system（RAG 未命中时）
    msgs = cm.prepare_messages(model_context_length=64000, user_input="u1")
    assert [m["role"] for m in msgs].count("system") == 1
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "SYS-2"


def test_cm_user_messages_preserved_through_prepare():
    cm = _make_cm()
    cm.set_system("S")
    cm.append("user", "q1")
    cm.append("assistant", "a1")
    cm.append("user", "q2")
    msgs = cm.prepare_messages(model_context_length=64000, user_input="q2")
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    # 历史 user/assistant 全部保留，且末条是本轮 user 输入
    assert roles.count("user") == 2
    assert roles[-1] == "user" and msgs[-1]["content"] == "q2"
    assert "a1" in [m["content"] for m in msgs if m["role"] == "assistant"]


def test_cm_tool_result_carries_tool_call_id():
    """tool 结果消息必须带 tool_call_id，且与 assistant 声明的 tool_calls id 对应
    （OpenAI 协议：每个 tool_call 必须有一条结果，否则下一轮 400）。"""
    cm = _make_cm()
    cm.set_system("S")
    cm.append("user", "do it")
    # 模拟一轮 toolcall：assistant 声明 tool_calls + 紧随其后的 tool 结果
    cm.append("assistant", None, tool_calls=[
        {"id": "call_1", "type": "function",
         "function": {"name": "read_file", "arguments": "{}"}}])
    cm.append("tool", "file content here", tool_call_id="call_1")

    raw = cm.to_list()
    tool_msgs = [m for m in raw if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0].get("tool_call_id") == "call_1"
    # prepare_messages（默认大窗口，不触发压缩）原样保留 tool_call_id
    msgs = cm.prepare_messages(model_context_length=64000, user_input="do it")
    out_tool = [m for m in msgs if m["role"] == "tool"]
    assert out_tool and out_tool[0].get("tool_call_id") == "call_1"


def test_cm_compress_two_tier_preserves_tool_protocol():
    """压缩 + tool 同场：即便触发 model-free 两级压缩，tool_call_id 与 assistant
    tool_calls 也必须保留，且 system 不得重复。

    布置要点：把大体积历史放在「中间」（COMPRESS 策略本就丢弃中间、只留 head+tail+占位），
    把 tool 轮留在 tail 窗口 —— 这样压缩一定触发、且不会把 tool 轮从 tail 砍掉。
    （注意：若把大历史放在 head 且预算极紧，compress 会从 tail 头部往前砍，
     可能留下 tool 结果却砍掉其 assistant 声明 → 协议断裂；那是个独立边缘 bug，
      不在此正常态契约的断言范围内，见下方 test_cm_compress_tiny_budget_tool_protocol_bug。）
    """
    cm = _make_cm()
    cm.set_system("S")
    # head 窗口（保留）：两条小消息
    cm.append("user", "small head user")
    cm.append("assistant", "small head reply")
    # 中间大体积历史（压缩时丢弃）
    cm.append("user", "x" * 4000)
    cm.append("assistant", "y" * 4000)
    # tail 窗口（保留，最新一步调了工具）
    cm.append("user", "call the tool")
    cm.append("assistant", None, tool_calls=[
        {"id": "call_t", "type": "function",
         "function": {"name": "read_file", "arguments": "{}"}}])
    cm.append("tool", "result payload", tool_call_id="call_t")

    # 极小 model_context_length → hist_budget 很小 → 触发 _compress_two_tier（model-free）
    msgs = cm.prepare_messages(model_context_length=300, user_input="call the tool")
    # 压缩确实发生了（中间大历史被丢弃）
    assert "x" * 4000 not in " ".join(m.get("content") or "" for m in msgs)
    # 关键 1：恰好 1 个 system（压缩注入的占位已被 strip）
    assert [m["role"] for m in msgs].count("system") == 1
    # 关键 2：tool 协议完整
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    asst_calls = [tc["id"] for m in msgs
                  if m["role"] == "assistant" and m.get("tool_calls")
                  for tc in m["tool_calls"]]
    assert tool_msgs, "压缩后 tool 结果应被保留（落在 tail 窗口）"
    assert tool_msgs[0].get("tool_call_id") == "call_t"
    assert "call_t" in asst_calls, "压缩后 assistant 的 tool_calls id 应保留"


def test_cm_compress_tiny_budget_tool_protocol_preserved():
    """回归（原边缘 bug，现已修复）：单条消息巨大、预算极紧时，contextmgr.compress 的
    COMPRESS 策略必须保持 `tool` 结果与其 `assistant` tool_calls 声明成对出现 ——
    要么整组（user→assistant(tool_calls)→tool）都留，要么都砍，绝不留下孤立 tool。

    复现：大历史放 head 且预算小到 head 自身就超预算 → 整组砍掉 head，tool 轮（tail 组）整组保留。
    """
    cm = _make_cm()
    cm.set_system("S")
    cm.append("user", "x" * 4000)            # head[0]，巨大
    cm.append("assistant", "y" * 4000)        # head[1]，巨大
    cm.append("user", "call the tool")
    cm.append("assistant", None, tool_calls=[
        {"id": "call_t", "type": "function",
         "function": {"name": "read_file", "arguments": "{}"}}])
    cm.append("tool", "result payload", tool_call_id="call_t")

    msgs = cm.prepare_messages(model_context_length=300, user_input="call the tool")
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    asst_calls = [tc["id"] for m in msgs
                  if m["role"] == "assistant" and m.get("tool_calls")
                  for tc in m["tool_calls"]]
    # 修复后：tool 与 assistant 声明成对保留（head 组被砍、tail 组整组留）
    assert tool_msgs, "压缩后 tool 结果应被保留（tail 组整组保留）"
    assert tool_msgs[0].get("tool_call_id") == "call_t"
    assert "call_t" in asst_calls, "压缩后 assistant 的 tool_calls id 应成对保留（不再孤立）"


def test_cm_compress_if_needed_inplace_model_free():
    """compress_if_needed 就地压缩（互斥于 prepare_messages 的只读语义）；
    在 compress_backend=None 下必须是纯 model-free 切片，不调 LFM。"""
    cm = _make_cm()
    cm.set_system("S")
    for i in range(5):
        cm.append("user", f"msg-{i}-" + "z" * 1500)
    before = len(cm.to_list())
    cm.compress_if_needed()  # 默认阈值远低于 5×1500 token
    after = cm.to_list()
    assert len(after) <= before
    assert after[0]["role"] == "system"  # system 永不被砍


def test_cm_ingest_wrappers_delegate_to_kb():
    """swe_agent CM 的 ingest_* 包装必须真实落到 kb（串起 contextmgr 侧新接口）。

    注意：swe_agent CM 的 ingest_library / ingest_code_file 返回 self（链式），
    不返回片段计数；计数应从 kb.stats() 取。
    """
    cm = _make_cm()
    cm.ingest_library("Redis 缓存击穿用互斥锁。\n无关段：买咖啡。", doc_id="redis")
    assert cm.kb.stats()["by_source"]["library"] >= 1
    # ingest_code_file 需要真实磁盘文件（LSP provider 走 didOpen）
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write("def ping():\n    return 'pong'\n")
        path = fh.name
    try:
        cm.ingest_code_file(path)
        assert cm.kb.stats()["by_source"]["code"] >= 1
        refs = cm.kb.retrieve("ping", budget_tokens=2000, sources={Source.CODE})
        assert any("ping" in (f.text or "") for f in refs)
    finally:
        os.unlink(path)


# ======================================================================
# Part B：Agent 类 —— 双 fake（假模型 + 假工具）控制流契约
# ======================================================================
class _FakeToolRig:
    """工具全 fake：monkeypatch ToolRegistry.dispatch，对任意工具名返回固定回执，
    绝不执行真实副作用（不写盘 / 不联网 / 不跑 shell）。"""

    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(ToolRegistry, "dispatch", self._fake_dispatch)

    def _fake_dispatch(self, args, ctx=None):
        name = args.get("action")
        self.calls.append((name, dict(args)))
        return f"[fake:{name}] ok"

    def sequence(self):
        return [n for (n, _) in self.calls]


class _FakeModel:
    """模型 fake：按「脚本」返回 toolcall，引导循环走测试关注的控制流点。
    签名兼容 M.chat_toolcalls(role, messages, tools=, model_override=)。"""

    def __init__(self, steps, fallback=None):
        self.steps = list(steps)
        self.fallback = fallback
        self.calls = 0
        self.last_messages = None

    def __call__(self, role="executor", messages=None, **kwargs):
        self.calls += 1
        self.last_messages = messages
        if self.steps:
            item = self.steps.pop(0)
        elif self.fallback:
            item = self.fallback
        else:
            return {"type": "toolcalls", "tool_calls": []}
        calls = item if isinstance(item, list) else [item]
        return {
            "type": "toolcalls",
            "tool_calls": [{
                "id": f"c{self.calls}_{i}",
                "name": n,
                "arguments": json.dumps(a),
            } for i, (n, a) in enumerate(calls)],
        }


def _make_agent(monkeypatch, role="executor", max_iter=5, on_iter_end=None,
                allowed=None):
    """双 fake 装配一个 Agent：假工具 + 隔离 CM + 角色配置。"""
    _FakeToolRig(monkeypatch)
    if allowed is not None:
        monkeypatch.setattr("swe_agent.agent.ROLE_TOOLS_ALLOWED", {role: set(allowed)})
    rc = RC.make_role_config(role)
    ctx = RunState(role=role)
    ctx.cm = _make_cm()
    agent = Agent(rc, RC.single_loop(max_iter=max_iter, on_iter_end=on_iter_end), ctx=ctx)
    return agent


def test_agent_step_wires_prepare_messages(monkeypatch):
    """接电契约：_step 必须把 outgoing 交给 ContextManager.prepare_messages，
    并把其结果喂给模型调用。spy prepare_messages 验证入参与透传。"""
    agent = _make_agent(monkeypatch, "executor", max_iter=1,
                        on_iter_end=lambda c, r: "break")
    captured = {}
    orig = agent.ctx.cm.prepare_messages

    def _spy(model_context_length=64000, user_input=""):
        captured["model_context_length"] = model_context_length
        captured["user_input"] = user_input
        return [{"role": "user", "content": "SPY_OUTPUT"}]
    monkeypatch.setattr(agent.ctx.cm, "prepare_messages", _spy)

    model = _FakeModel([("read_file", {"path": "a.py"})])
    monkeypatch.setattr(M, "chat_toolcalls", model)

    agent.run([{"role": "user", "content": "QUERY"}])

    # 入参：user_input 取自 buffer 末条 user；model_context_length 取角色模型上下文窗
    assert captured["user_input"] == "QUERY"
    expected_mcl = M.model_context_length(RC.make_role_config("executor").model_id())
    assert captured["model_context_length"] == expected_mcl
    # 透传：模型实际收到的 messages 就是 prepare_messages 的返回值
    assert model.last_messages == [{"role": "user", "content": "SPY_OUTPUT"}]


def test_agent_apply_toolcall_writes_tool_result_with_id(monkeypatch):
    """_apply_toolcall 必须把 assistant 的 tool_calls 写入 buffer，并对每个 id
    追加一条带 tool_call_id 的 tool 结果（协议完整性）。"""
    agent = _make_agent(monkeypatch, "executor", allowed={"read_file"})
    meta = {"type": "toolcalls", "tool_calls": [
        {"id": "t1", "name": "read_file", "arguments": json.dumps({"path": "a.py"})}]}
    reason = agent._apply_toolcall(meta)
    assert reason in ("continue", "all_done")
    raw = agent.ctx.cm.to_list()
    asst = [m for m in raw if m["role"] == "assistant" and m.get("tool_calls")]
    tool = [m for m in raw if m["role"] == "tool"]
    assert asst and tool
    assert tool[0].get("tool_call_id") == "t1"
    assert asst[0]["tool_calls"][0]["id"] == "t1"


def test_agent_apply_toolcall_stop_action_all_done(monkeypatch):
    """终止动作（executor 的 complete）命中 → 返回 all_done 且捕获 stop_result。"""
    agent = _make_agent(monkeypatch, "executor", allowed={"complete"})
    meta = {"type": "toolcalls", "tool_calls": [
        {"id": "t1", "name": "complete", "arguments": json.dumps({"summary": "done"})}]}
    reason = agent._apply_toolcall(meta)
    assert reason == "all_done"
    assert agent.ctx.metadata.get("stop_result", {}).get("summary") == "done"


def test_agent_apply_toolcall_empty_response_continues(monkeypatch):
    """空响应（finish_reason=stop 但无 tool_calls）→ executor 尊重 stop 返回 continue，不灌 nudge。"""
    agent = _make_agent(monkeypatch, "executor")
    meta = {"type": "content", "content": "全部完成。", "finish_reason": "stop"}
    reason = agent._apply_toolcall(meta)
    assert reason == "continue"
    nudges = [m for m in agent.ctx.cm.to_list()
              if m["role"] == "user" and "请调用" in (m.get("content") or "")]
    assert not nudges, "executor 尊重 stop，不应灌'调工具'nudge"


def test_agent_apply_toolcall_model_error_on_none(monkeypatch):
    """meta=None（模型连续不可用）→ 返回 model_error，不抛异常。"""
    agent = _make_agent(monkeypatch, "executor")
    assert agent._apply_toolcall(None) == "model_error"


def test_agent_apply_toolcall_no_tool_streak_stuck(monkeypatch):
    """连续未调工具达 LOOP_REPEAT_THRESHOLD → stuck（循环防护生效，不空转到 max_iter）。"""
    agent = _make_agent(monkeypatch, "executor", max_iter=LOOP_REPEAT_THRESHOLD + 4,
                        on_iter_end=lambda c, r: "break" if r == "stuck" else None)
    # 模型每轮只回空 tool_calls（= 未调工具），不 fallback 避免无限重发
    model = _FakeModel([], fallback=None)
    monkeypatch.setattr(M, "chat_toolcalls", model)
    reason = agent.run([{"role": "user", "content": "task"}])
    assert reason == "stuck", f"应判定 stuck，实际 {reason}"
    # 第 LOOP_REPEAT_THRESHOLD 次未调工具即触顶，远早于 max_iter
    assert model.calls == LOOP_REPEAT_THRESHOLD, (
        f"stuck 应在第 {LOOP_REPEAT_THRESHOLD} 次未调工具触发，实际 {model.calls} 次模型调用")


def test_agent_single_turn_executes_every_tool_call(monkeypatch):
    """一轮多 tool_call 必须全部执行且每个 id 都回执（复用 harness 核心契约，
    但在 Agent 类层面用最简双 fake 验证）。"""
    turn = [("read_file", {"path": "a.py"}),
            ("read_file", {"path": "b.py"})]
    agent = _make_agent(monkeypatch, "executor", max_iter=1,
                        on_iter_end=lambda c, r: "break", allowed={"read_file"})
    model = _FakeModel([turn])
    monkeypatch.setattr(M, "chat_toolcalls", model)
    agent.run([{"role": "user", "content": "task"}])

    rig = ToolRegistry.dispatch  # 已被 monkeypatch；改为取计数需另存，这里直接查 buffer
    raw = agent.ctx.cm.to_list()
    tool_ids = [m["tool_call_id"] for m in raw if m["role"] == "tool"]
    asst_ids = [tc["id"] for m in raw
                if m["role"] == "assistant" and m.get("tool_calls")
                for tc in m["tool_calls"]]
    assert tool_ids == asst_ids, "每个 tool_call_id 都必须有一条对应 tool 结果"


# ======================================================================
# Part C：contextmgr 侧新接口
# ======================================================================
def test_ragengine_ingest_code_file_builds_fragments():
    """新接口 ingest_code_file：从磁盘读真实文件并按符号切分（LSP 不可用→回落 AST）。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write("def ping():\n    return 'pong'\n\n\ndef pong():\n    return 'ping'\n")
        path = fh.name
    try:
        kb = RagEngine(budget_tokens=4096)
        n = kb.ingest_code_file(path)
        assert n >= 1
        refs = kb.retrieve("ping", budget_tokens=2000, sources={Source.CODE})
        assert any("ping" in (f.text or "") for f in refs)
    finally:
        os.unlink(path)


def test_ragengine_retrieve_include_levels_returns_triples():
    """新接口 retrieve(include_levels=True)：返回 (L3, L1, L0) 三元组。"""
    kb = RagEngine(budget_tokens=4096)
    kb.ingest_library("斐波那契数列用递归实现，fib(0)=0, fib(1)=1。\n无关段：今天天气好。",
                      doc_id="fib")
    # 默认（include_levels=False）返回 L0 列表
    flat = kb.retrieve("斐波那契", include_levels=False)
    assert flat and all(not isinstance(r, tuple) for r in flat)
    # include_levels=True 返回三元组
    triples = kb.retrieve("斐波那契", include_levels=True)
    assert triples, "应命中斐波那契片段"
    for t in triples:
        assert isinstance(t, tuple) and len(t) == 3, f"应为 (L3,L1,L0) 三元组，实际 {t!r}"
        _l3, _l1, l0 = t
        assert hasattr(l0, "text"), "L0 应为 Fragment（Truth）"


def test_ragengine_save_load_session_roundtrip():
    """新接口 save_session / load_session：buffer + Session 片段落盘后原样还原。"""
    kb = RagEngine(budget_tokens=4096)
    kb.set_system("SYS")
    kb.append("user", "hello")
    kb.append("assistant", "hi there")
    assert kb.stats()["by_source"]["session"] >= 2  # 两条非 system 消息各建 Session L0

    with tempfile.TemporaryDirectory() as d:
        kb.save_session(d)
        kb2 = RagEngine(budget_tokens=4096)
        kb2.load_session(d)
        # buffer 还原
        assert kb2.buffer == kb.buffer, "buffer 应原样 round-trip"
        # Session 片段还原
        assert kb2.stats()["by_source"]["session"] >= 2


def test_ragengine_build_context_returns_messages():
    """新接口 build_context：按预算自动 RAG + 剪裁，返回组装好的 messages。"""
    kb = RagEngine(budget_tokens=2000)
    kb.set_system("你是助手。")
    kb.ingest_library("Redis 缓存击穿用互斥锁或逻辑过期。\n无关段：喝茶。", doc_id="redis")
    kb.append("user", "Redis 击穿怎么处理？")
    msgs = kb.build_context("Redis 击穿", budget_tokens=2000)
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    assert any("Redis" in (m.get("content") or "") for m in msgs)


def test_ragengine_stats_shape():
    """新接口 stats：返回结构化计数。"""
    kb = RagEngine(budget_tokens=4096)
    kb.ingest_library("doc text here", doc_id="d")
    kb.append("user", "u")
    s = kb.stats()
    for key in ("fragments", "by_source", "buffer_messages", "buffer_tokens"):
        assert key in s
    assert s["by_source"]["library"] >= 1
    assert s["buffer_messages"] >= 1


def test_ragengine_ingest_code_and_library_in_memory():
    """内存态 ingest_code / ingest_library（无需磁盘）建立三线片段。"""
    kb = RagEngine(budget_tokens=4096)
    n_code = kb.ingest_code("def bar():\n    return 1\n", name="m.py")
    n_lib = kb.ingest_library("文档内容段落一。\n\n文档内容段落二。", doc_id="doc")
    assert n_code >= 1 and n_lib >= 1
    s = kb.stats()
    assert s["by_source"]["code"] >= 1
    assert s["by_source"]["library"] >= 1


def test_last_user_text_helper():
    """模块函数 _last_user_text：取 buffer 末条 user 内容（prepare_messages 的 RAG query）。"""
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "last question"},
    ]
    assert _last_user_text(msgs) == "last question"
    assert _last_user_text([{"role": "assistant", "content": "x"}]) == ""
