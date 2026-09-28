#!/usr/bin/env python3
"""Analyzer 收敛与降级契约测试（双 fake：model-free + tool-free）。

2026-09-03 修复后，run_analyzer 的四条控制流分支必须用确定性测试钉死，避免重蹈
「靠跑完整 e2e 才撞出来、且日志没有任何根因」的覆辙（e2e 01_fizzbuzz 里 analyzer
被软空响应静默旁路，日志上零失败痕迹）。

覆盖分支：
  R1  软空响应（模型不可达）→ 主循环 model_dead → 换 fallback 模型【重跑完整主循环】，
      而非旧行为（同 prompt 死等 + 孤立 1 次强制调用）。
  R4  模型可达但不收敛（重复动作 / 只吐散文）→ 收尾失败后，用【实际观察到的工具事实】
      合成最小摘要；一个事实都没有才返回空串（保持显式「未产出」语义，不再静默旁路）。
  A   正常路径：模型直接 finish_analysis → 返回非空结构化摘要。

所有测试不连 LLM / 不真实读盘 / 不联网：
  - ScriptedModel：monkeypatch M.chat_toolcalls，按脚本返回 toolcall 或软空；
  - 工具走 ToolRegistry.dispatch 整体桩（注册期已捕获工具 run，monkeypatch T.* 不生效；tool-free）；
  - _workspace_listing 也换成 canned 字符串（离线、确定性）。

运行：uv run pytest tests/test_analyzer_contracts.py -q
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import swe_agent.models as M
import swe_agent.roles as R
import swe_agent.config as C
import swe_agent.roles_config as RC
import swe_agent.management as MG
from swe_agent.management import RagEngine
from swe_agent.registry import ToolRegistry


def _finish_meta(summary, key_findings=None):
    """构造一个 finish_analysis 的 toolcall 返回（meta）。"""
    args = {"action": "finish_analysis", "summary": summary,
            "key_findings": key_findings or []}
    return {
        "type": "toolcalls",
        "actions": [{"action": "finish_analysis"}],
        "tool_calls": [{"id": "f1", "name": "finish_analysis",
                        "arguments": json.dumps(args, ensure_ascii=False)}],
    }


def _read_meta(path):
    """构造一个 read_file 的 toolcall 返回（meta）。"""
    args = {"action": "read_file", "path": path}
    return {
        "type": "toolcalls",
        "actions": [{"action": "read_file"}],
        "tool_calls": [{"id": "r1", "name": "read_file",
                        "arguments": json.dumps(args, ensure_ascii=False)}],
    }


class ScriptedModel:
    """按 steps 顺序返回 toolcall 脚本；耗尽后按 fallback 循环（模拟弱模型重发）。

    与 test_harness_contracts.ScriptedModel 同范式，但额外接受 model_override 以区分
    主模型 / fallback 模型轮次（run_analyzer 会在两次 _run_pass 中传入不同 model_override）。
    """

    def __init__(self, steps, fallback=None, by_model=None):
        self.steps = list(steps)
        self.fallback = fallback
        self.by_model = by_model or {}     # model_override -> 该模型专属 steps（先消耗）
        self.calls = 0

    def __call__(self, role="analyzer", messages=None, temperature=0.3, max_tokens=None,
                 tools=None, model_override=None, tool_choice=None):
        self.calls += 1
        bucket = self.by_model.get(model_override)
        if bucket is not None and bucket:
            item = bucket.pop(0)
        elif self.steps:
            item = self.steps.pop(0)
        elif self.fallback:
            item = self.fallback
        else:
            item = None
        if item is None:
            # 模拟软空响应（server 返 200 但无 tool_calls 无 content）：交给统一 loop 的
            # soft-empty 容忍逻辑（empty_streak 累计超 ANALYZER_SOFT_EMPTY_LIMIT 才 model_dead），
            # 不再由 ScriptedModel 直接返回 None 触发即时 model_error。
            return {"type": "prose", "content": ""}
        calls = list(item) if isinstance(item, list) else [item]
        return {
            "type": "toolcalls",
            "actions": [{"action": n} for (n, _a) in calls],
            "tool_calls": [{
                "id": f"c{self.calls}_{i}",
                "name": n,
                "arguments": json.dumps(a, ensure_ascii=False),
            } for i, (n, a) in enumerate(calls)],
        }


def _fake_is_empty(meta):
    if meta is None:
        return True
    if not isinstance(meta, dict):
        return False
    return not meta.get("tool_calls") and not meta.get("content")


class _FakeToolRig:
    """工具全 fake：monkeypatch ToolRegistry.dispatch（注册期已捕获工具 run，
    patch swe_agent.tools.* 不生效），对任意工具名返回确定性回执，绝不执行真实副作用
    （不写盘 / 不联网 / 不读真实文件）。"""
    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(ToolRegistry, "dispatch", self._fake_dispatch)

    def _fake_dispatch(self, args, ctx=None):
        name = args.get("action")
        self.calls.append((name, dict(args)))
        if name == "read_file":
            return "def fizzbuzz(n):\n    return []"
        if name == "grep":
            return ""
        if name == "glob":
            return "src/fizzbuzz.py"
        return f"[fake:{name}] ok"

    def sequence(self):
        return [n for (n, _) in self.calls]


def _install_common(monkeypatch):
    """所有测试共用的 fixture：隔离 make_agent 的重型链路 + 模型 / 工具 / 工作区清单 fake。

    run_analyzer 现走统一 Agent loop（make_agent → ContextManager + 条件 hook），故必须桩掉：
      - M.model_context_length / M.model_load_unload：loop 内部会查；
      - M.MODELS={}：短路 make_agent 的 lmstudio 分支（否则会 set_compress_backend 触发 LFM）；
      - MG._get_layered_kb：避免 rebuild 全仓库扫 RAG；
      - MG.set_compress_backend：中性化（全局 _COMPRESS_BACKEND 保持 None，压缩不触发）；
      - ToolRegistry.dispatch：工具全 fake（注册期已捕获工具 run，patch T.* 不生效）。
    """
    monkeypatch.setattr(M, "role_model_id", lambda role: "Ling-3.0-Tiny")
    monkeypatch.setattr(M, "is_empty", _fake_is_empty)
    monkeypatch.setattr(M, "model_context_length", lambda mid: 64000)
    monkeypatch.setattr(M, "model_load_unload", lambda mid: False)
    monkeypatch.setattr(M, "MODELS", {})
    monkeypatch.setattr(MG, "_get_layered_kb", lambda: RagEngine(budget_tokens=4096))
    monkeypatch.setattr(MG, "set_compress_backend", lambda fn=None: None)
    # 工具全 fake（tool-free）
    _FakeToolRig(monkeypatch)
    monkeypatch.setattr(R, "_workspace_listing", lambda: "src/fizzbuzz.py")
    # judge 默认 mock：返回 yes（避免误触真实 LM Studio；个别测试再覆盖为 no/notsure）
    monkeypatch.setattr(M, "judge", lambda kind, content, model_override=None: ("yes", "ok"))


# ----------------------------------------------------------------------
# A：正常路径 —— 模型直接交卷，返回非空结构化摘要
# ----------------------------------------------------------------------
def test_analyzer_happy_path_submits(monkeypatch):
    _install_common(monkeypatch)
    monkeypatch.setattr(M, "role_fallback", lambda role: None)
    model = ScriptedModel([("finish_analysis", {"action": "finish_analysis",
                                                 "summary": "实现 fizzbuzz(n) 返回列表",
                                                 "key_findings": ["3→Fizz", "5→Buzz"]})])
    monkeypatch.setattr(M, "chat_toolcalls", model)

    out = R.run_analyzer("用 Python 实现 fizzbuzz")
    assert out, "正常路径应返回非空摘要"
    assert "fizzbuzz" in out
    assert "Fizz" in out


# ----------------------------------------------------------------------
# R1：软空响应（主模型不可达）→ 换 fallback 模型重跑完整主循环并交卷
# 旧行为：同 prompt 死等 + 孤立 1 次强制调用，失手即返回空串（analyzer 被旁路）。
# ----------------------------------------------------------------------
def test_analyzer_soft_empty_then_fallback_reruns(monkeypatch):
    _install_common(monkeypatch)
    monkeypatch.setattr(M, "role_fallback", lambda role: "Ling-3.0-Tiny")
    # 主模型（model_override=None）连发软空 → model_dead；
    # fallback 模型（model_override="Ling-3.0-Tiny"）重跑并交卷。
    model = ScriptedModel(
        steps=[],
        by_model={
            None: [None, None],   # 主模型软空（每次 _run_pass 还会重试 1 次，故 2 轮耗尽）
            "Ling-3.0-Tiny": [("finish_analysis", {"action": "finish_analysis",
                                                    "summary": "从零实现 fizzbuzz",
                                                    "key_findings": []})],
        },
    )
    monkeypatch.setattr(M, "chat_toolcalls", model)

    out = R.run_analyzer("用 Python 实现 fizzbuzz")
    assert out, "主模型软空后应换 fallback 重跑并交卷，不应返回空串（旧 bug）"
    assert "fizzbuzz" in out
    # 证明 fallback 主循环确实被重跑（不仅是孤立 1 次强制调用）
    assert model.by_model["Ling-3.0-Tiny"] == [], "fallback 模型主循环未被完整消费"


# ----------------------------------------------------------------------
# R4-重复：模型反复重发同一探查 → 触发无进展收敛 → 收尾失败 → 用观察事实合成摘要
# ----------------------------------------------------------------------
def test_analyzer_repeat_actions_synthesizes_from_observations(monkeypatch):
    _install_common(monkeypatch)
    monkeypatch.setattr(M, "role_fallback", lambda role: None)  # 收尾无 fallback → 走 R4
    # 同一 read_file 连发，触发 ANALYZER_REPEAT_THRESHOLD(=4) 收敛 → stuck → R4 合成。
    # 显式 5 个 step + fallback 兜底，确保 consec_repeat 累计到阈值（弱模型重复下发同一探查）。
    model = ScriptedModel(
        [("read_file", {"action": "read_file", "path": "src/fizzbuzz.py"})] * 5,
        fallback=("read_file", {"action": "read_file", "path": "src/fizzbuzz.py"}),
    )
    monkeypatch.setattr(M, "chat_toolcalls", model)

    out = R.run_analyzer("用 Python 实现 fizzbuzz")
    assert out, "有工具观察事实时应合成最小摘要，不应静默返回空串"
    assert "analyzer 未提交" in out, "应带上 R4 兜底前缀，明确这是事实摘要而非模型结论"
    assert "已读取" in out, "摘要须来自实际工具观察（读了哪些文件），不得编造"


# ----------------------------------------------------------------------
# R4-散文 + judge：模型只吐散文、judge 判 no（空/占位符）→ 多次回灌后仍不过
# → 不采纳被拒散文，显式返回空串（保持与「静默旁路」区分的显式空语义）。
# ----------------------------------------------------------------------
def test_analyzer_prose_only_judge_rejects_returns_empty(monkeypatch):
    _install_common(monkeypatch)
    monkeypatch.setattr(M, "role_fallback", lambda role: None)
    monkeypatch.setattr(M, "judge", lambda kind, content, model_override=None: ("no", "只是客套话，无实质调研"))

    def prose_model(role="analyzer", messages=None, tools=None, model_override=None,
                    tool_choice=None, **kw):
        return {"type": "prose", "content": "我觉得应该先理解需求……"}

    monkeypatch.setattr(M, "chat_toolcalls", prose_model)

    out = R.run_analyzer("用 Python 实现 fizzbuzz")
    assert out == "", "judge 多次判 no 后应显式返回空串（不采纳被拒散文），而非伪造内容"


# ----------------------------------------------------------------------
# R2-无效轮计数：软空阈值由 BUILD 常量控制，不写死在 prompt
# ----------------------------------------------------------------------
def test_analyzer_soft_empty_limit_is_configurable(monkeypatch):
    _install_common(monkeypatch)
    monkeypatch.setattr(M, "role_fallback", lambda role: None)
    # 把阈值临时调高，验证常量确实驱动行为（软空未超阈值时不该立刻 model_dead）。
    # 用 4 个软空 + 1 个交卷：soft-empty 累计空响应，未超 ANALYZER_SOFT_EMPTY_LIMIT
    # 前循环继续，第 5 轮才交卷；若阈值仍=1（旧值），第 2 轮软空就 model_dead 提前放弃。
    monkeypatch.setattr(C, "ANALYZER_SOFT_EMPTY_LIMIT", 5)
    model = ScriptedModel([None, None, None, None,
                           ("finish_analysis", {"action": "finish_analysis",
                                                "summary": "迟到的交卷", "key_findings": []})])
    monkeypatch.setattr(M, "chat_toolcalls", model)

    out = R.run_analyzer("用 Python 实现 fizzbuzz")
    assert out, "软空未超阈值时应继续尝试并最终交卷"
    # 高阈值下模型被调用多次后才交卷，证明没有在第 1 次软空就放弃
    assert model.calls > 1, f"模型仅被调用 {model.calls} 次，疑似阈值未生效（软空即放弃）"


# ----------------------------------------------------------------------
# Judge 接受有效调研：模型用散文交出实质摘要，judge 判 yes → 采纳为交付，
# 且不触发外部「required 强制 finish_analysis」门禁（避免弱模型 required 下挂死）。
# 取代原 F3 长度阈值：判定交 judge 模型按显式标准，而非按字符数。
# ----------------------------------------------------------------------
def test_analyzer_judge_accepts_valid_research(monkeypatch):
    _install_common(monkeypatch)
    monkeypatch.setattr(M, "role_fallback", lambda role: "Ling-3.0-Tiny")
    forced_calls = []  # 记录是否触发了 required 强制 finish_analysis

    def prose_model(role="analyzer", messages=None, tools=None, model_override=None,
                    tool_choice=None, **kw):
        if (tool_choice and isinstance(tool_choice, dict)
                and tool_choice.get("type") == "function"
                and tool_choice.get("function", {}).get("name") == "finish_analysis"):
            forced_calls.append(1)  # 门禁若触发，记一笔
            return {"type": "prose", "content": "仍然不交卷"}
        # 实质性调研摘要（概念解释 + 需求说明俱备）→ judge 应判 yes
        return {"type": "prose", "content":
                "斐波那契需求：递推式 F(n)=F(n-1)+F(n-2)，边界 F(0)=0/F(1)=1；n<0 抛 ValueError；"
                "实现用整数递推放 src/fizzbuzz.py；测试 pytest 覆盖 n=0/1/5/10 与负数异常。"}

    monkeypatch.setattr(M, "chat_toolcalls", prose_model)
    # 默认 judge mock 返回 ("yes", "ok") → 散文被采纳

    out = R.run_analyzer("用 Python 实现 fizzbuzz")
    assert out, "judge 判 yes 的实质性散文应被采纳为交付"
    assert "斐波那契" in out, "交付内容须包含模型写好的调研摘要"
    assert forced_calls == [], ("findings 已采纳 → 强制 finish_analysis 门禁不得触发"
                                 "（否则弱模型 required 下会空转 / 挂死）")


# ----------------------------------------------------------------------
# Judge notsure 重试：judge 连续返回 notsure → 保持 temp=0 同 prompt 重试（不升温），
# 耗尽 JUDGE_MAX_RETRY 后降级为 no → 不采纳，显式空串。
# ----------------------------------------------------------------------
def test_analyzer_judge_notsure_retries_then_empty(monkeypatch):
    _install_common(monkeypatch)
    monkeypatch.setattr(M, "role_fallback", lambda role: None)
    monkeypatch.setattr(C, "JUDGE_MAX_RETRY", 2)
    monkeypatch.setattr(M, "judge", lambda kind, content, model_override=None: ("notsure", "无法判定"))

    def prose_model(role="analyzer", messages=None, tools=None, model_override=None,
                    tool_choice=None, **kw):
        return {"type": "prose", "content": "斐波那契需求：F(n)=F(n-1)+F(n-2)。"}

    monkeypatch.setattr(M, "chat_toolcalls", prose_model)

    out = R.run_analyzer("用 Python 实现 fizzbuzz")
    assert out == "", "judge 持续 notsure 耗尽重试后应显式空串，不采纳被拒散文"

