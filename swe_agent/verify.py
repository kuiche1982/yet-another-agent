#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/verify.py —— 文本描述 → 模型独立验证（第 4 道验收闸门）

两阶段：
1) planner 已在 roles.py 生成 verify_points（自然语言验收点），落到 GLOBAL_STATE["verify_points"]
   与 ./.swe_verify.json；本模块只负责「消费」它们。
2) run_tester：独立只读 agent（tester 角色，只用 read_file/glob/grep/shell + finish_verify），
   拿到需求 + 验收点，逐条实际运行/读取并引用证据，最后 finish_verify 提交判定。

设计要点（对齐 forge 教训 + 5×Why）：
- 独立性 = 利益冲突隔离：写代码的人验自己会放水。tester 是独立 loop、独立历史、只读工具集，
  BUILD 层用 ROLE_TOOLS['tester'] 硬控（不含任何 write/edit）；loop 内再二次校验动作名。
- 证据约束防幻觉：finish_verify 必须带 evidence（真实 shell 输出 / 文件片段），无证据 pass 视作作弊；
  tester 漏提交的验收点补判 skipped（未产出结构化判定，不计入失败、不触发 loop1 重跑）。
  单条验收点若因【文件读不到 / 只读工具不可用】而【无法实际核对】，该条 verdict 标 skipped，
  且 evidence 写【20 字以内】简要原因（如「文件不可读，无法核对」）——不猜 pass/fail、不展开。
  这就是 Generator-Critic 的接地原则：只有明确的 fail 用例才令 executor 重跑。
- 工具闸门：run_tester 入口先校验只读工具（read_file/glob/grep/shell）是否都注册；缺失直接
  skipped——tester 读不到代码只能盲判、且易空转，绝不进模型循环，避免把「工具缺失」误判成
  「代码缺陷」反复重跑 loop1（2026-09-02 复盘）。BUILD 层应确保工具注册，guard 是兜底防护。
- 时机：默认只在 done 时作为第 4 道闸门（与确定性 pytest 闸门叠加）；另提供 verify 工具供 executor 随时自测。
- 模型可配置：FORGE_TESTER_MODEL（空=复用 EXECUTOR_MODEL，即 qwen-7b；"off"=关闭）。
"""

import json
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from . import config as _C
from swe_agent.log import logger, _clip
from . import models as M
from .state import GLOBAL_STATE as _GLOBAL_STATE
from .registry import ActionContext, ToolRegistry
from . import roles_config as _RC
from .agent import Agent, RunState, LoopConfig

_TESTER_TOOLS = {"read_file", "glob", "grep", "shell", "finish_verify"}

# tester 独立验收的「核对能力」依赖这些只读工具；finish_verify 是终止动作（必注册）。
# 若只读工具缺失，tester 读不到实现代码、只能凭任务描述盲判（证据无价值），且弱模型可能
# 陷入「读不到就一直试」的空转 → supervisor 把「工具缺失」误判成「代码缺陷」反复重跑 loop1。
# 见 run_tester 入口的 _tester_tools_ready 闸门。
_TESTER_REQUIRED_READ_TOOLS = {"read_file", "glob", "grep", "shell"}

_TESTER_SYSTEM = """你是独立验收 agent（tester）。你的唯一职责是【独立裁判】：这份代码不是你写的，
你也绝不能写——你只负责对照「需求」与「验收点」，去查看真正的实现代码与运行结果，判断需求是否被满足。
你绝不当运动员，也不兼当裁判和下场选手。

你拥有只读工具：read_file / glob / grep / shell（shell 仅用于只读检查命令，如 ls/grep/cat 查看文件；绝不用来运行交付的程序或重跑 pytest）。你没有、也不能调用任何写文件/改文件工具。

所有路径相对于工作目录（即你的 cwd，直接写相对路径，如 `src/fib.py`；严禁带工作目录前缀、严禁绝对路径, 严谨更换目录， 每个文件只允许读取一次）。

逐条核对每个验收点的方法：
1. 先按验收点的描述，定位它对应的实现代码：用 grep / read_file 找到相关的类/函数定义，
   读它的真实签名与逻辑，核对是否实现了该验收点要求的行为——
   不是凭印象、也不是信任 executor 的说法，而是看【代码本体】到底写了什么。
2. 运行时/行为类验收点（如 CLI 可运行、算法输出正确、边界不崩）：通过【读代码 + 读测试】判定，
   不靠 shell 运行程序——运行目标程序可能卡在 stdin 输入导致超时，且单杠 pytest 已在 loop 2 由 harness 跑过。
   具体做法：读实现确认逻辑（如 main() 不依赖交互式 input、有 `if __name__=='__main__'` 入口、边界处理到位）；
   读测试文件确认有真实断言覆盖该行为（而非空测试或只 import 不 assert）。
3. 【以事实为基础】每条验收点都必须有【可核对的证据】：来自你实际读到的代码片段，或真实命令输出。
   没有证据就判 pass = 作弊；禁止「我认为应该没问题」「应该没问题」式结论——所有判定都必须能指到具体读到的东西，
   不能凭印象或信任 executor 的说法。
   【关键】若某验收点所需的实现文件读不到 / 只读工具不可用 / 命令无有效输出，导致你【无法实际核对】该用例状态，
   则这条验收点 verdict 必须标 skipped（不得猜 pass 或 fail），并在 evidence 写【20 字以内】的简要原因，
   例如「文件不可读，无法核对」「缺少只读工具，无法验证」。evidence 超过 20 字视为不合规。

提交规则（强制）：
4. 【只做一轮验证】。聚焦「读实现 + 读测试」，不要反复空跑 shell 或重复 read 同一文件；
   工具调用轮次上限由 harness 在 BUILD 层统一管控（max_iter），非提示词约束。若目标文件不存在 / 命令无有效输出，
   直接基于现有证据判对应验收点 fail 或 skipped 并 finish_verify，不要陷入「读不到就一直试」的死循环。
5. 【无论如何都必须调用 finish_verify 提交结论】——这是你唯一的出口。即使你判断不了、或中途认为无需继续，
   也必须调用 finish_verify 收尾，绝不能只回一段散文就结束（那会被 harness 视为「未产出结果」而跳过验收，
   不等于 pass，也不触发修复重跑）。
6. 每条验收点 verdict 取三值之一：pass（有证据确认满足）/ fail（有证据确认不满足）/ skipped（证据不足或无法判定，不要猜）。
   - pass / fail 的 evidence 写你实际读到的证据片段或命令输出（要能指到具体东西）。
   - skipped（因文件不可读 / 工具不可用 / 无法判定）的 evidence 必须【20 字以内】写明原因（如「文件不可读，无法核对」），
     不展开、不写长段；超过 20 字视为不合规。
7. 【验收点本身可能与事实不符，不要替它背锅】。你是只读裁判，无法修改代码去迎合错误的验收点。
   若你用多种方法（读实现 + 多次运行/读取同一处）都确认：代码实际行为正确、是【验收点描述本身写错】
   （例如验收点写 fizzbuzz(15) 第 11 元素应为 "Jazz"，但代码实际输出 "11"），则这条验收点 verdict 标 skipped，
   evidence 写【足够强的原因】并直接引用事实（如「运行 fizzbuzz(15)[10] 实际为 '11'，验收点写 'Jazz' 与事实不符」）。
   随后照常调用 finish_verify 提交——无论验收点对错，finish_verify 都是你唯一出口，绝不因此空转不交结论。"""


def _delivery_summary() -> Tuple[str, bool]:
    """构建交付摘要供 tester judge 评判，并做客观前置检查（非空实现/测试文件 + pytest 收集>0）。

    返回 (summary_text, objective_ok)：objective_ok=False 时交付明显不达标（空文件/空测试/无收集），
    调用方（verify_gate）可直接 fail，不浪费模型调用。tester judge 的「有效」主观判定在客观达标后才上。
    """
    ws = _C.WORKSPACE
    py_files = [p for p in ws.rglob("*.py") if p.is_file()]
    src_files = [p for p in py_files if not p.name.startswith("test")]
    test_files = [p for p in py_files if p.name.startswith("test")]
    src_nonempty = [p for p in src_files if p.stat().st_size > 0]
    test_nonempty = [p for p in test_files if p.stat().st_size > 0]
    collected = -1
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", str(ws)],
                           capture_output=True, text=True, timeout=60, cwd=str(ws))
        m = re.search(r"(\d+)\s+tests? collected", r.stdout)
        collected = int(m.group(1)) if m else 0
    except Exception:
        collected = -1
    summary = (f"实现文件：{len(src_nonempty)} 个非空 / {len(src_files)} 总；"
               f"测试文件：{len(test_nonempty)} 个非空 / {len(test_files)} 总；"
               f"pytest 收集用例数：{collected}。\n"
               f"非空实现文件（最多10）：{[str(p.relative_to(ws)) for p in src_nonempty][:10]}\n"
               f"非空测试文件（最多10）：{[str(p.relative_to(ws)) for p in test_nonempty][:10]}")
    objective_ok = bool(src_nonempty) and bool(test_nonempty) and collected > 0
    return summary, objective_ok


def _resolve_model(env_val: str, default: str) -> Optional[str]:
    v = (env_val or "").strip()
    if v.lower() in ("off", "none", "false"):
        return None
    return v or default


def _load_points() -> Tuple[str, List[Dict[str, Any]]]:
    """优先取 GLOBAL_STATE['verify_points']（planner 内存最新），回退读磁盘 .swe_verify.json。"""
    pts = _GLOBAL_STATE.get("verify_points") or []
    if pts:
        return str(_GLOBAL_STATE.get("goal", "")), list(pts)
    f = _C.WORKSPACE / ".swe_verify.json"
    if f.exists():
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
            return str(obj.get("goal") or ""), list(obj.get("verify_points") or [])
        except Exception:
            return "", []
    return "", []


def _render_task(goal: str, points: List[Dict[str, Any]],
                 contract: Optional[Dict[str, Any]] = None) -> str:
    lines = [f"【需求】\n{goal}\n"]
    if contract:
        mods = contract.get("modules") or []
        iface = contract.get("interface") or []
        if mods:
            lines.append("【实现契约 · 模块与公开签名】（按 path 用 read_file/grep 查看对应实现）：")
            for m in mods:
                pubs = m.get("public") or []
                pub_str = "；".join(pubs) if pubs else "（无公开签名）"
                lines.append(f"  - {m.get('path')}：{pub_str}")
            lines.append("")
        if iface:
            lines.append("【接口签名 interface（全部公开函数最终签名，与上方 public 逐字一致）】：")
            for sig in iface:
                lines.append(f"  - {sig}")
            lines.append("")
    lines.append("【验收点】（逐条核对，必须引用证据）：")
    for p in points:
        hint = f"\n   建议核对：{p['check_hint']}" if p.get("check_hint") else ""
        lines.append(f"  #{p['id']} {p['point']}{hint}")
    lines.append("\n请做【一轮】验证：全部核对完后【必须】调用 finish_verify 提交结论"
                 "（无论结论如何都要提交）。\n判定铁律：\n"
                 "- 一切以事实为基础：以你【实际读到的代码/输出】为准，没有读到的东西不要判 pass。\n"
                 "- 某验收点因【文件读不到 / 只读工具不可用 / 命令无有效输出】而【无法实际核对】时，"
                 "该条 verdict 必须标 skipped，且 evidence 写【20 字以内】简要原因"
                 "（如「文件不可读，无法核对」），不得猜 pass/fail。")
    return "\n".join(lines)


def _normalize_results(raw: Any, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        raw = []
    out = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        vid = r.get("id")
        verdict = str(r.get("verdict") or "").strip().lower()
        if verdict == "pass":
            v = "pass"
        elif verdict == "skipped":
            v = "skipped"
        else:
            # 'fail' 或无法识别的 verdict 一律 fail（保守：视为未满足）
            v = "fail"
        out.append({
            "id": int(vid) if isinstance(vid, int) else len(out) + 1,
            "verdict": v,
            "evidence": str(r.get("evidence") or "").strip()[:600],
        })
    # 补上 tester 漏提交的验收点：视为 skipped（未产出结构化判定），不计入失败、
    # 不触发 loop1 重跑——只有明确的 fail 用例才令 loop1 重跑（见 run_tester / _l1_gate）。
    got = {r["id"] for r in out}
    for p in points:
        if p["id"] not in got:
            out.append({"id": p["id"], "verdict": "skipped",
                        "evidence": "tester 未对此验收点提交判定（跳过，未计入失败）。"})
    return out


def _tester_on_iter_end(ctx: "RunState", reason: str) -> Optional[str]:
    """tester 内部循环闸门：模型一旦发出 finish_verify（终止动作，_apply_toolcall 返回
    'all_done'），立即终止循环并回报告 supervisor，绝不重跑。

    不加这道闸门时，_run_loop 把 all_done 当 continue 又拉回一轮，弱模型（lfm2.5-2.6b）
    看到「已收到 3 条验收判定」仍被重新提问，就反复重发 finish_verify，直到 loop-guard
    判 stuck 或撑满 max_iter——e2e 实测在 tester 阶段空转耗满 20min（2026-09-02）。

    2026-09-02 加固：循环防护触发（reason=="stuck"，连续重复动作未收敛）时直接返回 "break"
    终止本步循环——不再「换策略续跑」耗满 max_iter。tester 侧停滞是模型问题、非代码缺陷，
    由 run_tester 据此降级为 skipped（不触发 loop1 重跑），避免惩罚正确代码。
    """
    if reason == "all_done":
        return "done"
    if reason == "stuck":
        # 重复动作未收敛：直接截断，不续跑。run_tester 收到 "stuck" 降级为 skipped。
        return "break"
    return None


def _tester_tools_ready() -> Tuple[bool, List[str]]:
    """检查 tester 独立验收所需的只读工具是否都已注册。

    tester 的验收价值来自「真读实现代码核对」（read_file/glob/grep/shell）。若这些工具
    未注册（如当前 ToolRegistry 仅注册了 finish_verify），tester 实际读不到代码、只能凭
    任务描述盲判，证据无独立价值；且弱模型可能陷入「读不到就一直试」的空转。BUILD 层必须
    保证注册齐全；若缺失，run_tester 直接返回 skipped——supervisor 见 skipped 即 done，
    不重跑，避免把「工具缺失」误判成「代码缺陷」而陷入 loop1 重跑死循环（2026-09-02 复盘）。
    """
    registered = set(ToolRegistry.names())
    missing = sorted(t for t in _TESTER_REQUIRED_READ_TOOLS if t not in registered)
    return (not missing), missing


def run_tester(model: str, max_iter: int = _C.MAX_TESTER_ITER) -> Tuple[str, str, List[Dict[str, Any]]]:
    """独立只读 agent 逐条验证验收点。返回 (verdict, detail, results)。

    verdict ∈ pass / fail / skipped / no_points / error。results 为每条验收点的判定。
    只有 fail（存在明确失败用例）才令 supervisor 重跑 loop1；skipped / no_points
    均视为「本轮未产出失败证据」，不重跑——避免把 tester 自身收敛问题误罚成代码缺陷。

    重构：走统一 Agent（TOOLCALL 模式，RoleConfig 来自 roles_config）。tester 的
    工具集 / 终止动作（finish_verify）/ 只读硬控全在 RoleConfig + Agent._apply_toolcall
    统一处理；finish_verify 的 results 由 Agent 捕获到 ctx.metadata["stop_result"] 后归一化。
    """
    def _ret(verdict: str, detail: str, results: List[Dict[str, Any]]):
        # 永远打印 tester 最终判定（verdict + 原因 + 配额 + 用例数），
        # debug 阶段保留现场：旧代码触顶 skipped 时日志零数据，无法定位「为何不交卷」。
        logger.critical('tester_done verdict=%s reason=%s max_iter=%s n_results=%s', verdict, _clip(detail, 140), max_iter, len(results))
        return verdict, detail, results

    goal, points = _load_points()
    if not points:
        return _ret("no_points", "未生成验收点，跳过独立验收（确定性闸门仍生效）。", [])
    # 闸门：tester 只读工具未注册齐全 → 无法真读代码核对，直接 skipped（不进模型循环、
    # 不触发 supervisor 重跑）。这是 2026-09-02 复盘发现的「tester 工具缺失→空转→loop1
    # 死循环」的根因防护——BUILD 层应确保工具注册；缺失时优雅降级而非拖死 harness。
    ready, missing = _tester_tools_ready()
    if not ready:
        return _ret("skipped", (f"tester 所需只读工具未注册完全（缺失：{missing}），"
                           f"独立验收无法真正读代码核对，跳过验收（不触发 loop1 重跑）。"), [])
    # 注入 planner 契约（modules 路径 + interface 签名），让 tester 直接定位对应实现代码
    contract = _GLOBAL_STATE.get("plan") or {}
    # 统一 Agent：role="tester"，model_override=model（FORGE_TESTER_MODEL 或 EXECUTOR_MODEL）
    rc = _RC.make_role_config("tester", model_override=model)
    ctx = RunState(role="tester")
    # 关键：tester 内部循环在 finish_verify 命中后必须立即终止（on_iter_end 返回 done），
    # 否则 _run_loop 把 all_done 当 continue 反复重跑，弱模型原地空转（见 _tester_on_iter_end）。
    agent = Agent(rc, _RC.single_loop(max_iter=max_iter, on_iter_end=_tester_on_iter_end), ctx=ctx)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": rc.system_prompt},
        {"role": "user", "content": _render_task(goal, points, contract)},
    ]
    reason = agent.run(messages)
    # 模型不可达（chat_toolcalls 重试后仍 None）→ 硬崩溃
    if reason == "model_error":
        return _ret("error", "tester 模型调用失败（无 tool call 且无 content），独立验收无法进行。", [])
    # 循环防护触发（tester 重复动作未收敛）：tester 自身停滞，非代码缺陷，
    # 降级为 skipped，不触发 loop1 重跑——否则会误罚正确代码（2026-09-02 复盘）。
    if reason == "stuck":
        return _ret("skipped", ("tester 触发循环防护（重复动作未收敛），未提交 finish_verify，"
                           "跳过独立验收（不触发 loop1 重跑）。"), [])
    # 取终止动作捕获的 results
    stop = ctx.metadata.get("stop_result")
    if not stop:
        # 模型以 finish_reason=stop 收尾但未提交 finish_verify（未产出结构化验收结果）
        # → 视为 skipped，不触发 loop1 重跑——避免弱模型空转到 max_iter 才返 fail 的死循环
        # （executor 反复空响应驱动，见 2026-09-02 复盘）。supervisor 见到 skipped 直接 done。
        if ctx.metadata.get("last_finish_reason") == "stop":
            return _ret("skipped", "tester 以 finish_reason=stop 收尾但未提交 finish_verify（未产出验收结果，跳过独立验收）。", [])
        # 触顶未交卷：tester 自身能力/收敛问题，非代码缺陷 → skipped（不触发 loop1 重跑）。
        # 旧行为返回 fail，会令 supervisor 把「tester 没交卷」误判成「代码有缺陷」而重跑整个
        # attempt——惩罚正确代码（2026-09-02 复盘的空转回路）。与上面 finish_reason=stop 分支对齐。
        return _ret("skipped", (f"tester 达 max_iter={max_iter} 上限仍未提交 finish_verify，"
                           f"跳过独立验收（MaxIter reached，不触发 loop1 重跑）。"), [])
    results = _normalize_results(stop.get("results"), points)
    # harness 只看有没有「失败用例」：failed_case>0 才令 loop1 重跑；
    # pass / skipped / 漏提交（已补 skipped）均不计入失败，不触发重跑。
    failed = [r for r in results if r["verdict"] == "fail"]
    if failed:
        detail = "；".join(
            f"#{r['id']} {r['verdict']}: {r['evidence'][:160]}" for r in failed[:5])
        return _ret("fail", f"独立验收未通过（{len(failed)}/{len(results)} 个用例失败）：{detail}", results)
    passed = [r for r in results if r["verdict"] == "pass"]
    skipped_pts = [r for r in results if r["verdict"] == "skipped"]
    return _ret("pass", (f"独立验收通过（{len(passed)} 通过 / {len(skipped_pts)} 跳过，"
                f"无失败用例）。"), results)


def verify_gate(model: Optional[str] = None, max_iter: int = _C.MAX_TESTER_ITER) -> Tuple[str, str]:
    """第 4 道闸门的对外入口。model=None 时按配置解析（FORGE_TESTER_MODEL / EXECUTOR_MODEL）。

    2026-09-03：交付有效性预判（结构化 judge）——
    先做客观检查（非空实现/测试文件 + pytest 收集>0），不达标直接 fail（省模型调用）；
    达标再上模型 judge 判「有效单测+有效实现」（yes/notsure 放行，no→fail），最后进深度 tester agent。
    """
    vmodel = model or _resolve_model(_C.TESTER_MODEL, _C.EXECUTOR_MODEL)
    logger.critical('tester_start model=%s max_iter=%s', vmodel, max_iter)
    if vmodel is None:
        return "skipped", "独立验收已关闭（FORGE_TESTER_MODEL=off），跳过。"
    # 客观前置：空文件/空测试/无收集 → 直接 fail（不浪费模型调用，且符合「不能是空文件不能是空测试」）
    summary, objective_ok = _delivery_summary()
    if not objective_ok:
        return "fail", f"交付客观检查未通过（实现/测试文件为空或 pytest 无收集）：{summary}"
    # 模型 judge：「有效单测+有效实现」主观判定（硬 prompt 铁律已内置空/占位符→no）
    jres, jreason = M.judge("tester", summary)
    if jres == "no":
        return "fail", f"独立验收交付判定未通过（judge）：{jreason} | {summary}"
    verdict, detail, _ = run_tester(vmodel, max_iter)
    return verdict, detail
