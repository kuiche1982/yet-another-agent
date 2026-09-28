#!/usr/bin/env python3
"""mermaid DSL 能力评估执行器。

数据集固在 scripts/mermaid_cases.py，本文件只负责执行与落盘，后续换模型复用。

用法：
    python scripts/mermaid_probe.py                        # 全量跑当前模型
    python scripts/mermaid_probe.py --model <id>           # 评估别的模型
    python scripts/mermaid_probe.py --only r1_dataflow,w1_flow
    python scripts/mermaid_probe.py --thinking on          # 保留思维链（对比用）

输出：logs/<model>_<YYYY-MM-DD-HH-mm>.jsonl，一题一行，含完整输入与输出。

已知坑（2026-09-04 实测）：qwen3.5-4b-mtplx-optimized-speed 默认开 thinking，
且 reasoning token 占用 max_tokens 配额，配额耗尽时 content 为空而
finish_reason=length，服务端记 content_empty_reason=truncated_inside_reasoning。
故默认关闭 thinking（extra_body enable_thinking=false）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mermaid_cases import CASES  # noqa: E402

BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
DEF_MODEL = os.environ.get("PROBE_MODEL", "qwen3.5-4b-mtplx-optimized-speed")
TIMEOUT = float(os.environ.get("LMSTUDIO_TIMEOUT", "240"))

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

READ_SYS = (
    "You are a precise Mermaid diagram analyst. Answer only what is asked, "
    "concisely, using the node IDs from the source. No preamble."
)
WRITE_SYS = (
    "You are a Mermaid expert. Output ONLY the mermaid code block, no explanation, "
    "no prose before or after. Start the first line with the diagram type keyword."
)


# ---------------------------------------------------------------- 工具


def norm(s: str) -> str:
    # 去掉 markdown 反引号（模型常把节点 id 包成 `A`），避免挡住正则
    return re.sub(r"\s+", " ", (s or "").replace("`", "")).lower()


def strip_fence(text: str) -> str:
    """从模型输出抽取 mermaid 代码。支持三种形态：
    1) 标准 ```mermaid / ``` 围栏
    2) lfm2.5 系列的 tool_call 包装：mermaidExtract(mermaidCode='...')]
    3) 无包装的裸文本（兜底）
    """
    # 1) 标准围栏
    m = re.search(r"```(?:mermaid)?\s*(.*?)```", text or "", re.S)
    if m:
        return m.group(1).strip()
    # 2) lfm2.5 的 tool_call 包装：mermaidCode='...' 或 mermaidCode="..."
    m = re.search(r"mermaidCode\s*=\s*'(.*?)'\)\]", text or "", re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r'mermaidCode\s*=\s*"(.*?)"\)\]', text or "", re.S)
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def extract_tool_mermaid(text: str) -> str:
    """仅抽取 lfm2.5 的 tool_call 包装内容（mermaidCode=...），无则返回空。"""
    m = re.search(r"mermaidCode\s*=\s*'(.*?)'\)\]", text or "", re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r'mermaidCode\s*=\s*"(.*?)"\)\]', text or "", re.S)
    if m:
        return m.group(1).strip()
    return ""


KNOWN_HEADS = (
    "flowchart", "graph", "sequencediagram", "classdiagram", "statediagram",
    "erdiagram", "gantt", "pie", "gitgraph", "mindmap", "timeline",
    "journey", "quadrantchart", "c4context", "block-beta", "architecture-beta",
    "xychart-beta", "sankey-beta", "treemap-beta",
)


def lint_mermaid(code: str) -> list[str]:
    """轻量语法体检：不依赖渲染器，只抓常见硬错误。"""
    errs: list[str] = []
    lines = [l.rstrip() for l in (code or "").splitlines() if l.strip()]
    if not lines:
        return ["空代码"]
    head = lines[0].strip().lower()
    if not any(head.startswith(k) for k in KNOWN_HEADS):
        errs.append(f"首行不是图类型声明: {lines[0][:40]!r}")
    # subgraph/end 配对只在 flowchart/graph 系成立；sequenceDiagram 的 alt...end、
    # classDiagram 等另有语义，不能套用这条规则
    is_flow = head.startswith(("flowchart", "graph"))
    if is_flow:
        sub_open = sum(1 for l in lines if re.match(r"\s*subgraph\b", l, re.I))
        sub_close = sum(1 for l in lines if re.match(r"\s*end\s*$", l, re.I))
        if sub_open != sub_close:
            errs.append(f"subgraph({sub_open}) 与 end({sub_close}) 不匹配")
    for i, l in enumerate(lines, 1):
        s = l.strip()
        if re.match(r"^\s*end\s*$", s, re.I):
            continue
        if is_flow and re.search(r"(^|[>|\s])end(\s*($|[-|]))", s, re.I) and "--" in s:
            errs.append(f"L{i}: end 被用作节点 id（保留字冲突）")
        if s.count("[") != s.count("]"):
            errs.append(f"L{i}: 方括号不配对 -> {s[:40]!r}")
        if s.count("{") != s.count("}"):
            errs.append(f"L{i}: 花括号不配对 -> {s[:40]!r}")
    # sequenceDiagram 专属硬错：participant 用冒号而非 as；自造中文分支关键字
    is_seq = head.startswith("sequencediagram")
    if is_seq:
        for i, l in enumerate(lines, 1):
            s = l.strip()
            if re.match(r"participant\s+\S+\s*:\s", s) and not re.match(
                    r"participant\s+\S+\s+as\s+\S+", s, re.I):
                errs.append(f"L{i}: participant 用了冒号而非 as -> {s[:40]!r}")
            if re.match(r"(如果|若|否则|不然|当.+时)", s):
                errs.append(f"L{i}: 自造中文分支（应使用 alt/else/end）-> {s[:40]!r}")
    return errs


# 否定前缀：出现在关键词紧邻左侧时，整词语义反转（不/没/无/否/未）
NEG = set("不没无否未")


def _has_nonneg_match(a: str, pattern: str) -> bool:
    """在归一化文本里找 pattern，但若匹配段紧邻否定前缀则跳过（避免
    “没有出边”被“有出边”误命中、“不能正确渲染”被“正确渲染”误命中）。"""
    for m in re.finditer(pattern, a, re.I):
        if m.start() == 0 or a[m.start() - 1] not in NEG:
            return True
    return False


def check_hit(text: str, c: dict) -> bool:
    a = norm(text)
    # 否定表述优先：命中 not 里任一项（且未被否定前缀反转）即判错。
    if c.get("not"):
        for k in c["not"]:
            kn = norm(k)
            for m in re.finditer(re.escape(kn), a):
                if m.start() == 0 or a[m.start() - 1] not in NEG:
                    return False
    if "re" in c:
        try:
            return _has_nonneg_match(a, c["re"])
        except re.error:
            return False
    if "all" in c:
        return all(norm(k) in a for k in c["all"])
    return any(norm(k) in a for k in c.get("any", []))


def build_messages(case: dict) -> tuple[str, str]:
    sysmsg = READ_SYS if case["kind"] == "read" else WRITE_SYS
    if case.get("src"):
        user = f"下面是 mermaid 源码：\n\n```mermaid\n{case['src']}\n```\n\n{case['ask']}"
    else:
        user = case["ask"]
    return sysmsg, user


# ---------------------------------------------------------------- 执行


def new_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT, max_retries=0)


def call(model: str, sysmsg: str, user: str, *, box: dict, temperature: float,
         max_tokens: int, thinking: bool, retries: int, sleep: float) -> dict:
    """返回 {raw, reasoning, error, seconds, usage, server, attempts}。

    连接策略（2026-09-04 实测，两条都是坑）：
    - 单纯复用 client：LM Studio 会「首次 200、之后全部 404 {'detail':'Not Found'}」，
      且一旦 404 后续全挂，curl 不复现 —— 与服务端 session/postcommit 抢占有关。
    - 每请求新建 client：能规避 404，但每次都是新 session，服务端刷
      cross_session_foreground_preempted，吞吐从 ~30 tok/s 掉到 ~12 tok/s。
    → 折中：跨用例复用同一个 client，只在遇到 404 时重建连接再重试。

    thinking 开关用 extra_body 传。注意：
    - 必须传 bool：字符串 "off" 是 truthy，会反向开启（踩过的坑）。
    - Ling 后端不认顶层 enable_thinking，必须走 chat_template_kwargs 才真关 thinking，
      否则仍吐 reasoning_content（白占 token、慢、有截断 raw 的风险）。
    """
    if thinking:
        extra = {"enable_thinking": True}
    elif "ling" in model.lower():
        # Ling 后端特例：顶层 enable_thinking 无效，必须 chat_template_kwargs
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        extra = {"enable_thinking": False}
    payload = dict(
        model=model,
        messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    last_err = None
    last_dt = 0.0
    for attempt in range(1, retries + 1):
        client = box["c"]
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**payload, extra_body=extra)
            dt = time.time() - t0
            msg = resp.choices[0].message
            raw = (msg.content or "").strip()
            reasoning = getattr(msg, "reasoning_content", None) or ""
            try:
                stats = dict(getattr(resp, "mtplx_stats", None) or {})
            except Exception:  # noqa: BLE001
                stats = {}
            usage = getattr(resp, "usage", None)
            return {
                "raw": raw,
                "reasoning": reasoning,
                "error": None,
                "seconds": round(dt, 2),
                "attempts": attempt,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                } if usage else {},
                "server": {
                    "finish_reason": resp.choices[0].finish_reason,
                    "enable_thinking": stats.get("request_enable_thinking"),
                    "content_empty_reason": stats.get("content_empty_reason"),
                    "answer_tokens": stats.get("answer_tokens"),
                    "reasoning_tokens": stats.get("reasoning_tokens"),
                    "tok_s": stats.get("tok_s"),
                    "served_model": stats.get("served_model_id"),
                },
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            last_dt = time.time() - t0
            if "404" in last_err or "NotFound" in last_err:
                # 连接/session 已失效：换新连接再试
                try:
                    box["c"].close()
                except Exception:  # noqa: BLE001
                    pass
                box["c"] = new_client()
            if attempt < retries:
                time.sleep(sleep * attempt)
    return {"raw": "", "reasoning": "", "error": last_err, "seconds": round(last_dt, 2),
            "attempts": retries, "usage": {}, "server": {}}


def run_case(case: dict, args, run_id: str, seq: int, box: dict) -> dict:
    sysmsg, user = build_messages(case)
    r = call(args.model, sysmsg, user, box=box, temperature=args.temperature,
             max_tokens=args.max_tokens, thinking=args.thinking,
             retries=args.retry, sleep=args.sleep)

    raw = r["raw"]
    is_code = case["kind"] in ("write", "edit")
    code = strip_fence(raw) if is_code else ""
    # 代码类题目用抽出后的代码判分（首行正则等），理解类用全文
    target = code if is_code else raw

    checks = [{"name": c["name"], "hit": check_hit(target, c)} for c in case["checks"]]
    passed = sum(1 for c in checks if c["hit"])

    empty = not raw.strip()
    if empty and not r["error"]:
        r["error"] = "empty_content"

    rec = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_id,
        "seq": seq,
        "model": args.model,
        "case": {"id": case["id"], "kind": case["kind"], "title": case["title"],
                 "gt": case["gt"]},
        "input": {"system": sysmsg, "user": user, "src": case.get("src")},
        "output": {
            "raw": raw,
            "code": code,
            "lint": lint_mermaid(code) if is_code else [],
            "chars": len(raw),
            "empty": empty,
            "reasoning": r["reasoning"],
        },
        "judge": {"checks": checks, "passed": passed, "total": len(checks),
                  "score": f"{passed}/{len(checks)}"},
        "perf": {"seconds": r["seconds"], **r["usage"], "attempts": r["attempts"]},
        "server": r["server"],
        "params": {"temperature": args.temperature, "max_tokens": args.max_tokens,
                   "thinking": args.thinking},
        "error": r["error"],
    }
    warn: list[str] = []
    if not args.thinking and r["server"].get("enable_thinking") is True:
        warn.append("thinking_not_disabled")
    if raw and r["reasoning"]:
        warn.append("reasoning_present_when_disabled")
    rec["warn"] = warn
    return rec


def rescore(in_path: Path, out_path: Path) -> int:
    """对已有 jsonl 用当前 CASES 的 checks / lint 重算 judge（不调模型）。

    用途：修改判分规则后，旧跑记录里的 output 仍有效，只需重算分数即可，
    不必重跑模型（temp=0 确定性，重跑只会复读同一段文本）。
    """
    cases_by_id = {c["id"]: c for c in CASES}
    recs = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    if not recs:
        print(f"empty: {in_path}", file=sys.stderr)
        return 2
    model = recs[0]["model"]
    for r in recs:
        cid = r["case"]["id"]
        case = cases_by_id.get(cid)
        if not case:
            print(f"skip unknown case {cid}", file=sys.stderr)
            continue
        raw = r["output"]["raw"]
        is_code = case["kind"] in ("write", "edit")
        code = r["output"]["code"]
        target = code if is_code else raw
        checks = [{"name": c["name"], "hit": check_hit(target, c)} for c in case["checks"]]
        passed = sum(1 for c in checks if c["hit"])
        r["judge"] = {"checks": checks, "passed": passed, "total": len(checks),
                      "score": f"{passed}/{len(checks)}"}
        r["output"]["lint"] = lint_mermaid(code) if is_code else []
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                        encoding="utf-8")
    tot = sum(r["judge"]["passed"] for r in recs)
    allc = sum(r["judge"]["total"] for r in recs)
    print(f"rescored {in_path.name} -> {out_path.name}")
    print(f"model={model}  n={len(recs)}  要点命中 {tot}/{allc} ({tot / allc * 100:.1f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--only", default="", help="逗号分隔的 case id")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=2.0, help="重试/用例间隔秒")
    ap.add_argument("--out", default="", help="自定义输出 jsonl 路径")
    ap.add_argument("--rescore", default="",
                    help="对一个已有 jsonl 用当前 checks/lint 重新判分（不调用模型），"
                         "输出 <原名>_rescored.jsonl")
    args = ap.parse_args()

    if args.rescore:
        return rescore(Path(args.rescore),
                       Path(args.rescore).with_name(
                           Path(args.rescore).stem + "_rescored.jsonl"))

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    cases = [c for c in CASES if not only or c["id"] in only]
    if not cases:
        print(f"no case matched: {args.only}", file=sys.stderr)
        return 2

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\-]+", "_", args.model)
    out = Path(args.out) if args.out else (
        LOG_DIR / f"{safe}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.jsonl")

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    # 必须转成 bool：args.thinking 原值是 "on"/"off" 字符串，"off" 是 truthy
    args.thinking = (args.thinking == "on")
    box = {"c": new_client()}

    print(f"model={args.model}  thinking={'on' if args.thinking else 'off'}  "
          f"temp={args.temperature} max_tokens={args.max_tokens}  cases={len(cases)}")
    print(f"out -> {out}\n")

    results = []
    with out.open("w", encoding="utf-8") as fh:
        for i, c in enumerate(cases, 1):
            rec = run_case(c, args, run_id, i, box)
            results.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

            flag = "ERR" if rec["error"] else rec["judge"]["score"]
            print(f"[{rec['case']['id']:12s}] {flag:5s} {rec['perf']['seconds']:6.1f}s  "
                  f"{rec['case']['title']}")
            for k in rec["judge"]["checks"]:
                print(f"      {'OK  ' if k['hit'] else 'MISS'} {k['name']}")
            for e in rec["output"]["lint"]:
                print(f"      LINT {e}")
            if rec["warn"]:
                print(f"      WARN {','.join(rec['warn'])}")
            if rec["error"]:
                print(f"      ERR  {rec['error']}")
            print()
            time.sleep(args.sleep)

    try:
        box["c"].close()
    except Exception:  # noqa: BLE001
        pass

    tot = sum(r["judge"]["passed"] for r in results)
    allc = sum(r["judge"]["total"] for r in results)
    nerr = sum(1 for r in results if r["error"])
    dur = sum(r["perf"]["seconds"] for r in results)
    print("=" * 60)
    print(f"模型          : {args.model}")
    print(f"用例          : {len(results)} 题，错误 {nerr} 题")
    print(f"要点命中      : {tot}/{allc}  ({tot / allc * 100:.1f}%)" if allc else "n/a")
    print(f"总耗时        : {dur:.1f}s，均值 {dur / max(len(results), 1):.1f}s/题")
    print(f"JSONL         : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
