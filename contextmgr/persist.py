"""contextmgr —— 持久化层（落盘为 markdown，mermaid 作 L2 介质）。

设计要点（呼应最初「mermaid DSL 作可视化介质」前提）：
- 每个 L0 片段 -> 一个 .md 文件，frontmatter 记录元数据 + 增量追踪字段：
    updated_level : int  最高算到的级别 (0 L0 / 1 L1 / 2 L2)，L3 不落盘
    src_mtime     : float 源文件修改时间，rebuild 用它判断要不要重派生
- 正文按 L0/L1/L2 三级分开写，**L3 不保存**（L3 是纯派生视图：label 来自 L1、指针来自 L0，
  加载时由 _regen_l3 从 L1 重建，无需落盘、不把低级别塞进 L3）：
    ## L0 (Truth)      原始文本（唯一真相）
    ## L1 Structured   summary + key_points + entities（文本蒸馏）
    ## L2 Visual        ```mermaid ... ``` 详图（mermaid 是 L2 介质）
- 语料级产物（在 cache 根）：
    L2_overview.md  聚合 mermaid mindmap 脑图（所有片段 label，导航用）—— 属 L2
    L3_index.md     纯文本 label + 指针列表（检索入口，向下 descend 到 L0；由 L1 重建）
- rebuild(root, cache_dir)：按源文件 mtime 增量
    * mtime 未变且 updated_level>=2 -> 直接从磁盘 load（重建 L3），不重蒸馏、不调 LLM
    * mtime 变了 -> 从 L0 重派生（L0 文本变，L1/L2 必然过期）
    * 源文件已删 -> 清理其落盘片段，避免陈旧条目

依赖：仅标准库（不引 pyyaml / markdown 解析库）。
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from typing import Any

from .store import FragmentStore
from .types import Fragment, L1Structured, L2Visual, L3Index, Source

_FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
_SECTION_RE = re.compile(r"^## (L0 \(Truth\)|L1 Structured|L2 Visual)\s*$", re.M)

# —— 索引器版本（派生语义版本）——
# 何时 +1：片段的**派生语义**变了，即「同样的源文件文本会派生出不同的片段」时：
#   * frontmatter 增删字段（例：Library 片段补 lineno/end_lineno）
#   * 切片规则变化（例：切块大小 / 符号切分策略）
#   * 蒸馏输出形状变化（L1/L2 结构）
# 为什么需要：增量判定的依据是「源文件内容指纹（mtime/size/sha）」，它**看不见派生逻辑变了**。
# 于是「给 ingest 新增字段」后，未改动的源文件会被判 fresh 而永远沿用旧片段 ——
# 新字段只对「之后被改动的文件」生效，存量缓存永不更新（实测：proj/r0 的 102 个片段 0 个带 lineno）。
# 机制：manifest 记这个号；与当前值不符 → 忽略 pickle 快速路径、不复用任何旧片段、强制全量重派生一次。
SCHEMA_VERSION = 1


def _l0_fingerprint(text: str) -> tuple[str, int]:
    """L0 指纹：sha256 hex + size，用于 rebuild 增量判定（兜底 mtime 回拨）。

    不写绝对路径，只算片段文本本身。三源共用。
    """
    return (hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(), len(text))


def _enc(v: Any) -> str:
    """frontmatter 标量/复杂值序列化：复杂值走 JSON，标量尽量裸写。"""
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return '""'
    return str(v)


def _dec(v: str) -> Any:
    v = v.strip()
    if v == "" or v == '""':
        return ""
    if (v.startswith("[") or v.startswith("{") or v.startswith('"')) and v.endswith(("]", "}", '"')):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    if v == "true":
        return True
    if v == "false":
        return False
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    return v


def _fid_to_path(cache_dir: str, fid: str) -> str:
    """fid 含 ':' '/' '#'，落盘文件名需 sanitize。"""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", fid)
    return os.path.join(cache_dir, f"{safe}.md")


def _updated_level(l1: L1Structured | None, l2: L2Visual | None) -> int:
    if l2 is not None:
        return 2
    if l1 is not None:
        return 1
    return 0


def _regen_l3(store: FragmentStore) -> None:
    """从 L1 重建 L3（label 来自 summary/key_points，指针来自 L0）。L3 不落盘，加载时调用。"""
    store._l3.clear()
    for frag in store.all_fragments():
        l1 = store._l1.get(frag.fid)
        if l1 is not None and l1.merged_with_l0 and not l1.summary:
            # 代码源：L1 合并 L0，用首行/实体当 label
            label = (l1.entities[0] if l1.entities else frag.text.splitlines()[0][:40]
                     if frag.text else frag.fid)
        elif l1 is not None and l1.summary:
            label = l1.summary
        elif l1 is not None and l1.key_points:
            label = l1.key_points[0]
        else:
            label = frag.text[:40].replace("\n", " ")
        label = re.sub(r"\s+", " ", label).strip()[:60] or frag.fid
        store._l3[frag.fid] = L3Index(
            fid=frag.fid,
            label=label,
            l0_pointer=frag.l0_pointer(),
            source=frag.source,
            priority=frag.priority,
            tokens=frag.tokens,
        )


def fragment_to_md(frag: Fragment, l1: L1Structured | None, l2: L2Visual | None,
                   src_mtime: float, distiller: str,
                   full_text: str | None = None,
                   src_sha256: str = "", src_size: int = 0) -> str:
    """把一个片段的 L0–L2 序列化成 markdown 文本（L3 不写）。

    full_text：源文件全文（Library/Code 适用）。提供时算文件级 sha256+size 写进
    frontmatter（src_sha256/src_size），供 rebuild 增量判定兜底 mtime 回拨。
    src_sha256/src_size：由 rebuild 直接传入（基于文件字节算）—— 保证 rebuild 比对
    的 sha 与写入的 sha 严格一致（避开 universal newlines / encode 差异）。
    Session 片段无源文件，传 None 即可，src_sha256/src_size 默认空。
    """
    sha, size = _l0_fingerprint(frag.text)
    fm: dict[str, Any] = {
        "fid": frag.fid,
        "source": frag.source.value,
        "origin": frag.origin,
        "offset": frag.offset,
        "length": frag.length,
        "priority": frag.priority,
        "tokens": frag.tokens,
        "updated_level": _updated_level(l1, l2),
        "src_mtime": src_mtime,
        "distiller": distiller,
    }
    if full_text is not None or src_sha256:
        # 文件级指纹：mtime 回拨（git checkout / rsync -t）也能检测
        fm["src_sha256"] = src_sha256 or hashlib.sha256(
            (full_text or "").encode("utf-8", errors="ignore")
        ).hexdigest()
        fm["src_size"] = src_size or (len(full_text) if full_text is not None else size)
    else:
        fm["l0_sha256"] = sha   # snippet 级（保留供 forensic，会话/无源文件时使用）
        fm["src_size"] = size
    # Code: 携带 lineno/end_lineno 供 L1 标注文件:行号位置
    if frag.lineno:
        fm["lineno"] = frag.lineno
        fm["end_lineno"] = frag.end_lineno
    if l1 is not None:
        fm["key_points"] = l1.key_points
        fm["entities"] = l1.entities
        fm["merged_with_l0"] = l1.merged_with_l0
        if l1.calls:           # Code L1.calls：供 write_l2_callgraphs 聚合模块 def→def 边
            fm["calls"] = l1.calls

    lines = ["---"]
    for k, v in fm.items():
        lines.append(f"{k}: {_enc(v)}")
    lines.append("---")
    lines.append("")

    # L0
    lines.append("## L0 (Truth)")
    lines.append(frag.text.rstrip())
    lines.append("")

    # L1
    lines.append("## L1 Structured")
    if l1 is not None:
        if l1.summary:
            lines.append(f"summary: {l1.summary}")
        if l1.key_points:
            lines.append("key_points:")
            for kp in l1.key_points:
                lines.append(f"  - {kp}")
        if l1.entities:
            lines.append("entities: " + ", ".join(l1.entities))
    lines.append("")

    # L2 (mermaid 介质)
    lines.append("## L2 Visual")
    if l2 is not None and l2.diagram.strip():
        lines.append("```mermaid")
        lines.append(l2.diagram.strip())
        lines.append("```")
    lines.append("")

    return "\n".join(lines)


def md_to_fragment(text: str) -> tuple[Fragment, L1Structured | None,
                                        L2Visual | None, dict]:
    """把 markdown 文本解析回 (frag, l1, l2, meta)。L3 不解析（加载时重建）。"""
    m = _FRONT_RE.match(text)
    meta: dict[str, Any] = {}
    body = text
    if m:
        for line in m.group(1).splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            meta[k.strip()] = _dec(v)
        body = text[m.end():]

    parts: dict[str, str] = {}
    positions = [(mm.start(), mm.group(1)) for mm in _SECTION_RE.finditer(body)]
    for i, (pos, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        parts[name] = body[pos:end]

    def _sec(name: str) -> str:
        blk = parts.get(name, "")
        blk = re.sub(r"^## .*?\n", "", blk, count=1)
        return blk.strip()

    l0_text = _sec("L0 (Truth)")

    l1 = None
    l1_blk = _sec("L1 Structured")
    # L1.calls：先从 frontmatter meta 取（fragment_to_md 写在 fm["calls"]），缺省回退 body 解析
    calls: list[str] = []
    raw_calls = meta.get("calls")
    if isinstance(raw_calls, list):
        calls = [str(c) for c in raw_calls]
    if l1_blk:
        summary = ""
        sm = re.search(r"^summary:\s*(.*)$", l1_blk, re.M)
        if sm:
            summary = sm.group(1).strip()
        kps = re.findall(r"^\s*-\s+(.*)$", l1_blk, re.M)
        ents = []
        em = re.search(r"^entities:\s*(.*)$", l1_blk, re.M)
        if em:
            ents = [e.strip() for e in em.group(1).split(",") if e.strip()]
        # body 形式兼容（早期缓存可能写在 body 而非 frontmatter）
        if not calls:
            cm = re.search(r"^calls:\s*(\[.*\]|\S.*)$", l1_blk, re.M)
            if cm:
                try:
                    calls = json.loads(cm.group(1))
                except json.JSONDecodeError:
                    calls = [s.strip() for s in cm.group(1).strip("[]").split(",") if s.strip()]
        l1 = L1Structured(
            fid=meta["fid"],
            summary=summary,
            key_points=[k.strip() for k in kps],
            entities=ents,
            calls=calls,
            merged_with_l0=bool(meta.get("merged_with_l0", False)),
        )

    l2 = None
    l2_blk = _sec("L2 Visual")
    dia = ""
    dm = re.search(r"```mermaid\s*\n(.*?)\n```", l2_blk, re.S)
    if dm:
        dia = dm.group(1).strip()
    if dia or l2_blk:
        l2 = L2Visual(fid=meta["fid"], diagram=dia, tokens=int(meta.get("tokens", 0)))

    frag = Fragment(
        fid=meta["fid"],
        source=Source(meta.get("source", "library")),
        text=l0_text,
        priority=int(meta.get("priority", 0)),
        tokens=int(meta.get("tokens", 0)),
        origin=str(meta.get("origin", "")),
        offset=int(meta.get("offset", 0)),
        length=int(meta.get("length", 0)),
        lineno=int(meta.get("lineno", 0)),
        end_lineno=int(meta.get("end_lineno", 0)),
    )
    return frag, l1, l2, meta


def save_fragment(cache_dir: str, frag: Fragment, l1: L1Structured | None,
                  l2: L2Visual | None, src_mtime: float, distiller: str,
                  full_text: str | None = None,
                  src_sha256: str = "", src_size: int = 0) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    md = fragment_to_md(frag, l1, l2, src_mtime, distiller,
                        full_text=full_text, src_sha256=src_sha256, src_size=src_size)
    with open(_fid_to_path(cache_dir, frag.fid), "w", encoding="utf-8") as fh:
        fh.write(md)


def load_store(cache_dir: str) -> FragmentStore:
    """从 cache_dir 读回全部片段（磁盘 load，不调蒸馏）；L3 由 L1 重建。"""
    store = FragmentStore()
    if not os.path.isdir(cache_dir):
        return store
    for name in os.listdir(cache_dir):
        if not name.endswith(".md") or name in ("L2_overview.md", "L2_callgraphs.md", "L3_index.md"):
            continue
        with open(os.path.join(cache_dir, name), "r", encoding="utf-8") as fh:
            frag, l1, l2, _ = md_to_fragment(fh.read())
        store.add_fragment(frag)
        if l1 is not None:
            store._l1[l1.fid] = l1
        if l2 is not None:
            store._l2[l2.fid] = l2
    _regen_l3(store)
    return store


def _scan_cache(cache_dir: str) -> dict[str, list[dict]]:
    """扫描 cache 现有片段 frontmatter，按 origin 分组（用于增量比对）。"""
    out: dict[str, list[dict]] = {}
    if not os.path.isdir(cache_dir):
        return out
    for name in os.listdir(cache_dir):
        if not name.endswith(".md") or name in ("L2_overview.md", "L2_callgraphs.md", "L3_index.md"):
            continue
        with open(os.path.join(cache_dir, name), "r", encoding="utf-8") as fh:
            txt = fh.read()
        m = _FRONT_RE.match(txt)
        if not m:
            continue
        meta: dict[str, Any] = {}
        for line in m.group(1).splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            meta[k.strip()] = _dec(v)
        origin = meta.get("origin", "")
        out.setdefault(origin, []).append({
            "fid": meta.get("fid", ""),
            "mtime": float(meta.get("src_mtime", 0.0)),
            "size": int(meta.get("src_size", -1)),
            # 文件级指纹优先（Library/Code）；无则用 l0_sha256（snippet 级，Session 兼容）
            "sha": str(meta.get("src_sha256") or meta.get("l0_sha256", "")),
            "updated_level": int(meta.get("updated_level", 0)),
            "source": meta.get("source", ""),
        })
    return out


def _scan_sources(root: str, lib_ext: set[str], code_ext: set[str],
                  ignore_dirs: set[str] | None) -> dict:
    """廉价 stat 级扫描：返回 {relpath: (mtime, size)}，与 rebuild 主循环同过滤。

    仅做 os.stat（不读文件内容），用于快速判定缓存是否仍新鲜；命中则整库走 pickle 读回。
    """
    import glob
    out: dict[str, tuple[float, int]] = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if not os.path.isfile(path):
            continue
        if ignore_dirs and any(part in ignore_dirs for part in path.split("/")):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in lib_ext and ext not in code_ext:
            continue
        try:
            st = os.stat(path)
            rel = os.path.relpath(path, root)
            out[rel] = (st.st_mtime, st.st_size)
        except OSError:
            continue
    return out


def _load_manifest(manifest_path: str) -> tuple[int, dict | None]:
    """读 manifest → (schema_version, {relpath: [mtime, size, sha]})。

    不存在 / 损坏 / 结构不对 → (0, None)（version=0 必然 != SCHEMA_VERSION，
    调用方据此走「强制重派生」路径，与「无缓存」同语义）。

    sha 是「内容指纹」，让「mtime 被 touch / 回拨但内容没变」不触发重派生
    （与 `_cache_fresh` 的哲学一致：内容才是真相，mtime 只当廉价前置筛子）。
    兼容旧版 2 元组 [mtime, size]（sha 补空串 → 首轮重派生一次，之后自愈）；
    旧版 manifest 无 version 字段 → 读作 0 → 触发一次全量重派生后自愈。
    """
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            man = json.load(fh)
    except Exception:
        return 0, None
    if not isinstance(man, dict) or not isinstance(man.get("items"), dict):
        return 0, None
    try:
        ver = int(man.get("version", 0))
    except (TypeError, ValueError):
        ver = 0
    out: dict[str, list] = {}
    for rel, sig in man["items"].items():
        sig = list(sig) if isinstance(sig, (list, tuple)) else []
        out[rel] = [sig[0] if len(sig) > 0 else 0.0,
                    sig[1] if len(sig) > 1 else -1,
                    sig[2] if len(sig) > 2 else ""]
    return ver, out


def _manifest_matches(man: dict, current: dict) -> bool:
    """清单是否全等（按 relpath 的 (mtime,size) 逐条相等）——只做廉价 stat 级比对。"""
    if set(man.keys()) != set(current.keys()):
        return False
    for rel, sig in current.items():
        old = man.get(rel) or []
        if len(old) < 2 or old[0] != sig[0] or old[1] != sig[1]:
            return False
    return True


def _save_manifest(manifest_path: str, current: dict, shas: dict | None = None,
                   version: int | None = None) -> None:
    """落盘 manifest（含内容指纹 + 索引器版本）。

    shas 未覆盖的条目沿用旧值，避免丢失已知 sha。
    version 缺省 = 当前 SCHEMA_VERSION —— 写完即声明「本缓存由当前派生逻辑生成」。
    """
    items = {}
    for rel, sig in current.items():
        items[rel] = [sig[0], sig[1], (shas or {}).get(rel, "")]
    try:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump({"version": SCHEMA_VERSION if version is None else int(version),
                       "items": items}, fh)
    except Exception:
        pass


def _sha_file(path: str) -> str:
    """文件内容 sha256；读失败返回空串（= 不可信指纹，调用方按「已变」处理）。"""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def _is_session_frag(frag) -> bool:
    """Session 源片段由 save/load_session 管理，不参与 rebuild 的 prune。"""
    src = getattr(frag, "source", None)
    return getattr(src, "value", str(src)).endswith("session")


def _adopt(cm, loaded) -> None:
    """把整库读回的三张表并入 cm.store（省去逐碎片 .md 读回）。"""
    cm.store._frags.update(loaded._frags)
    cm.store._l1.update(loaded._l1)
    cm.store._l2.update(loaded._l2)


def _write_indexes(cache_dir: str, store) -> None:
    """重算 L3（L3 不落盘，每次启动从 L1 重建）并落盘 L2/L3 聚合视图。"""
    _regen_l3(store)
    write_l2_overview(cache_dir, store)
    write_l2_callgraphs(cache_dir, store)
    write_l3_index(cache_dir, store)


def _save_consolidated(store_pkl: str, store) -> None:
    """把整库（_frags/_l1/_l2 三张表）一次性 pickle 落盘，供下次快速整库读回。

    fail-open：任何异常静默跳过（退化为逐碎片 .md 路径，不影响正确性）。
    """
    try:
        with open(store_pkl, "wb") as fh:
            pickle.dump((store._frags, store._l1, store._l2), fh)
    except Exception:
        pass


def _load_consolidated(store_pkl: str):
    """整库一次读回；失败返回 None（调用方退化为逐文件逻辑）。"""
    try:
        with open(store_pkl, "rb") as fh:
            frags, l1, l2 = pickle.load(fh)
    except Exception:
        return None
    store = FragmentStore()
    store._frags = frags
    store._l1 = l1
    store._l2 = l2
    _regen_l3(store)
    return store


def _safe_node(s: str) -> str:
    """mermaid mindmap 节点文本清洗：去掉会破坏语法的字符。"""
    s = s.replace("\n", " ").strip()
    s = re.sub(r'[()\[\]{}:"/\\]', "", s)
    if len(s) > 38:
        s = s[:38]
    if not s:
        s = "item"
    if s.lower() == "end":
        s = "end_x"
    return s


def write_l2_overview(cache_dir: str, store: FragmentStore) -> None:
    """语料级 L2 脑图：所有片段 label 聚合成一张 mermaid mindmap（导航用，L2 介质）。"""
    frags = {f.fid: f for f in store.all_fragments()}
    by_origin: dict[str, list[str]] = {}
    for l3 in store.all_l3():
        origin = frags.get(l3.l0_pointer[0])
        key = origin.origin if origin else l3.source.value
        by_origin.setdefault(key, []).append(_safe_node(l3.label))

    lines = ["# L2 Overview — 语料级脑图（mermaid mindmap，L2 介质）", ""]
    lines.append("```mermaid")
    lines.append("mindmap")
    lines.append("  root((KnowledgeBase))")
    for origin, labels in sorted(by_origin.items()):
        lines.append(f"    {_safe_node(origin)}")
        for lab in labels:
            lines.append(f"      {lab}")
    lines.append("```")
    lines.append("")
    with open(os.path.join(cache_dir, "L2_overview.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_l2_callgraphs(cache_dir: str, store: FragmentStore) -> None:
    """Code 源模块级调用图（def→def 边）：每个 .py 模块一张 mermaid graph。

    数据来自 L1.calls（CodeDistiller 抽取的本 def 内 ast.Name 调用）。
    跨片段 def→def 边：caller.L1.calls ∩ module_defined → caller→callee。
    只画同模块内调用；跨模块（import foo.bar）作为孤立 callee 节点标灰、不画 def 边，
    避免视图噪声与跨文件不一致。
    """
    # 收集所有 code fragments
    code_frags = [f for f in store.all_fragments() if f.source == Source.CODE]
    by_module: dict[str, list[tuple[Fragment, L1Structured | None]]] = {}
    for f in code_frags:
        l1 = store._l1.get(f.fid)
        by_module.setdefault(f.origin or f.fid, []).append((f, l1))

    sections: list[str] = []
    for module, items in sorted(by_module.items()):
        # 模块内全部 def 名（来自 L1.entities，1 entity per fragment after split_code_fragments）
        def_names: list[str] = []
        for _, l1 in items:
            if l1 and l1.entities:
                def_names.extend(l1.entities)
        defined = set(def_names)

        nodes = [f'  D{i}["{_safe_node(n)}"]' for i, n in enumerate(def_names[:10])]
        edges: list[str] = []
        callees_outside: set[str] = set()
        # caller idx map
        idx_by_name: dict[str, int] = {n: i for i, n in enumerate(def_names[:10])}
        for _, l1 in items:
            if not l1 or not l1.entities:
                continue
            caller = l1.entities[0]
            ci = idx_by_name.get(caller)
            if ci is None:
                continue
            for callee in (l1.calls or []):
                if callee in defined and callee != caller and callee in idx_by_name:
                    edges.append(f'  D{ci} -->|"calls"| D{idx_by_name[callee]}')
                else:
                    callees_outside.add(callee)
        # 外部 callee 节点（灰底）
        ext_nodes = [f'  X{k}["{_safe_node(c)}"]:::ext' for k, c in enumerate(sorted(callees_outside)[:8])]
        body = "\n".join(nodes + edges + ext_nodes)
        if not body:
            continue
        section = (f"## {module}\n\n"
                   f"```mermaid\ngraph TD\n{body}\n"
                   f"  classDef ext fill:#eee,stroke:#aaa,color:#555;\n```\n")
        sections.append(section)

    if not sections:
        return
    lines = ["# L2 Callgraphs — Code 模块级 def→def 调用图（L2 介质）", "",
             "数据源：CodeDistiller.L1.calls（AST ast.Name 调用，model-free）。",
             "同模块边 = `caller -->|calls| callee`；外部调用 = 灰底孤立节点。",
             ""]
    lines.extend(sections)
    lines.append("")
    with open(os.path.join(cache_dir, "L2_callgraphs.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_l3_index(cache_dir: str, store: FragmentStore) -> None:
    """语料级 L3 索引：纯文本 label + 回 L0 指针（检索入口，向下 descend）。

    注意：此文件由 L1 重建生成（L3 不落盘），mermaid 是 L2 介质、不在此处。
    """
    lines = ["# L3 Index — 一行索引 + 回 L0 指针（检索入口，由 L1 重建）", ""]
    lines.append("格式：`- [category] label  =>  fid(offset,length)`")
    lines.append("")
    for l3 in sorted(store.all_l3(), key=lambda x: (x.source.value, x.label)):
        fid, off, length = l3.l0_pointer
        lines.append(f"- [{l3.category}] {l3.label}  =>  {fid}({off},{length})")
    lines.append("")
    with open(os.path.join(cache_dir, "L3_index.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _cache_fresh(cached: list[dict], *, mtime: float, size: int, sha: str) -> bool:
    """增量判定：缓存片段是否仍与当前源文件一致（全部片段都吻合才算 fresh）。

    双路判定，缺一不可：
    - **快路径**：src_mtime 与当前 mtime 完全相等 → 直接判 fresh，不比 sha。
      常规场景（文件没被动过）走这条，免掉一次 hash。
    - **慢路径（兜底）**：sha256 + size 都吻合 → 判 fresh。
      覆盖 mtime 被回拨（git checkout 旧版 / rsync -t）与 mtime 被推到未来
      （touch、编辑器「保存但无改动」）两类场景。

    mtime 不再做 `cached.mtime >= file_mtime` 的单向比较 —— 那会让「touch 一下」
    就白白重派生整份语料。内容指纹才是真相，mtime 只当廉价前置筛子。

    sha 为空（旧 cache 无 src_sha256 字段 / 读文件失败）→ 一律判 stale，走重派生。
    """
    if not cached:
        return False
    for c in cached:
        if c.get("updated_level", 0) < 2:
            return False
        if c.get("mtime") == mtime:
            continue                      # 快路径：mtime 逐位相等
        if not sha or c.get("sha", "") != sha or c.get("size", -1) != size:
            return False                  # 慢路径：内容指纹不符 -> stale
    return True


def _load_cached(cm, cache_dir: str, cached: list[dict], stat: dict) -> None:
    """把缓存里的一批片段原样 load 回 store（不蒸馏）。"""
    for c in cached:
        fpath = _fid_to_path(cache_dir, c["fid"])
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as fh:
            frag, l1, l2, _ = md_to_fragment(fh.read())
        cm.store.add_fragment(frag)
        if l1 is not None:
            cm.store._l1[l1.fid] = l1
        if l2 is not None:
            cm.store._l2[l2.fid] = l2
        stat["loaded"] += 1


def _incremental_from_store(cm, loaded, current: dict, man: dict, root: str, base: str,
                            lib_ext: set[str], code_ext: set[str], cache_dir: str,
                            store_pkl: str, manifest_path: str,
                            distiller_name: str, stat: dict) -> dict:
    """以既有 pickle 为基底做增量：只重算「内容确实变了 / 新增」的文件，清掉已消失的来源。

    这是「agent 自己改一个 .py → 下次检索全库回落重建」的根治路径。旧实现只要 manifest
    失配就退回 `_scan_cache`（读回上千个 .md 碎片再解析）再做逐文件 diff，代价 50~140s；
    这里直接在内存 store 上比对 manifest，单文件改动的代价降到「重蒸馏这一个文件」。

    变更判定与 `_cache_fresh` 同源：mtime+size 相同 → 未变；否则比内容 sha，
    sha 相同（touch / mtime 回拨）只刷新 mtime，不重派生。内容指纹才是真相。
    """
    _adopt(cm, loaded)
    stat["files"] = len(current)

    changed: list[tuple] = []           # (rel, ext, origin_key, mtime, size, sha)
    present: set[str] = set()
    shas: dict[str, str] = {}
    for rel, sig in current.items():
        ext = os.path.splitext(rel)[1].lower()
        stem = os.path.splitext(os.path.basename(rel))[0]
        path = os.path.join(root, rel)
        origin_key = f"kb:{stem}" if ext in lib_ext else os.path.relpath(path, base)
        present.add(origin_key)
        old = man.get(rel)
        if old is not None and old[0] == sig[0] and old[1] == sig[1]:
            shas[rel] = old[2]          # 快路径：mtime/size 全等 → 沿用已知指纹
            continue
        sha = _sha_file(path)
        if old is not None and sha and old[2] == sha:
            shas[rel] = sha             # 内容没变（仅 mtime 被动过）→ 不重派生
            continue
        shas[rel] = sha
        changed.append((rel, ext, origin_key, sig[0], sig[1], sha))

    # 1) 清旧片段：源已消失 → prune 计数；源已变 → 旧片段整体作废（随后按新内容重建）
    # 「已消失」以当前 store 里的全部 origin 为准（含 pickle 基底与同进程既有片段），
    # 与旧版按 .md 缓存 prune 的语义一致。
    gone_origins = {(getattr(f, "origin", "") or "") for f in cm.store.all_fragments()
                    if not _is_session_frag(f)} - present
    drop = ((gone_origins | {c[2] for c in changed}) - {""})
    pruned = dropped = 0
    for fid, frag in list(cm.store._frags.items()):
        if _is_session_frag(frag):
            continue
        origin = getattr(frag, "origin", "") or ""
        if origin not in drop:
            continue
        cm.store._frags.pop(fid, None)
        cm.store._l1.pop(fid, None)
        cm.store._l2.pop(fid, None)
        dropped += 1
        fpath = _fid_to_path(cache_dir, fid)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
                if origin in gone_origins:
                    pruned += 1
            except OSError:
                pass
    stat["pruned"] = pruned
    stat["loaded"] = max(0, len(loaded._frags) - dropped)

    # 2) 重算变化 / 新增文件
    for rel, ext, origin_key, mtime, size, sha in changed:
        path = os.path.join(root, rel)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError as e:  # noqa: BLE001
            print(f"  [skip] 读不了 {path}: {e}")
            continue
        if ext in lib_ext:
            n = cm.ingest_library(text, doc_id=origin_key)
            stat["library"] += n
        else:
            n = cm.ingest_code(text, name=origin_key)
            stat["code"] += n
        stat["ingested"] += n
        # 按 origin 重存该文件产生的全部片段（含更新，避免磁盘陈旧）
        for fid, frag in list(cm.store._frags.items()):
            if getattr(frag, "origin", "") != origin_key:
                continue
            save_fragment(cache_dir, frag, cm.store._l1.get(fid), cm.store._l2.get(fid),
                          mtime, distiller_name, src_sha256=sha, src_size=size)

    _write_indexes(cache_dir, cm.store)
    _save_consolidated(store_pkl, cm.store)
    _save_manifest(manifest_path, current, shas)
    return stat


def rebuild(cm, root: str, cache_dir: str, distiller_name: str,
            lib_ext: set[str], code_ext: set[str], base: str | None = None,
            ignore_dirs: set[str] | None = None) -> dict:
    """增量重建：扫描 root，按源文件 mtime 决定磁盘 load 还是重派生。

    返回统计 {files, ingested, loaded, pruned, library, code, schema_rebuild}。

    索引器版本闸门：manifest 记 SCHEMA_VERSION；与当前值不符（含旧版无 version 字段 → 读作 0）
    → 忽略 pickle 快速路径、不复用任何旧片段、全量重派生一次，并清掉 fid 已消失的残留 .md。
    这是「给 ingest 新增字段（如 Library 的 lineno）」能对存量未变更文件生效的唯一通路。
    cm 需提供：ingest_library(text, doc_id), ingest_code(text, name), store。
    L3 不落盘；磁盘 load / 落盘片段统一由 _regen_l3 从 L1 重建（每次启动重建）。
    base = 仓库根，用于计算 code 源 origin key，须与 ingest_code(name=rel) 的 rel 一致。
    ignore_dirs = 需跳过的目录名集合（如 {'.venv','node_modules','__pycache__'}）；
        传入后既不扫描这些目录、也会把已缓存的对应片段 prune 掉（保持缓存干净）。
        默认 None = 不过滤（向后兼容既有调用方）。
    """
    import glob

    root = os.path.abspath(root)
    base = base or os.path.dirname(root)
    os.makedirs(cache_dir, exist_ok=True)
    present_origins: set[str] = set()
    stat = {"files": 0, "ingested": 0, "loaded": 0, "pruned": 0,
            "library": 0, "code": 0, "schema_rebuild": 0}

    # —— 快路径 A：清单全等 → 整库 pickle 一次读回 ——
    # 命中「源文件清单(mtime,size)」全等且 consolidated 存在时整库一次读入，
    # 跳过逐碎片 .md 的「扫描+解析+再读回」（碎片数上千时这是主要性能悬崖）。
    current = _scan_sources(root, lib_ext, code_ext, ignore_dirs)
    store_pkl = os.path.join(cache_dir, ".store.pkl")
    manifest_path = os.path.join(cache_dir, ".manifest.json")
    man_version, man = _load_manifest(manifest_path)
    # 索引器版本闸门：派生逻辑变了 → 源文件内容指纹**看不见**这件事，必须整体作废 manifest。
    # 否则未改动的源文件会被 `_cache_fresh` 判 fresh，永远沿用旧片段（新字段传播不到存量缓存）。
    schema_ok = man is not None and man_version == SCHEMA_VERSION
    loaded = _load_consolidated(store_pkl) if os.path.isfile(store_pkl) else None
    if schema_ok and loaded is not None:
        if _manifest_matches(man, current):
            _adopt(cm, loaded)
            _write_indexes(cache_dir, cm.store)
            return {"files": len(current), "ingested": 0, "loaded": len(loaded._frags),
                    "pruned": 0, "library": 0, "code": 0, "schema_rebuild": 0}
        # —— 快路径 B：清单有差异 → 在 pickle 基底上做 diff，只重算「内容确实变了」的文件 ——
        # 关键：绝不能在这里退回 _scan_cache（读回上千个 .md 碎片再解析 = 50~140s）。
        # agent 边改代码边检索时每次写盘都会走到这条路径，代价必须是「重蒸馏那几个文件」。
        return _incremental_from_store(cm, loaded, current, man, root, base,
                                       lib_ext, code_ext, cache_dir, store_pkl,
                                       manifest_path, distiller_name, stat)

    # 走到这里有两种情况，都必须**全量重派生**：
    #   (a) 无 pickle / 无 manifest（旧版缓存）；
    #   (b) manifest 存在但版本 != SCHEMA_VERSION（派生逻辑已变）。
    # (b) 的危险点：源文件指纹是吻合的，`_cache_fresh` 会误判 fresh → 故把「可复用的旧片段
    # 集合」置空（只把 existing 留给随后的 prune），做到「一律重派生」。
    # 注意：_scan_cache 绝不能放到函数开头——命中快路径时这本是上千次 .md 读回/解析。
    existing = _scan_cache(cache_dir)
    reuse = existing if schema_ok else {}
    stat["schema_rebuild"] = 1 if (man is not None and not schema_ok) else 0
    shas: dict[str, str] = {}   # {相对 root 的路径: 内容 sha}，供 manifest 记录指纹
    # 本次重建「承认」的落盘片段：origin -> {fid}。用于随后清掉被新派生结果淘汰的旧 .md
    # （源文件变短 / 切块数减少 → 旧 fid 会滞留磁盘，被 load_store / _scan_cache 读回成幽灵片段）。
    written: dict[str, set[str]] = {}

    for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if not os.path.isfile(path):
            continue
        # 跳过忽略目录（如 .venv / node_modules / __pycache__）：既不扫描、也不把其缓存片段留下
        if ignore_dirs and any(part in ignore_dirs for part in path.split("/")):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in lib_ext and ext not in code_ext:
            continue
        try:
            mtime = os.path.getmtime(path)
            size = os.path.getsize(path)
        except OSError:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        # origin key 必须与 ingest 写入一致：lib 用 f"kb:{stem}"，code 用 rel(相对仓库根)
        rel = os.path.relpath(path, base)
        origin_key = f"kb:{stem}" if ext in lib_ext else rel
        stat["files"] += 1
        present_origins.add(origin_key)

        cached = reuse.get(origin_key)
        # 文件级 sha（同时用于「未变→skip」与「变了→rederive」两路，确保落盘与判定同源）
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
            sha = hashlib.sha256(raw).hexdigest()
        except OSError:
            sha = ""
        # manifest 以「相对 root」为键（与 _scan_sources 的键空间一致），code 源的 rel 才与之重合
        shas[os.path.relpath(path, root)] = sha
        if cached and _cache_fresh(cached, mtime=mtime, size=size, sha=sha):
            # 未变 -> 磁盘 load（重建 L3），不重蒸馏
            _load_cached(cm, cache_dir, cached, stat)
            written.setdefault(origin_key, set()).update(c["fid"] for c in cached)
            continue

        # 变了 / 新增 / sha 缺失（兼容旧 cache）-> 重派生
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError as e:  # noqa: BLE001
            print(f"  [skip] 读不了 {path}: {e}")
            continue
        # 旧片段整体作废：文件变短 / 切块数减少时，旧 fid 会滞留 store 与 L3 索引
        # （检索会命中已不存在的行）—— 必须按 origin 成组删除，与 _incremental_from_store 同语义。
        for fid, frag in list(cm.store._frags.items()):
            if getattr(frag, "origin", "") == origin_key:
                cm.store._frags.pop(fid, None)
                cm.store._l1.pop(fid, None)
                cm.store._l2.pop(fid, None)
        if ext in lib_ext:
            n = cm.ingest_library(text, doc_id=f"kb:{stem}")
            stat["library"] += n
        else:
            n = cm.ingest_code(text, name=rel)
            stat["code"] += n
        stat["ingested"] += n
        # 按 origin 匹配，强制重存该文件产生的全部片段（含更新，避免同进程重复 rebuild 时磁盘陈旧）
        for fid, frag in list(cm.store._frags.items()):
            if frag.origin != origin_key:
                continue
            l1 = cm.store._l1.get(fid)
            l2 = cm.store._l2.get(fid)
            save_fragment(cache_dir, frag, l1, l2, mtime, distiller_name,
                          src_sha256=sha, src_size=size)
            written.setdefault(origin_key, set()).add(fid)
    # prune：源已删 -> 清落盘片段（跳过 Session：由 save/load_session 管理，不在 root 扫描内）
    for origin, items in existing.items():
        if origin in present_origins:
            continue
        if any(it.get("source") == "session" for it in items):
            continue
        for c in items:
            fpath = _fid_to_path(cache_dir, c["fid"])
            if os.path.isfile(fpath):
                os.remove(fpath)
                stat["pruned"] += 1

    # 清掉本次重建不再承认的落盘片段（语义见 written 的定义）。
    # 判据必须用 written 而不是 cm.store：store 里可能仍留着旧 fid（Session 片段、
    # 更早 run 的残留），拿 store 当判据会导致「永远扫不掉」。只处理本次真正处理过的 origin，
    # 因此「本轮未重派生（fresh 复用）」与「prune 循环已处理（源已消失）」的 origin 都不受影响。
    for origin, items in existing.items():
        keep = written.get(origin)
        if keep is None:
            continue
        for c in items:
            if c["fid"] in keep:
                continue
            fpath = _fid_to_path(cache_dir, c["fid"])
            if os.path.isfile(fpath):
                os.remove(fpath)
                stat["pruned"] += 1

    _write_indexes(cache_dir, cm.store)
    # 合并 store 落盘（快路径用）：整库一次 pickle，避免下次逐碎片 .md 读回的性能悬崖。
    _save_consolidated(store_pkl, cm.store)
    _save_manifest(manifest_path, current, shas)
    return stat


def save_session(cm, cache_dir: str) -> None:
    """落盘当前会话：buffer（原始消息序列 role+content）+ Session 片段(L0/L1/L2)。

    关闭「火车票缺口」：进程退出前调用一次，重启后 load_session 还原，会话连续性不丢。
    L3 不落盘（由 L1 重建）；Session 片段 src_mtime 用落盘时刻（非文件 mtime，会话无源文件）。
    """
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "session_buffer.json"), "w", encoding="utf-8") as fh:
        json.dump(cm.buffer, fh, ensure_ascii=False)
    for frag in cm.store.all_fragments():
        if frag.source is not Source.SESSION:
            continue
        l1 = cm.store._l1.get(frag.fid)
        l2 = cm.store._l2.get(frag.fid)
        save_fragment(cache_dir, frag, l1, l2, time.time(), "session")


def load_session(cm, cache_dir: str) -> None:
    """加载会话：恢复 buffer + Session 片段到 store（L3 由 L1 重建）。

    与 rebuild（加载 lib/code）互补；推荐顺序：先 load_session 再 rebuild，
    使 L2_overview / L3_index 也含会话标签。两者都跑后 store 含三线、L3 完整。
    """
    bpath = os.path.join(cache_dir, "session_buffer.json")
    if os.path.isfile(bpath):
        try:
            with open(bpath, "r", encoding="utf-8") as fh:
                cm.buffer = json.load(fh)
        except json.JSONDecodeError:
            cm.buffer = []
    if not os.path.isdir(cache_dir):
        return
    for name in os.listdir(cache_dir):
        if not name.endswith(".md") or name in ("L2_overview.md", "L2_callgraphs.md", "L3_index.md"):
            continue
        fpath = os.path.join(cache_dir, name)
        with open(fpath, "r", encoding="utf-8") as fh:
            frag, l1, l2, _ = md_to_fragment(fh.read())
        if frag.source is not Source.SESSION:
            continue
        if frag.fid not in cm.store._frags:
            cm.store.add_fragment(frag)
            if l1 is not None:
                cm.store._l1[l1.fid] = l1
            if l2 is not None:
                cm.store._l2[l2.fid] = l2
    _regen_l3(cm.store)
