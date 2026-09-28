#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/harness.py —— 客观校验层（OBJECTIVE verification, pure code, 0 LLM）

从 demo.py 抽取的「终态校验 / 技术栈感知校验」逻辑，全部是确定性代码，不调用任何模型：

1) 忠实移植的helper（行为与原 demo.py 一致，含所有 prior fix）：
   _eval_pytest_result / _is_js_test / _try_run_js_tests / _js_bin_available /
   detect_stack / _check_dangling_refs / _build_validation / _eval_validation /
   _detect_test_stack_mismatch / _validation_error_signature / _try_package_selfheal /
   _run_test_bar  (单杠校验：工作区测试 count>0 且全绿即完成)

2) 自描述传感器（BaseSensor 子类），统一输出 SensorFact，由 SensorRegistry 统一触发：
   FileDriftSensor / ModuleImportSensor / UnitTestSensor /
   LspDiagnosticSensor / LanguageCliSensor

3) 入口：run_harness_pipeline(workspace, plan, stack) -> List[SensorFact]

注意：_norm_cmd / _exclude_heavy / _workspace_signature / _turn_fingerprint / _loop_nudge /
_loop_replan / _action_fp 已迁移到 state.py，此处只 import 不重定义。
"""

import os
import re
import json
import shlex
import shutil
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .config import (
    WORKSPACE, OUTPUT_BUDGET, DRIFT_CODE_EXTS,
    STATS, VENV_BIN,
    _TEST_FILE_RE, LSP_CMD, LINT_ENABLED, LINT_MAX_ISSUES,
)
from .state import (
    GLOBAL_STATE, _telemetry, _exclude_heavy, _safe_rel,
    RUN_TELEMETRY, _norm_cmd, _action_fp, _HEAVY_DIRS,
)
from .registry import (
    BaseSensor, SensorRegistry, SensorFact, make_fact, sensor, HarnessContext,
)
from swe_agent.log import logger


# ======================================================================
# 本地常量（与 demo.py 保持一致，避免跨模块依赖）
# ======================================================================
_SKIP_SUFFIX = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".bin", ".safetensors", ".gguf",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".zip",
    ".gz", ".tar", ".pdf", ".db", ".sqlite", ".wav", ".mp3", ".mp4",
}
# 前端引用的本地资源前缀：这些不属于「缺失文件」告警范围
_SKIP_REF_PREFIXES = ("http://", "https://", "//", "#", "data:", "mailto:", "{{")

# 校验状态（harness 自持，避免与 config 全局耦合；保持单次进程内累积）
_VAL_SIG_HIST: List[str] = []   # 历次校验失败的根因签名
_VAL_DOOMED_STREAK = 0          # 连续相同根因计数
_SELFHEAL_DONE: set = set()     # 已自助修复过的缺失模块（保证自愈只做一次）
_DRIFT_REPORTED: set = set()    # 已回灌过的漂移项（避免重复刷屏）

# （历史）每轮 shell 校验的「必然失败」回灌机制已移除——反馈回灌整体关闭，
# 失败细节不再注入模型上下文，由循环防护/单杠闸门决定后续。


def _new_subagent_fence() -> Dict[str, Any]:
    """为一次子智能体调用创建独立的 fence 桶。

    把循环闸门状态（shell 失败 streak / 动作计数）按桶隔离，使子智能体的
    shell 失败不会污染父 agent 的全局 streak，反之亦然——避免「子 agent 跑挂测试
    把父的必然失败闸门误触发 / 或被父清零掩盖」这类抓不住问题的交叉污染。
    父路径（顶层 agent）不使用本桶，继续走模块级全局变量。
    """
    return {"shell_fail_streak": 0, "shell_fail_last": "", "actions": {}}


# ======================================================================
# shell 执行（lazy import，避免与 .tools 产生循环依赖）
# ======================================================================
def _fallback_exec_shell(cmd: str, timeout: int = 120) -> str:
    """demo.exec_shell 的兜底实现：跑命令、捕获输出、附带 exit code。"""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=str(WORKSPACE))
        out = (p.stdout or "") + (p.stderr or "")
        return out + f"\nexit code: {p.returncode}\n"
    except Exception as e:  # noqa: BLE001
        return f"{e}\nexit code: 1\n"


def _get_exec_shell():
    try:
        from .tools import exec_shell  # type: ignore
        return exec_shell
    except Exception:  # noqa: BLE001
        return _fallback_exec_shell


# ======================================================================
# 忠实移植：技术栈感知校验核心
# ======================================================================
def _eval_pytest_result(final: str):
    """解析 pytest 输出，返回 (all_pass, passed_n, failed_n, error_n)。"""
    m_pass = re.search(r"(\d+)\s+passed", final)
    m_fail = re.search(r"(\d+)\s+failed", final)
    m_err = re.search(r"(\d+)\s+error", final)
    passed_n = int(m_pass.group(1)) if m_pass else 0
    failed_n = int(m_fail.group(1)) if m_fail else 0
    error_n = int(m_err.group(1)) if m_err else 0
    all_pass = ("exit code: 0" in final) and failed_n == 0 and error_n == 0 and passed_n > 0
    return all_pass, passed_n, failed_n, error_n


def _is_js_test(f: str) -> bool:
    name = Path(f).name
    return f.endswith((".js", ".jsx", ".ts", ".tsx")) and (
        name.startswith("test_") or ".test." in name or name.endswith(".spec.js")
        or name.endswith(".spec.ts"))


def _module_root(lang: str) -> Optional[Path]:
    """在 WORKSPACE 内寻找某语言的模块根（含清单文件的最浅目录）。"""
    manifest = {"go": "go.mod", "rust": "Cargo.toml",
                "java": ("pom.xml", "build.gradle", "build.gradle.kts")}.get(lang)
    if not manifest:
        return None
    manifests = manifest if isinstance(manifest, tuple) else (manifest,)
    best: Optional[Path] = None
    for p in WORKSPACE.rglob("*"):
        if p.is_file() and p.name in manifests:
            cand = p.parent
            if best is None or len(cand.parts) < len(best.parts):
                best = cand
    return best


def _file_contains(rel: str, substr: str) -> bool:
    try:
        return substr in (WORKSPACE / rel).read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return False


def _present_langs(workspace: Path) -> set:
    """扫描工作区，返回实际存在的语言集合（用于按需启用 linter）。"""
    langs = set()
    for p in workspace.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix == ".py":
            langs.add("python")
        elif p.suffix == ".go":
            langs.add("go")
        elif p.suffix == ".rs":
            langs.add("rust")
        elif p.suffix in (".ts", ".tsx", ".js", ".jsx"):
            langs.add("typescript")
            langs.add("javascript")
        elif p.suffix == ".java":
            langs.add("java")
    return langs


def _ensure_go_module() -> Optional[str]:
    """Go 任务兜底：工作区有 .go 文件但缺 go.mod 时，自动 go mod init。

    小模型（如 qwen-7b）经常忘记执行 `go mod init`，导致 `go test/build` 一律报
    'no main module' 而永远无法收敛。这是确定性 BUILD 层兜底，不依赖模型自觉。
    幂等：go.mod 已存在则直接返回。"""
    go_files = [p for p in WORKSPACE.rglob("*.go") if p.is_file()]
    if not go_files:
        return None
    if (WORKSPACE / "go.mod").exists():
        return None
    mod = WORKSPACE.name or "app"
    try:
        r = subprocess.run(["go", "mod", "init", mod], cwd=str(WORKSPACE),
                           capture_output=True, text=True, timeout=60)
        if (WORKSPACE / "go.mod").exists():
            return f"go mod init {mod} 成功（自动兜底）"
        return f"go mod init 失败: {r.stderr.strip()[:200]}"
    except Exception as e:  # noqa: BLE001
        return f"go mod init 异常: {e}"


def _try_run_js_tests(test_files: List[str], base_dir: Path) -> Tuple[Optional[str], str]:
    """尽量用 jest 运行 JS 测试文件；不可行时返回 (None, 说明) 走语法兜底。"""
    if not test_files:
        return None, ""
    allow_install = os.environ.get("DEMO_JS_TEST_INSTALL", "1") != "0"
    pkg = base_dir / "package.json"
    created_pkg = False

    def _uses_esm(files):
        for f in files:
            fp = base_dir / f
            if fp.exists():
                try:
                    txt = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if re.search(r"^\s*import\s.+from|^\s*export\s", txt, re.M):
                    return True
        return False

    uses_esm = _uses_esm(test_files)
    babel_deps = {"babel-jest": "^29", "@babel/core": "^7", "@babel/preset-env": "^7"}
    jest_cfg = {
        "testEnvironment": "jsdom",
        "setupFilesAfterEnv": ["<rootDir>/jest.setup.cjs"],
    }
    if not pkg.exists():
        created_pkg = True
        pkg_data = {
            "name": "agent-sandbox-app", "private": True,
            "scripts": {"test": "jest"},
            "devDependencies": {"jest": "^29", "jsdom": "^24", **babel_deps},
            "jest": jest_cfg,
        }
        pkg.write_text(json.dumps(pkg_data, indent=2), encoding="utf-8")
    else:
        try:
            pkg_data = json.loads(pkg.read_text(encoding="utf-8"))
        except Exception:
            pkg_data = {}
        pkg_data.setdefault("devDependencies", {})
        changed = False
        for k, v in babel_deps.items():
            if k not in pkg_data["devDependencies"]:
                pkg_data["devDependencies"][k] = v
                changed = True
        existing_jest = pkg_data.get("jest")
        if isinstance(existing_jest, dict):
            if "testEnvironment" not in existing_jest:
                existing_jest["testEnvironment"] = "jsdom"
                changed = True
            if "setupFilesAfterEnv" not in existing_jest:
                existing_jest["setupFilesAfterEnv"] = ["<rootDir>/jest.setup.cjs"]
                changed = True
            pkg_data["jest"] = existing_jest
        elif existing_jest is None:
            pkg_data["jest"] = jest_cfg
            changed = True
        if changed:
            pkg.write_text(json.dumps(pkg_data, indent=2), encoding="utf-8")
    babel_cfg = base_dir / "babel.config.cjs"
    babel_cfg.write_text(
        "module.exports = { presets: [['@babel/preset-env', { targets: { node: 'current' } }]] };\n",
        encoding="utf-8")
    setup_cfg = base_dir / "jest.setup.cjs"
    setup_cfg.write_text(
        "if (typeof window !== 'undefined') {\n"
        "  for (const k of ['Event','CustomEvent','EventTarget','Node','HTMLElement',\n"
        "       'HTMLInputElement','HTMLFormElement','HTMLButtonElement','DocumentFragment',\n"
        "       'getComputedStyle','MouseEvent','KeyboardEvent']) {\n"
        "    try { if (window[k] !== undefined && global[k] === undefined) global[k] = window[k]; } catch (e) {}\n"
        "  }\n"
        "  if (global.window === undefined) global.window = window;\n"
        "  if (global.document === undefined) global.document = window.document;\n"
        "}\n",
        encoding="utf-8")

    def _runnable(bin_name: str) -> bool:
        p = subprocess.run(
            f"cd {shlex.quote(str(base_dir))} && test -x ./node_modules/.bin/{bin_name}",
            shell=True, capture_output=True, text=True, timeout=20)
        return p.returncode == 0

    jest_ok = _runnable("jest")
    babel_ok = _runnable("babel-jest")

    if not jest_ok:
        if not allow_install:
            try:
                babel_cfg.unlink()
            except Exception:
                pass
            return None, (f"[校验] 检测到 {len(test_files)} 个 JS 测试文件，但运行环境缺少 jest"
                          "（DEMO_JS_TEST_INSTALL=0 且未预装），无法执行功能测试；"
                          "仅完成语法级 node --check。")
        logger.info('%s', f"[校验] 未检测到 jest，尝试安装 jest+jsdom+babel（ESM 支持）……{('(已探测到 ESM 语法)' if uses_esm else '')}")
        ip = subprocess.run(
            f"cd {shlex.quote(str(base_dir))} && npm install --no-audit --no-fund --silent",
            shell=True, capture_output=True, text=True, timeout=300)
        if ip.returncode != 0:
            if created_pkg:
                try:
                    pkg.unlink()
                except Exception:
                    pass
            try:
                babel_cfg.unlink()
            except Exception:
                pass
            return None, (f"[校验] 检测到 {len(test_files)} 个 JS 测试文件，但依赖安装失败"
                           "（可能无网络或 npm 不可用），无法执行功能测试；"
                           "仅完成语法级 node --check，功能正确性未自动验证。")
    elif uses_esm and not babel_ok:
        if allow_install:
            logger.info('%s', '[校验] 检测到 ESM 语法且 babel-jest 缺失，补充安装 babel……')
            subprocess.run(
                f"cd {shlex.quote(str(base_dir))} && npm install --no-audit --no-fund --silent",
                shell=True, capture_output=True, text=True, timeout=300)

    files_arg = " ".join(shlex.quote(f) for f in test_files)
    cmd = f"cd {shlex.quote(str(base_dir))} && npx jest --passWithNoTests {files_arg}"
    return cmd, ""


def _js_bin_available(runner: Optional[str], base_dir: Path) -> bool:
    if not runner:
        return False
    p = subprocess.run(
        f"cd {shlex.quote(str(base_dir))} && "
        f"(test -x ./node_modules/.bin/{runner} || npx --no-install {runner} --version >/dev/null 2>&1)",
        shell=True, capture_output=True, text=True, timeout=20)
    return p.returncode == 0


def detect_stack(workspace: Optional[Path] = None) -> Dict[str, Any]:
    """基于本轮生成的文件 + 工作区现状，判定技术栈并返回默认工具链。

    `workspace` 参数化（默认全局 WORKSPACE，调用时解析以支持 monkeypatch 重绑），
    使调用方（如 _run_test_bar）可针对特定工作区判定技术栈，避免隐式依赖模块级
    全局 WORKSPACE（#25 去全局耦合）。
    """
    if workspace is None:
        workspace = WORKSPACE
    gen = [_safe_rel(f) for f in GLOBAL_STATE.get("generated_files", [])]
    files = set(gen)
    for p in workspace.rglob("*"):
        if p.is_file() and p.suffix.lower() not in _SKIP_SUFFIX:
            files.add(str(p.relative_to(workspace)))
    files = sorted(files)
    has_pkg = any(f.endswith("package.json") for f in files)
    has_py = any(f.endswith(".py") for f in files)
    has_js = any(f.endswith((".js", ".jsx", ".ts", ".tsx")) for f in files)
    has_html = any(f.endswith(".html") for f in files)
    has_go = any(f.endswith("go.mod") for f in files) or any(
        f.endswith(".go") for f in files)
    has_rust = any(f.endswith("Cargo.toml") for f in files)
    has_java = any(f.endswith(("pom.xml", "build.gradle", "build.gradle.kts")) for f in files)

    pkg: Dict[str, Any] = {}
    if has_pkg:
        try:
            pkg_path = next(f for f in files if f.endswith("package.json"))
            pkg = json.loads((workspace / pkg_path).read_text(encoding="utf-8") or "{}")
        except Exception:
            pkg = {}
    deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
    scripts = pkg.get("scripts") or {}
    test_script = scripts.get("test")
    runner = None
    for r in ("jest", "vitest", "mocha", "tape", "ava"):
        if r in deps:
            runner = r
            break

    if has_pkg:
        if test_script:
            test_cmd = "npm test"
        elif runner == "jest":
            test_cmd = "npx jest"
        elif runner == "vitest":
            test_cmd = "npx vitest run"
        elif runner == "mocha":
            test_cmd = "npx mocha"
        else:
            test_cmd = None
        lang = "node"
    elif has_py:
        lang, test_cmd = "python", "pytest"
    elif has_js or has_html:
        lang, test_cmd = "frontend", None
    elif has_go:
        lang, test_cmd = "go", "go test ./..."
    elif has_rust:
        lang, test_cmd = "rust", None
    elif has_java:
        lang, test_cmd = "java", None
    else:
        lang, test_cmd = "unknown", None

    return {"lang": lang, "test_cmd": test_cmd, "runner": runner,
            "has_pkg": has_pkg, "has_html": has_html, "has_js": has_js,
            "has_py": has_py, "has_go": has_go, "has_rust": has_rust,
            "has_java": has_java, "files": files}


def _check_dangling_refs(files: List[str]) -> List[str]:
    """扫描 HTML 里 href/src 引用的本地文件，找出磁盘上不存在的（如缺 style.css）。"""
    missing = []
    for f in files:
        if not f.endswith(".html"):
            continue
        try:
            text = (WORKSPACE / f).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for ref in re.findall(r'(?:href|src)\s*=\s*"([^"]+)"', text):
            if ref.startswith(_SKIP_REF_PREFIXES):
                continue
            target = (WORKSPACE / f).parent / ref
            if not target.exists():
                missing.append(f"{f} 引用了缺失文件: {ref}")
    return missing


def _build_validation(stack: Dict[str, Any], workspace: Optional[Path] = None):
    """返回 (gen_tests, val_cmd_or_None, note)。

    注意：历史上这里曾有一个 _flatten_workspace 步骤，会把 WORKSPACE 下嵌套的
    test_*.py / src 拉平到根目录。该步骤会直接 shutil.move 文件，一旦 WORKSPACE
    落到仓库根（运行 pytest 的默认 cwd）就会把 tests/ 与 .venv 下所有 test_*.py
    搬到仓库根、搬空工程，且属于「框架改文件系统」的脆弱副作用。已按用户决策整体
    移除——单一根目录契约改由模型写对路径 + tester 按真实位置递归读取来保证，
    不再用框架级文件搬移兜底。
    """
    if workspace is None:
        workspace = WORKSPACE
    gen = [_safe_rel(f) for f in GLOBAL_STATE.get("generated_files", [])]
    lang = stack["lang"]
    ws_tests = [f for f in [_safe_rel(x) for x in stack.get("files", [])]
                if f.endswith(".py") and (Path(f).name.startswith("test_")
                                         or "/tests/" in f.replace("\\", "/"))]
    if lang == "python":
        gen_tests = [f for f in gen if f.endswith(".py") and (
            Path(f).name.startswith("test_") or "/tests/" in f.replace("\\", "/"))]
        # 合并 generated_files 与磁盘实测，去重后只保留「真实存在于工作区」的文件，
        # 过滤幽灵路径（模型可能在 generated_files 记了 tests/test_x.py，但实际写在根目录 test_x.py，
        # 旧逻辑 `gen_tests or ws_tests` 会短路到幽灵列表，导致 pytest 报 file not found）。
        eff_tests = [_f for _f in dict.fromkeys(list(gen_tests) + list(ws_tests))
                     if (workspace / _f).exists()]
        if eff_tests:
            return eff_tests, "pytest -q " + " ".join(shlex.quote(f) for f in eff_tests), ""
        return [], "pytest -q", ""
    if lang == "node":
        js_files = _exclude_heavy([f for f in stack["files"] if f.endswith((".js", ".jsx", ".ts", ".tsx"))])
        js_tests = [f for f in gen if _is_js_test(f)]
        # 只保留真实存在的测试文件，过滤 generated_files 里的幽灵路径
        js_tests = [_f for _f in dict.fromkeys(js_tests) if (workspace / _f).exists()]
        note = ""
        if stack.get("test_cmd") and _js_bin_available(stack.get("runner"), workspace):
            test_cmd = stack["test_cmd"]
            if js_files:
                return js_tests, " && ".join("node --check " + shlex.quote(f) for f in js_files) + " && " + test_cmd, ""
            return js_tests, test_cmd, ""
        if js_tests:
            cmd, note = _try_run_js_tests(js_tests, workspace)
            if cmd:
                return js_tests, cmd, note
        if js_files:
            return js_tests, " && ".join("node --check " + shlex.quote(f) for f in js_files), note
        return js_tests, None, "[校验] 未找到可执行测试，且无 JS 文件可校验。"
    if lang == "frontend":
        js_files = _exclude_heavy([f for f in stack["files"] if f.endswith((".js", ".jsx", ".ts", ".tsx"))])
        if not js_files:
            return [], None, "[校验] 纯前端但无 JS 文件，无可校验内容。"
        cmd_parts = ["node --check " + shlex.quote(f) for f in js_files]
        note = ""
        js_tests = _exclude_heavy([f for f in stack["files"] if _is_js_test(f)])
        if js_tests:
            run_cmd, tnote = _try_run_js_tests(js_tests, workspace)
            if run_cmd:
                cmd_parts.append(run_cmd)
            else:
                note = tnote
        return js_tests, " && ".join(cmd_parts), note
    if lang == "go":
        go_tests = [f for f in stack["files"] if f.endswith("_test.go")]
        root = _module_root("go")
        cd = f"cd {shlex.quote(str(root))} && " if root else ""
        if go_tests:
            return go_tests, f"{cd}go test ./...", ""
        return [], f"{cd}go build ./...", "[校验] 无 _test.go，退化为 go build 编译门禁。"
    if lang == "rust":
        rs_files = [f for f in stack["files"] if f.endswith(".rs")]
        has_test_attr = any(_file_contains(f, "#[test]") or _file_contains(f, "#[tokio::test]")
                            for f in rs_files)
        has_tests_dir = any("tests/" in f.replace("\\", "/") for f in stack["files"])
        root = _module_root("rust")
        cd = f"cd {shlex.quote(str(root))} && " if root else ""
        if has_test_attr or has_tests_dir:
            return [], f"{cd}cargo test", ""
        return [], f"{cd}cargo build", "[校验] 无 #[test]，退化为 cargo build 编译门禁。"
    if lang == "java":
        java_tests = [f for f in stack["files"]
                      if f.endswith("Test.java") or "/src/test/" in f.replace("\\", "/")]
        root = _module_root("java")
        cd = f"cd {shlex.quote(str(root))} && " if root else ""
        if any(f.endswith("pom.xml") for f in stack["files"]):
            tool = "mvn -q"
        elif any(f.endswith(("build.gradle", "build.gradle.kts")) for f in stack["files"]):
            tool = "gradle"
        else:
            tool = "mvn -q"
        if java_tests:
            return java_tests, f"{cd}{tool} test", ""
        return [], f"{cd}{tool} compile", "[校验] 无测试，退化为编译门禁。"
    return [], None, "[校验] 无法判定技术栈或无源码文件，无法自动校验。"


def _eval_validation(final: str, stack: Dict[str, Any]):
    """对校验输出做稳健判读，返回 (all_pass, detail_str)。"""
    lang = stack["lang"]
    if lang == "python":
        ok, p, f, e = _eval_pytest_result(final)
        return ok, f"pytest: {p} passed / {f} failed / {e} error"
    if lang == "node":
        m_p = re.search(r"(\d+)\s+passing", final) or re.search(r"Tests:\s*(\d+)\s+passed", final)
        m_f = re.search(r"(\d+)\s+failing", final) or re.search(r"Tests:\s*(\d+)\s+failed", final)
        p = int(m_p.group(1)) if m_p else 0
        f = int(m_f.group(1)) if m_f else 0
        ok = ("exit code: 0" in final) and f == 0 and p > 0 and "FAIL" not in final
        return ok, f"js tests: {p} passing / {f} failing"
    if lang == "frontend":
        syntax_ok = ("exit code: 0" in final) and ("SyntaxError" not in final)
        jest_ok = True
        detail = "node --check 语法门禁"
        if "passing" in final or "Tests:" in final:
            m_p = re.search(r"Tests:\s*(\d+)\s+passed", final) or re.search(r"(\d+)\s+passing", final)
            m_f = re.search(r"Tests:\s*(\d+)\s+failed", final) or re.search(r"(\d+)\s+failing", final)
            p = int(m_p.group(1)) if m_p else 0
            f_ = int(m_f.group(1)) if m_f else 0
            jest_ok = (f_ == 0 and p > 0 and "FAIL" not in final)
            detail = f"node --check + jest: {p} passed / {f_} failed"
        ok = syntax_ok and jest_ok
        return ok, detail
    if lang == "go":
        ok = ("exit code: 0" in final) and ("FAIL" not in final) and ("# command" not in final)
        npass = len(re.findall(r"--- PASS:", final))
        nfail = len(re.findall(r"--- FAIL:", final))
        return ok, f"go test: {npass} passed / {nfail} failed"
    if lang == "rust":
        m_p = re.search(r"test result: ok\. (\d+) passed", final)
        m_f = re.search(r"test result: FAILED\. (\d+) failed", final)
        p = int(m_p.group(1)) if m_p else 0
        f_ = int(m_f.group(1)) if m_f else 0
        ok = ("exit code: 0" in final) and f_ == 0 and ("FAILED" not in final)
        return ok, f"cargo test: {p} passed / {f_} failed"
    if lang == "java":
        ok = ("exit code: 0" in final) and ("BUILD FAILURE" not in final) and ("FAIL" not in final)
        return ok, "java test/compile"
    return False, "no validation"


def _detect_test_stack_mismatch(stack: Dict[str, Any]) -> Tuple[bool, List[str], str]:
    """测试栈是否与实际技术栈不匹配（结构性错误，非偶发失败）。"""
    required = stack["lang"]
    files = stack.get("files", [])
    py_tests = [f for f in files if f.endswith(".py") and (
        Path(f).name.startswith("test_") or "/tests/" in f.replace("\\", "/"))]
    js_tests = [f for f in files if _is_js_test(f)]
    if required == "python":
        if py_tests:
            return False, [], "python"
        if js_tests:
            return True, js_tests, "python"
        return False, [], "python"
    if required in ("node", "frontend"):
        if js_tests:
            return False, [], required
        if py_tests:
            return True, py_tests, required
        return False, [], required
    return False, [], required


def _validation_error_signature(final: str, stack: Dict[str, Any]) -> str:
    """从校验输出抽取『失败根因签名』，用于判定『必然失败』。"""
    pats = [
        r"(ModuleNotFoundError:\s*No module named '[^']*')",
        r"(ImportError:\s*cannot import name '[^']*')",
        r"(ImportError:\s*[^;\n]{0,80})",
        r"(AttributeError:\s*'[^']*' object has no attribute '[^']*')",
        r"(TypeError:\s*[^;\n]{0,80})",
        r"(SyntaxError:\s*[^;\n]{0,80})",
        r"(Cannot find module '[^']*')",
        r"(ReferenceError:\s*[^;\n]{0,80})",
        r"(AssertionError[^;\n]{0,80})",
        r"(Error collecting [^\n]{0,140})",
    ]
    for p in pats:
        m = re.search(p, final)
        if m:
            return m.group(1).strip().replace("\n", " ")[:160]
    tail = re.sub(r"\s+", " ", final[-500:]).strip()
    return "sig:" + hashlib.md5(tail.encode("utf-8")).hexdigest()[:10]


def _try_package_selfheal(final: str, stack: Dict[str, Any]) -> Tuple[bool, str]:
    """包结构自愈（TDD 安全网）：Python 测试因『本地模块未打包』ModuleNotFoundError 时自动补 __init__.py / conftest.py。"""
    global _SELFHEAL_DONE
    if stack.get("lang") != "python":
        return False, ""
    if "No module named" not in final:
        return False, ""
    m = re.search(r"ModuleNotFoundError:\s*No module named '([^']+)'", final)
    if not m:
        return False, ""
    mod = m.group(1).split(".")[0]
    if mod in _SELFHEAL_DONE:
        return False, ""
    created: List[str] = []
    dirs_to_add: List[Path] = []
    target = WORKSPACE / mod
    if target.is_dir():
        if not (target / "__init__.py").exists():
            (target / "__init__.py").write_text("", encoding="utf-8")
            created.append(f"{mod}/__init__.py")
        dirs_to_add.append(target)
    else:
        hit = None
        for p in WORKSPACE.rglob(f"{mod}.py"):
            if p.name == f"{mod}.py":
                hit = p.parent
                break
        if hit is None:
            _SELFHEAL_DONE.add(mod)
            return False, ""
        dirs_to_add.append(hit)
    dirs_to_add.append(WORKSPACE)
    need_lines = [f'sys.path.insert(0, r"{d}")' for d in sorted({str(x) for x in dirs_to_add})]
    cp = WORKSPACE / "conftest.py"
    if not cp.exists():
        cp.write_text("import sys, os\n" + "\n".join(need_lines) + "\n", encoding="utf-8")
        created.append("conftest.py")
    else:
        txt = cp.read_text(encoding="utf-8", errors="ignore")
        append = [ln for ln in need_lines if ln not in txt]
        if append:
            cp.write_text(txt.rstrip() + "\n" + "\n".join(append) + "\n", encoding="utf-8")
            created.append("conftest.py(append sys.path)")
    _SELFHEAL_DONE.add(mod)
    if created:
        return True, "harness 已自动补全包结构（" + ", ".join(created) + \
                     "），请直接重新运行 pytest 验证，不要再手动改包结构。"
    return False, ""


def _is_test_cmd(cmd: str) -> bool:
    """判断 shell 命令是否为「运行测试」类命令（需要接入必然失败防护）。"""
    c = (cmd or "").lower()
    return ("pytest" in c) or ("jest" in c) or ("npm test" in c) \
        or ("npm run test" in c) or ("unittest" in c)


# ======================================================================
# 单杠校验：工作区测试 count>0 且全绿即完成（唯一的成功标准）
# ======================================================================
def run_lint(lang: Optional[str] = None) -> Tuple[str, str, List[str]]:
    """编码后静态校验（pytest 之前的第 3 道本地闸门）。返回 (verdict, detail, issues)。

    verdict ∈ pass / fail / no_files。issues 为具体 lint 错误（截断到 LINT_MAX_ISSUES）。
    仅做「不依赖运行的静态检查」：Python 用 py_compile（语法/缩进）+ 可选 flake8；
    Go 用 go vet；Node 用 node --check。不替代 pytest（语义正确性由 pytest + tester 把关）。
    """
    if not LINT_ENABLED:
        return "pass", "LINT_ENABLED=0，跳过 lint。", []
    ws = WORKSPACE
    if lang is None:
        lang = detect_stack().get("lang", "python")
    issues: List[str] = []
    try:
        files = [p for p in ws.rglob("*") if p.is_file()]
    except Exception:
        return "pass", "无法读取工作区，跳过 lint。", []
    if lang == "python":
        # 收敛扫描范围：只编译 agent 实际编写的源/测试文件，跳过 conftest/__init__（harness 自愈产物，
        # 非业务代码，且曾因解释器 std 流损坏在此处崩溃），并对残留超大工作区封顶，避免拖垮 lint。
        py_files = [p for p in files
                    if p.suffix == ".py" and "__pycache__" not in p.parts
                    and p.name not in ("conftest.py", "__init__.py")]
        MAX_LINT_FILES = 80
        if len(py_files) > MAX_LINT_FILES:
            py_files = py_files[:MAX_LINT_FILES]
        if not py_files:
            return "no_files", "工作区无 .py 源文件，跳过 lint。", []
        py = str(VENV_BIN / "python")
        warns: List[str] = []  # 环境告警：不计入 lint 失败判定
        for p in py_files:
            try:
                r = subprocess.run([py, "-m", "py_compile", str(p)],
                                  capture_output=True, text=True, timeout=60,
                                  cwd=str(ws))
            except Exception as e:
                # 子进程起不来（环境/资源问题）→ 视为环境抖动，记警告不阻断
                warns.append(f"{p.name}: py_compile 异常 {e}")
                continue
            if r.returncode != 0:
                stderr = (r.stderr or r.stdout or "")
                # 解释器自身崩溃（init_sys_streams / Bad file descriptor 等 std 流损坏）属于环境故障，
                # 不是代码语法错误，不能据此判 lint fail 把 agent 卡死在一轮轮无效重编码上。
                if ("Fatal Python error" in stderr or "init_sys_streams" in stderr
                        or "Bad file descriptor" in stderr):
                    warns.append(f"{p.name}: [环境告警] 解释器启动失败（非代码错误），跳过该文件编译")
                    continue
                issues.append(f"{p.name}: {_first_lines(stderr, 6)}")
        # 可选 flake8（若环境已装）
        try:
            r2 = subprocess.run([py, "-m", "flake8", "--max-line-length=200",
                                 "--select=E9,F63,F7,F82", str(ws)],
                                capture_output=True, text=True, timeout=120, cwd=str(ws))
            if r2.returncode == 0 and r2.stdout.strip():
                for line in r2.stdout.strip().splitlines():
                    issues.append(line)
        except Exception:
            pass
    elif lang == "go":
        gomod = list(ws.rglob("go.mod"))
        if not gomod:
            return "no_files", "工作区无 go.mod，跳过 lint。", []
        try:
            r = subprocess.run(["go", "vet", "./..."], capture_output=True, text=True,
                              timeout=180, cwd=str(gomod[0].parent))
            if r.returncode != 0:
                issues.append(_first_lines(r.stderr or r.stdout, 10))
        except Exception as e:
            issues.append(f"go vet 异常 {e}")
    elif lang in ("node", "frontend"):
        js = [p for p in files if p.suffix in (".js", ".jsx", ".ts", ".tsx")]
        if not js:
            return "no_files", "工作区无 JS/TS 文件，跳过 lint。", []
        for p in js:
            try:
                r = subprocess.run(["node", "--check", str(p)],
                                  capture_output=True, text=True, timeout=60)
            except Exception:
                continue
            if r.returncode != 0:
                issues.append(f"{p.name}: {_first_lines(r.stderr, 4)}")
    else:
        return "no_files", f"未知语言 {lang}，跳过 lint。", []
    issues = issues[: (LINT_MAX_ISSUES or 40)]
    if issues:
        return "fail", f"lint 发现 {len(issues)} 个问题：\n" + "\n".join(issues), issues
    warn_note = (f"（{len(warns)} 条环境告警已忽略，不影响判定）" if warns
                 else "")
    if warns:
        warn_note += "\n" + "\n".join(warns)
    return "pass", "lint 通过（无语法/未定义错误）。" + warn_note, []


def _first_lines(text: str, n: int) -> str:
    return "\n".join((text or "").splitlines()[:n])


def _run_test_bar(messages: List[Dict[str, str]],
                  workspace: Optional[Path] = None) -> Tuple[str, str, str]:
    """单杠校验：工作区测试 `count>0 且全绿` -> "pass"；有测试但失败 -> "fail"；无测试 -> "no_tests"。

    这是 agent 完成的唯一判定标准。失败明细回灌给 Executor，逼它 read_file 定位并修复，
    而非换思路甩锅。无隐藏测试、无交付物检查、无必然失败熔断——只有“测试在跑且全过”。

    判据一律来自「真实执行结果」（exit code + 解析器报出的 passed/failed/error 计数），
    不做任何静态猜测：历史上此处曾有一层 AST 审计（_audit_tests）试图用纯 stdlib ast
    反作弊（拦 assert True / 空测试），但它只扫 `tree.body` 顶层、看不见 `class TestXxx`
    里的测试方法，把 pytest 4 passed 的真实通过误判为 `no_test_functions` 连续 2 次
    触发 VAL_DOOMED_THRESHOLD 判 unsolvable（2026-09-08 实锤，logs/harness.log 里有
    6 次同类误杀）。删掉它：单杠闸门只认执行事实，测试「质量」交给独立 tester 验收，
    绝不让手搓启发式压过 ground truth。
    """
    _ensure_go_module()  # Go 兜底：确保 go.mod 存在
    ws = workspace if workspace is not None else WORKSPACE
    stack = detect_stack(ws)
    lang = stack["lang"]
    if not _workspace_has_tests(lang, ws):
        return "no_tests", "工作区尚无任何测试文件", (
            "请在实现之外编写测试（Python: test_*.py + pytest；Go: *_test.go + go test），"
            "并用对应命令跑通后再结束。")
    gen_tests, val_cmd, note = _build_validation(stack, ws)
    logger.info('%s', f'\n执行单杠校验（技术栈={lang}）……')
    if val_cmd:
        logger.info('%s', f'[cmd] {val_cmd}')
        final = _get_exec_shell()(val_cmd, timeout=240)
        logger.info('%s', final)
    else:
        final = note
        logger.info('%s', final)
    all_pass, detail = _eval_validation(final, stack)
    GLOBAL_STATE["final_validation"] = {"passed": bool(all_pass), "detail": detail}
    if all_pass:
        return "pass", detail, final
    return "fail", detail, final


def _workspace_has_tests(lang: str, workspace: Optional[Path] = None) -> bool:
    """扫描工作区是否已有测试文件（按技术栈判断），不依赖 generated_files 状态。

    `workspace` 参数化（默认全局 WORKSPACE，调用时解析），见 detect_stack 同款说明。
    """
    ws = workspace if workspace is not None else WORKSPACE
    try:
        files = [p for p in ws.rglob("*") if p.is_file()]
    except Exception:
        return False
    if lang == "python":
        return any((p.name.startswith("test_") or p.name.endswith("_test.py"))
                   and p.suffix == ".py" and "__pycache__" not in p.parts for p in files)
    if lang == "go":
        return any(p.name.endswith("_test.go") for p in files)
    if lang in ("node", "frontend"):
        return any(_is_js_test(str(p)) for p in files)
    if lang == "rust":
        return any("tests" in p.parts or _file_contains(str(p), "#[test]")
                   or _file_contains(str(p), "#[tokio::test]") for p in files)
    if lang == "java":
        return any(p.name.endswith("Test.java") or "/src/test/" in str(p).replace("\\", "/")
                   for p in files)
    return False

# ======================================================================
# 自描述传感器（Objective verification sensors）
# ======================================================================
@sensor("file_drift",
        "Compare disk code files against the GLM plan contract modules; flag unplanned new files as drift.",
        severity="error")
class FileDriftSensor(BaseSensor):
    def run(self, ctx: HarnessContext) -> SensorFact:
        try:
            plan = ctx.plan if isinstance(ctx.plan, dict) else (GLOBAL_STATE.get("plan") or {})
            planned = {m.get("path") for m in (plan.get("modules") or []) if m.get("path")}
            if not planned:
                return make_fact("file_drift", True, "no plan contract; drift check skipped", severity="info")
            issues: List[str] = []
            ws = WORKSPACE.resolve()
            for p in sorted(ws.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in DRIFT_CODE_EXTS:
                    continue
                if any(part in _HEAVY_DIRS for part in p.parts):
                    continue
                try:
                    rel = str(p.relative_to(ws))
                except Exception:
                    continue
                if _TEST_FILE_RE.search(rel) or rel.startswith("tests/"):
                    continue
                if rel not in planned:
                    issues.append(rel)
            new_issues = [i for i in issues if i not in _DRIFT_REPORTED]
            if new_issues:
                _DRIFT_REPORTED.update(new_issues)
                return make_fact("file_drift", False,
                                 "unplanned files (drift): " + ", ".join(new_issues),
                                 payload={"drift": new_issues})
            return make_fact("file_drift", True, "no file drift vs plan", payload={"drift": []})
        except Exception as e:  # noqa: BLE001
            return make_fact("file_drift", True, f"drift check error skipped: {e}", severity="info")


@sensor("module_import",
        "Attempt to import/compile each module declared in the plan/generated files; "
        "Python uses importlib; Go/Rust/Java use a build gate. Flag failures.",
        severity="error")
class ModuleImportSensor(BaseSensor):
    def run(self, ctx: HarnessContext) -> SensorFact:
        try:
            stack = ctx.stack if isinstance(ctx.stack, dict) else detect_stack()
            lang = stack.get("lang", "unknown")
            if lang == "python":
                return self._python_import()
            if lang in ("go", "rust", "java"):
                return self._compile_gate(lang)
            return make_fact("module_import", True,
                             f"module import/compile check skipped for stack '{lang}' "
                             f"(no module-import concept)", severity="info")
        except Exception as e:  # noqa: BLE001
            return make_fact("module_import", True, f"import check error skipped: {e}", severity="info")

    @staticmethod
    def _python_import() -> SensorFact:
        import importlib
        plan = GLOBAL_STATE.get("plan") or {}
        mods = [m.get("path") for m in (plan.get("modules") or []) if m.get("path")]
        for f in GLOBAL_STATE.get("generated_files", []):
            if str(f).endswith(".py"):
                mods.append(str(f))
        failures: List[str] = []
        for m in mods:
            rel = _safe_rel(m)
            if not rel.endswith(".py"):
                continue
            modname = rel[:-3].replace("/", ".").replace("\\", ".")
            try:
                importlib.import_module(modname)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{modname}: {type(e).__name__}: {e}")
        if failures:
            return make_fact("module_import", False,
                             "import failures: " + "; ".join(failures[:10]),
                             payload={"failures": failures})
        return make_fact("module_import", True, "all declared modules importable",
                         payload={"failures": []})

    @staticmethod
    def _compile_gate(lang: str) -> SensorFact:
        root = _module_root(lang)
        if lang == "go":
            if not shutil.which("go"):
                return make_fact("module_import", True, "go not installed; compile gate skipped",
                                 severity="info")
            cmd = f"cd {shlex.quote(str(root))} && go build ./..." if root else "go build ./..."
        elif lang == "rust":
            if not shutil.which("cargo"):
                return make_fact("module_import", True, "cargo not installed; compile gate skipped",
                                 severity="info")
            cmd = f"cd {shlex.quote(str(root))} && cargo build --quiet" if root else "cargo build --quiet"
        elif lang == "java":
            if shutil.which("mvn"):
                cmd = f"cd {shlex.quote(str(root))} && mvn -q compile" if root else "mvn -q compile"
            elif shutil.which("gradle"):
                cmd = f"cd {shlex.quote(str(root))} && gradle compileJava" if root else "gradle compileJava"
            else:
                return make_fact("module_import", True, "no java build tool; compile gate skipped",
                                 severity="info")
        else:
            return make_fact("module_import", True, f"no compile gate for {lang}",
                             severity="info")
        try:
            exec_shell = _get_exec_shell()
            out = exec_shell(cmd, timeout=240)
            if "exit code: 0" in out and ("error" not in out.lower() or "0 error" in out):
                return make_fact("module_import", True, f"{lang} build OK", payload={"cmd": cmd})
            return make_fact("module_import", False,
                             f"{lang} compile failed:\n{out[-1500:]}",
                             payload={"cmd": cmd, "output": out[-2000:]}, severity="error")
        except Exception as e:  # noqa: BLE001
            return make_fact("module_import", True, f"compile gate error skipped: {e}", severity="info")


@sensor("unit_test",
        "Run the detected stack's default test command and verify pass.",
        severity="error")
class UnitTestSensor(BaseSensor):
    def run(self, ctx: HarnessContext) -> SensorFact:
        try:
            exec_shell = _get_exec_shell()
            stack = ctx.stack if isinstance(ctx.stack, dict) else detect_stack()
            gen_tests, val_cmd, val_note = _build_validation(stack)
            if val_cmd:
                final = exec_shell(val_cmd, timeout=240)
            else:
                final = val_note
            missing_refs = _check_dangling_refs(stack["files"])
            if missing_refs:
                final += "\n" + "\n".join(missing_refs)
            all_pass, detail = _eval_validation(final, stack)
            if not all_pass or missing_refs:
                return make_fact("unit_test", False, detail,
                                 payload={"detail": detail, "output": final[-2000:]})
            return make_fact("unit_test", True, detail, payload={"detail": detail})
        except Exception as e:  # noqa: BLE001
            return make_fact("unit_test", False, f"unit test sensor error: {e}", severity="error")


@sensor("lsp_diagnostic",
        "Run language-server diagnostics via the LSP plugin if a server is configured; "
        "otherwise fall back to a pylsp/ruff availability note.",
        severity="warning")
class LspDiagnosticSensor(BaseSensor):
    # 扫描时跳过的重型/无关目录，避免拉起整库索引
    _SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
                  "dist", "build", ".mypy_cache", ".tox", ".idea",
                  ".pytest_cache", "site-packages", ".eggs", ".hg"}

    def run(self, ctx: HarnessContext) -> SensorFact:
        # 1) 优先使用 LSP 插件（swe_agent.lsp）：若已配置服务器，则真正跑诊断
        try:
            from . import lsp as _lsp
            mgr = _lsp.get_lsp()
            if mgr.is_configured():
                hits = self._scan_workspace(_lsp, ctx.workspace)
                if hits:
                    summary = "; ".join(
                        f"{p.name}: {d.strip().splitlines()[-1]}" for p, d in hits[:5]
                    )
                    return make_fact(
                        "lsp_diagnostic", True,
                        f"LSP plugin active; {len(hits)} file(s) with diagnostics. {summary}",
                        payload={"files": [str(p) for p, _ in hits], "count": len(hits)},
                        severity="warning",
                    )
                return make_fact("lsp_diagnostic", True,
                                 "LSP plugin active; no diagnostics on scanned source files.",
                                 severity="info")
        except Exception:  # noqa: BLE001
            pass  # 落到下面的兜底可用性探测

        # 2) 兜底：插件未配置时，沿用原可用性探测（LSP_CMD / pylsp / ruff）
        try:
            cmd = None
            if LSP_CMD:
                cmd = LSP_CMD
            elif shutil.which("pylsp"):
                cmd = ["pylsp"]
            elif shutil.which("ruff"):
                cmd = ["ruff", "--version"]
            if cmd:
                return make_fact("lsp_diagnostic", True,
                                 f"LSP/diagnostic tool available: {cmd}", severity="info")
            return make_fact("lsp_diagnostic", True,
                             "no LSP server available; diagnostic skipped", severity="info")
        except Exception as e:  # noqa: BLE001
            return make_fact("lsp_diagnostic", True, f"LSP check skipped: {e}", severity="info")

    @staticmethod
    def _scan_workspace(_lsp, workspace) -> List[Tuple[Path, str]]:
        """扫描工作区内源文件，对配置了语言服务器的文件调用 hint()，返回有诊断的 (path, diag)。"""
        ws = Path(workspace) if workspace else None
        if not ws or not ws.exists():
            return []
        hits: List[Tuple[Path, str]] = []
        scanned = 0
        for ext in (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java"):
            if scanned >= 40:  # 上限，避免整库索引过慢
                break
            for p in ws.rglob(f"*{ext}"):
                if any(part in LspDiagnosticSensor._SKIP_DIRS for part in p.parts):
                    continue
                if scanned >= 40:
                    break
                scanned += 1
                try:
                    d = _lsp.get_lsp().hint(str(p))
                except Exception:
                    d = ""
                if d and "LSP 诊断" in d:
                    hits.append((p, d))
        return hits


@sensor("language_cli",
        "Run pyright / go vet / eslint if available; otherwise note skip.",
        severity="warning")
class LanguageCliSensor(BaseSensor):
    def run(self, ctx: HarnessContext) -> SensorFact:
        try:
            from . import lsp as _lsp
            configured = set(getattr(_lsp, "PLUGIN_LSP_SERVERS", {}).keys())
            stack = ctx.stack if isinstance(ctx.stack, dict) else detect_stack()
            lang = stack.get("lang", "unknown")

            def _go_vet():
                root = _module_root("go")
                if shutil.which("go") and root:
                    return [f"cd {shlex.quote(str(root))} && go vet ./..."]
                return []
            def _cargo_clippy():
                return ["cargo clippy --quiet"] if shutil.which("cargo") else []
            def _eslint():
                return ["eslint . --max-warnings=-1"] if shutil.which("eslint") else []
            def _pyright():
                return ["pyright --version"] if shutil.which("pyright") else []
            def _java_check():
                root = _module_root("java")
                if shutil.which("mvn") and root:
                    return [f"cd {shlex.quote(str(root))} && mvn -q test-compile"]
                if shutil.which("gradle") and root:
                    return [f"cd {shlex.quote(str(root))} && gradle compileTestJava"]
                return []

            # 插件声明的语言 -> 对应 linter；实现「插件驱动」而非硬编码列表
            lint_map = {
                "python": _pyright,
                "go": _go_vet,
                "rust": _cargo_clippy,
                "typescript": _eslint, "typescriptreact": _eslint,
                "javascript": _eslint, "javascriptreact": _eslint,
                "java": _java_check,
            }
            # 只对本工作区实际存在的语言 + 当前技术栈运行 linter，
            # 避免「插件声明了 rust」却在纯 Go 工程里误跑 cargo clippy
            present = _present_langs(WORKSPACE)
            checks: List[str] = []
            seen = set()
            for l in list(configured) + [lang]:
                if l in seen:
                    continue
                seen.add(l)
                if l != lang and l not in present:
                    continue
                fn = lint_map.get(l)
                if fn:
                    for c in fn():
                        if c not in checks:
                            checks.append(c)
            if not checks:
                return make_fact("language_cli", True,
                                 f"no language CLI available (plugin servers: {sorted(configured)}); skip",
                                 severity="info")
            results: List[str] = []
            for c in checks:
                try:
                    p = subprocess.run(c, shell=True, capture_output=True, text=True,
                                       timeout=120, cwd=str(WORKSPACE))
                    results.append(f"{c} -> rc={p.returncode}")
                except Exception as e:  # noqa: BLE001
                    results.append(f"{c} -> error {e}")
            return make_fact("language_cli", True, "; ".join(results),
                             severity="info", payload={"checks": checks})
        except Exception as e:  # noqa: BLE001
            return make_fact("language_cli", True, f"language CLI check skipped: {e}", severity="info")


# ======================================================================
# 流水线入口
# ======================================================================
def run_harness_pipeline(workspace=None, plan=None, stack=None) -> List[SensorFact]:
    """构建 HarnessContext 并跑全量传感器，返回标准化事实集合。"""
    ws = Path(workspace) if workspace is not None else WORKSPACE
    if plan is None:
        plan = GLOBAL_STATE.get("plan")
    ctx = HarnessContext(workspace=ws, plan=plan, stack=stack)
    return SensorRegistry.run_pipeline(ctx)


__all__ = [
    "_eval_pytest_result", "_is_js_test", "_try_run_js_tests", "_js_bin_available",
    "detect_stack", "_check_dangling_refs", "_build_validation", "_eval_validation",
    "_detect_test_stack_mismatch", "_validation_error_signature", "_try_package_selfheal",
    "_is_test_cmd", "_new_subagent_fence", "_run_test_bar",
    "FileDriftSensor", "ModuleImportSensor", "UnitTestSensor",
    "LspDiagnosticSensor", "LanguageCliSensor",
    "run_harness_pipeline",
]
