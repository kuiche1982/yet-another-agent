"""单杠校验（_run_test_bar）的判据必须来自「真实执行结果」，不得被静态启发式误杀。

背景（2026-09-08 实锤）：
    _run_test_bar 曾有一层 AST 审计 `_audit_tests`（纯 stdlib ast 反作弊，拦空测试 /
    assert True）。它只扫 `tree.body` 顶层找 `def test_*`，看不见 `class TestXxx` 里的
    方法 → 工作区里 pytest 实跑 4 passed / exit 0 的 `test_life.py`（全为类方法）被判
    `no_test_functions`，连续 2 次触 VAL_DOOMED_THRESHOLD 判 unsolvable。
    logs/harness.log 里有 6 次同类误杀记录。

    fix：整层删除 `_audit_tests`。单杠闸门只认执行事实（exit code + passed/failed/error
    计数），测试「质量」交给独立 tester 验收。本文件锁定该不变量。

这些用例会真实起 pytest 子进程（秒级），不是 mock —— 判据层必须被真实执行验证。
"""

import textwrap

from swe_agent import harness
from swe_agent import tools as _tools


def _patch_ws(tmp_path, monkeypatch):
    """把 harness 与 tools 的 WORKSPACE 都指向 tmp 工作区（两处是独立绑定，都要打）。"""
    monkeypatch.setattr(harness, "WORKSPACE", tmp_path, raising=True)
    monkeypatch.setattr(_tools, "WORKSPACE", tmp_path, raising=True)


def test_class_based_tests_pass_the_bar(tmp_path, monkeypatch):
    """回归守卫：纯类方法（TestXxx::test_*）的测试文件，pytest 全绿就必须判 pass。

    这是 2026-09-08 的直接回归点：旧 AST 审计看不见类内方法 → 误判 no_test_functions。
    """
    (tmp_path / "test_cls.py").write_text(textwrap.dedent("""
        class TestAdd:
            def test_one(self):
                assert 1 + 1 == 2

            def test_two(self):
                assert 2 + 2 == 4
    """).strip() + "\n", encoding="utf-8")
    _patch_ws(tmp_path, monkeypatch)

    status, detail, out = harness._run_test_bar([])

    assert status == "pass", f"类方法测试被误判：status={status} detail={detail} out={out}"
    assert "2 passed" in detail, detail


def test_module_level_tests_still_pass(tmp_path, monkeypatch):
    """基线：模块级 def test_* 同样判 pass（删审计不得连正常路径一起退化）。"""
    (tmp_path / "test_fn.py").write_text(textwrap.dedent("""
        def test_one():
            assert 1 + 1 == 2

        def test_two():
            assert 3 * 3 == 9
    """).strip() + "\n", encoding="utf-8")
    _patch_ws(tmp_path, monkeypatch)

    status, detail, out = harness._run_test_bar([])

    assert status == "pass", f"status={status} detail={detail} out={out}"


def test_failing_tests_still_block(tmp_path, monkeypatch):
    """反向守卫：删掉审计不等于「什么都放行」——真失败的测试必须判 fail。"""
    (tmp_path / "test_bad.py").write_text(textwrap.dedent("""
        def test_broken():
            assert 1 + 1 == 3
    """).strip() + "\n", encoding="utf-8")
    _patch_ws(tmp_path, monkeypatch)

    status, detail, out = harness._run_test_bar([])

    assert status == "fail", f"失败测试被放过：status={status} detail={detail}"
    assert "1 failed" in detail, detail


def test_collect_nothing_is_not_pass(tmp_path, monkeypatch):
    """反向守卫：有 test_*.py 但收不到任何用例（no tests ran）不得判绿。

    count>0 由 pytest 自己的 exit code / 计数表达，不需要 AST 启发式补位。
    """
    (tmp_path / "test_empty.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    _patch_ws(tmp_path, monkeypatch)

    status, detail, out = harness._run_test_bar([])

    assert status != "pass", f"空测试文件被判绿：detail={detail} out={out}"


def test_no_ast_audit_layer_exists():
    """防回归：AST 启发式闸门不得复活。

    任何「不看执行结果、只靠静态解析判失败」的闸门都应走 tester 独立验收，
    不允许回到 _run_test_bar 里当阻断条件。
    """
    assert not hasattr(harness, "_audit_tests"), (
        "harness._audit_tests 不应存在：单杠判据只能是真实执行结果，"
        "静态 AST 启发式会误杀（类方法测试 4 passed 仍判 no_test_functions）。")
