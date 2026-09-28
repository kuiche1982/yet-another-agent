import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """验证 harness sensor 管线已语言无关化（go/rust/java 不再绑定 Python）。

    在 examples/game_of_life_go 工作区：
      - detect_stack 识别为 go
      - unit_test 触发 go build / go test
      - module_import 走 go build 编译门禁
      - file_drift 不误报 *_test.go / *Test.java
      - language_cli 由插件声明的语言驱动（含 go vet）
    """
    import os
    import sys
    import shutil
    import tempfile
    from pathlib import Path

    REPO = Path(__file__).resolve().parent.parent
    EX = REPO / "examples" / "game_of_life_go"
    # 注入隔离 WORKSPACE：把 go 示例拷到临时目录，绝不把全局 WORKSPACE 钉在仓库子树。
    # 此前直接指向 examples/game_of_life_go（仓库内子目录），会泄漏全局状态、且一旦
    # _flatten_workspace 在仓库路径下触发即搬空工程。测试应「注入」而非「继承」默认值。
    WS = Path(tempfile.mkdtemp(prefix="go_ws_")) / "game_of_life_go"
    shutil.copytree(EX, WS)

    sys.path.insert(0, str(REPO))
    import swe_agent.config as C
    import swe_agent.harness as H
    import swe_agent.plugins as P

    # 切到隔离的 golang 工作区 WS（临时副本）；务必在 finally 还原全局 WORKSPACE /
    # GLOBAL_STATE，避免泄漏到后续测试或 e2e。
    saved_hw = H.WORKSPACE
    saved_cw = C.WORKSPACE
    saved_gf = H.GLOBAL_STATE.get("generated_files")
    saved_plan = H.GLOBAL_STATE.get("plan")
    H.WORKSPACE = WS
    C.WORKSPACE = WS
    H.GLOBAL_STATE["generated_files"] = []
    H.GLOBAL_STATE["plan"] = {}
    try:
        # 加载插件（lsp 插件会填充 PLUGIN_LSP_SERVERS，使 lsp_diagnostic 走插件路径）
        P.load_plugins(C.PLUGINS_ROOT, enable_mcp=False)

        passed = 0
        failed = 0


        def check(name, ok, detail=""):
            nonlocal passed, failed
            if ok:
                passed += 1
                print(f"[PASS] {name}")
            else:
                failed += 1
                print(f"[FAIL] {name}  -> {detail}")


        # ---------- 1) detect_stack 识别 go ----------
        st = H.detect_stack()
        check("detect_stack 识别为 go", st["lang"] == "go", f"lang={st['lang']}")
        check("go 模块根解析正确", str(H._module_root("go")) == str(WS), str(H._module_root("go")))

        # ---------- 2) file_drift 正则补全 ----------
        import swe_agent.config as Cfg
        check("正则匹配 *_test.go", bool(Cfg._TEST_FILE_RE.search("pkg/main_test.go")), "go test 未被识别")
        check("正则匹配 *Test.java", bool(Cfg._TEST_FILE_RE.search("pkg/FooTest.java")), "java 测试未被识别")
        check("正则不匹配 main.go", not Cfg._TEST_FILE_RE.search("pkg/main.go"), "误匹配普通源文件")
        check("正则仍匹配 test_*.py", bool(Cfg._TEST_FILE_RE.search("pkg/test_foo.py")), "回归：py 测试未识别")

        # ---------- 3) file_drift 端到端：不误报测试文件 ----------
        plan = {"modules": [{"path": "main.go"}]}
        H.GLOBAL_STATE["plan"] = plan
        try:
            (WS / "main_test.go").write_text(
                "package main\nimport \"testing\"\nfunc TestNothing(t *testing.T) {}\n", encoding="utf-8")
            (WS / "extra_unplanned.go").write_text(
                "package main\n// 故意未计划文件\n", encoding="utf-8")
            ctx = H.HarnessContext(workspace=WS, plan=plan, stack=None)
            fact = H.FileDriftSensor().run(ctx)
            drift = fact.payload.get("drift", [])
            check("file_drift 不误报 main_test.go", "main_test.go" not in drift, f"drift={drift}")
            check("file_drift 抓到 extra_unplanned.go", "extra_unplanned.go" in drift, f"drift={drift}")
        finally:
            for f in ("main_test.go", "extra_unplanned.go"):
                p = WS / f
                if p.exists():
                    p.unlink()

        # ---------- 4) 全量管线：unit_test / module_import / language_cli ----------
        facts = {f.sensor_name: f for f in H.run_harness_pipeline(workspace=WS)}
        print("  --- 管线事实 ---")
        for n, f in facts.items():
            print(f"    {n}: ok={f.ok} sev={f.severity} :: {f.message[:90]}")

        ut = facts.get("unit_test")
        check("unit_test 存在且非 False", ut is not None and ut.ok is not False,
              ut.message if ut else "缺失")
        check("unit_test 走了 go 命令", ut is not None and ("go build" in ut.message or "go test" in ut.message),
              ut.message if ut else "")

        mi = facts.get("module_import")
        # 无 Python 模块 -> 走 go build 编译门禁（ok 应为 True，因代码可编译）
        check("module_import 对 go 走编译门禁", mi is not None and mi.ok is True,
              mi.message if mi else "缺失")

        lc = facts.get("language_cli")
        lc_checks = (lc.payload or {}).get("checks", [])
        check("language_cli 由插件驱动含 go vet", any("go vet" in c for c in lc_checks),
              f"checks={lc_checks}")

        ld = facts.get("lsp_diagnostic")
        check("lsp_diagnostic 走插件路径(非 fallback)", ld is not None and ld.message.startswith("LSP plugin active"),
              ld.message if ld else "缺失")

        # ---------- 5) go test 判读路径（临时加一个测试文件）----------
        try:
            (WS / "foo_test.go").write_text(
                "package main\nimport \"testing\"\n"
                "func TestAdd(t *testing.T) { if 1+1 != 2 { t.Fatal(\"math broken\") } }\n",
                encoding="utf-8")
            st2 = H.detect_stack()
            gen, cmd, note = H._build_validation(st2)
            check("_build_validation 生成 go test 命令", "go test" in cmd, f"cmd={cmd}")
            exec_shell = H._get_exec_shell()
            out = exec_shell(cmd, timeout=240)
            ok, detail = H._eval_validation(out, st2)
            check("go test 判读通过(True)", ok is True, f"detail={detail} | out={out[-200:]}")
        finally:
            p = WS / "foo_test.go"
            if p.exists():
                p.unlink()

        print(f"\n结果: {passed} 通过 / {failed} 失败")
        assert failed == 0, f"{failed} checks failed"
    finally:
        H.WORKSPACE = saved_hw
        C.WORKSPACE = saved_cw
        if saved_gf is None:
            H.GLOBAL_STATE.pop("generated_files", None)
        else:
            H.GLOBAL_STATE["generated_files"] = saved_gf
        if saved_plan is None:
            H.GLOBAL_STATE.pop("plan", None)
        else:
            H.GLOBAL_STATE["plan"] = saved_plan
        # 清理注入的隔离 WORKSPACE（临时目录），绝不残留仓库路径
        shutil.rmtree(WS.parent, ignore_errors=True)


if __name__ == "__main__":
    main()


def test_main():
    main()
