#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # -*- coding: utf-8 -*-
    """验证：
    1) config 从仓库根 .env 加载 GLM_API_TOKEN（不再硬编码）；
    2) 三个插件（web_search / web_fetch / ocr）在 load_plugins 后均已注册；
    3) 每个插件发起请求时都携带 Authorization: Bearer <token>，
       即「插件能拿到 token 并发起授权 API 调用」。
    为去网络依赖，用 monkeypatch 截获 requests.post 的 header 来证明鉴权头构造正确；
    最后再 best-effort 做一次真实 web_fetch 调用（仅打印，不强制）。
    """
    import sys
    import requests

    import swe_agent.config as C
    import swe_agent.plugins as P
    from swe_agent.registry import ToolRegistry

    failures = []


    def check(cond, msg):
        if cond:
            print(f"  [OK] {msg}")
        else:
            print(f"  [FAIL] {msg}")
            failures.append(msg)


    # ---------- 1) .env 加载 ----------
    print("== 1) .env 加载 ==")
    token = C.GLM_API_TOKEN
    check(bool(token), "GLM_API_TOKEN 非空（.env 已注入）")
    env_file = C.REPO_ROOT / ".env"
    check(env_file.exists(), f".env 存在于 {env_file}")
    if env_file.exists():
        env_text = env_file.read_text(encoding="utf-8")
        check(f"GLM_API_TOKEN={token}" in env_text, ".env 中的 token 与 config 取到的一致")
    print(f"  (token 长度 {len(token)}，已脱敏不打印明文)")


    # ---------- 2) 插件注册 ----------
    print("\n== 2) 插件加载 + 工具注册 ==")
    state = P.load_plugins(C.PLUGINS_ROOT, enable_mcp=False)
    print(f"  load_plugins 返回：{state['plugins']} / 工具 {state['tools']}")
    for name in ("web_search", "web_fetch", "ocr"):
        check(name in ToolRegistry.names(), f"工具 {name} 已注册")


    # ---------- 3) 鉴权头构造（monkeypatch） ----------
    print("\n== 3) 插件发起请求携带 Bearer 授权头 ==")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["auth"] = (kwargs.get("headers") or {}).get("Authorization")
        class R:
            status_code = 200
            def json(self):
                if "reader" in url:
                    return {"reader_result": {"title": "T", "content": "hello", "url": url}}
                if "files/ocr" in url:
                    return {"words_result": [{"words": "abc", "location": {"left": 1, "top": 2, "width": 3, "height": 4}}], "message": "success"}
                return {"search_result": [{"title": "x", "link": "https://x", "content": "y"}]}
        return R()

    orig = requests.post
    requests.post = fake_post
    try:
        # 造一张占位图片给 ocr 用
        import os
        dummy = "/tmp/_ocr_verify_dummy.png"
        with open(dummy, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # 仅占位，monkeypatch 不会真正发送

        for action, params in [
            ("web_search", {"query": "python pathlib"}),
            ("web_fetch", {"url": "https://example.com"}),
            ("ocr", {"image_path": dummy}),
        ]:
            captured.clear()
            res = ToolRegistry.dispatch({"action": action, **params})
            check(captured.get("auth") == f"Bearer {token}",
                  f"{action} -> 请求 {captured.get('url')} 携带 Authorization: Bearer <token>")
    finally:
        requests.post = orig


    # ---------- 4) best-effort 真实调用 ----------
    print("\n== 4) best-effort 真实调用（仅打印，不强制） ==")
    try:
        real = ToolRegistry.dispatch({"action": "web_fetch", "url": "https://example.com"})
        first = real.splitlines()[0] if real else ""
        print(f"  真实 web_fetch 首行：{first[:80]}")
    except Exception as e:
        print(f"  真实调用未执行（可能为网络/配额限制，不影响上面 1-3 的结论）：{e}")


    print("\n==== 结论 ====")
    if failures:
        print(f"存在 {len(failures)} 项失败：")
        for f in failures:
            print(f"  - {f}")
        assert False
    print("✅ 通过：token 由 .env 注入，web_search / web_fetch / ocr 三插件均能拿到 token 并构造 Bearer 授权请求。")


if __name__ == "__main__":
    main()


def test_main():
    main()
