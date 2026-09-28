#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # -*- coding: utf-8 -*-
    """单元测试：ToolRegistry 的 dispatch + 可拼装钩子（before/after/on_success/on_fail）+ 全局钩子。

    不依赖任何 LLM / 网络：纯本地驱动 dispatch，验证
      - 动作按名称路由到对应实现；
      - before 钩子可拦截（短路），且全局 before 仍先执行（计数器）；
      - after 钩子可覆盖结果；
      - 成功/失败分支正确触发 on_success / on_fail；
      - 未知动作安全回退；
      - ctx 透传给接受 ctx 的工具；多余参数按签名过滤（含 **kwargs 全收）。
    """
    from swe_agent.registry import ToolRegistry, tool, ActionContext

    # 本测试不依赖 supervisor，但为隔离其它模块可能注册的全局钩子，先清空。
    ToolRegistry.clear_global_hooks()

    calls: list = []


    def reset():
        calls.clear()


    @tool(name="t_echo", description="回声工具（测试用）", category="fs",
          schema={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]})
    def t_echo(ctx, msg=""):
        calls.append(("run", msg))
        return f"ECHO:{msg}"


    @tool(name="t_bad", description="恒失败工具（测试用）", category="fs",
          schema={"type": "object", "properties": {}})
    def t_bad(ctx):
        return "error: boom"


    @tool(name="t_ctx", description="回显 ctx.action（测试用）", category="meta",
          schema={"type": "object", "properties": {}})
    def t_ctx(ctx):
        return f"ctx.action={ctx.action}"


    @tool(name="t_strict", description="严格参数工具（测试用）", category="fs",
          schema={"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]})
    def t_strict(ctx, a):
        return f"A={a}"


    @tool(name="t_kw", description="kwargs 工具（测试用）", category="fs",
          schema={"type": "object", "properties": {"a": {"type": "string"}}})
    def t_kw(ctx, **kw):
        return f"kw={kw}"


    # ---- 钩子 ----
    def block_hook(ctx, params, result):
        if params.get("msg") == "BLOCK":
            return "blocked_by_hook"
        return None


    def after_hook(ctx, params, result):
        return result + "|AFTER"


    def on_success_hook(ctx, params, result):
        calls.append(("success", result))
        return None


    def on_fail_hook(ctx, params, result):
        calls.append(("fail", result))
        return None


    def counter_hook(ctx, params, result):
        calls.append(("count", ctx.action))
        return None


    ToolRegistry.get("t_echo").before.append(block_hook)
    ToolRegistry.get("t_echo").after.append(after_hook)
    ToolRegistry.get("t_echo").on_success.append(on_success_hook)
    ToolRegistry.get("t_echo").on_fail.append(on_fail_hook)
    ToolRegistry.get("t_bad").on_fail.append(on_fail_hook)
    ToolRegistry.register_global_hook("before", counter_hook)


    fails: list = []


    def check(name, ok, detail=""):
        print(("✅" if ok else "❌"), name, detail)
        if not ok:
            fails.append(name)


    # 1) 正常分发：run + after + on_success + 计数
    reset()
    r = ToolRegistry.dispatch({"action": "t_echo", "msg": "hi"})
    check("dispatch 路由到正确实现", r == "ECHO:hi|AFTER", repr(r))
    check("on_success 触发", ("success", "ECHO:hi|AFTER") in calls)
    check("全局 before(计数) 触发", ("count", "t_echo") in calls)
    check("工具本体执行", ("run", "hi") in calls)

    # 2) before 拦截：被拦截前全局 before(计数) 仍先跑，工具本体不执行
    reset()
    r = ToolRegistry.dispatch({"action": "t_echo", "msg": "BLOCK"})
    check("before 钩子可拦截短路", r == "blocked_by_hook", repr(r))
    check("拦截时工具本体不执行", ("run", "BLOCK") not in calls)
    check("拦截前全局 before 仍执行(计数)", ("count", "t_echo") in calls)

    # 3) on_fail：工具返回失败前缀时触发
    reset()
    r = ToolRegistry.dispatch({"action": "t_bad"})
    check("失败动作返回原结果", r == "error: boom", repr(r))
    check("on_fail 触发", ("fail", "error: boom") in calls)
    check("失败时 on_success 不触发", not any(c[0] == "success" for c in calls))

    # 4) 未知动作安全回退
    reset()
    r = ToolRegistry.dispatch({"action": "nope"})
    check("未知动作回退为 error", r.startswith("error: 未知动作"), repr(r))

    # 5) ctx 透传
    reset()
    r = ToolRegistry.dispatch({"action": "t_ctx"})
    check("ctx.action 透传给工具", r == "ctx.action=t_ctx", repr(r))

    # 6) 参数按签名过滤（严格参数，多余 key 被忽略）
    reset()
    r = ToolRegistry.dispatch({"action": "t_strict", "a": "1", "b": "extra"})
    check("多余参数被过滤(严格签名)", r == "A=1", repr(r))

    # 7) **kwargs 工具收下全部参数
    reset()
    r = ToolRegistry.dispatch({"action": "t_kw", "a": "1", "b": "2"})
    check("**kwargs 收下全部参数", r == "kw={'a': '1', 'b': '2'}", repr(r))

    # 8) 钩子不影响未被挂载的工具（t_bad 未挂 after，原样返回）
    reset()
    r = ToolRegistry.dispatch({"action": "t_bad"})
    check("无 after 钩子时结果不变", r == "error: boom", repr(r))


    print()
    if fails:
        print(f"❌ 失败 {len(fails)} 项：{fails}")
        raise SystemExit(1)
    print("✅ 全部通过（dispatch / before 拦截 / after 覆盖 / on_success / on_fail / 未知回退 / ctx 透传 / 参数过滤）")


if __name__ == "__main__":
    main()


def test_main():
    main()
