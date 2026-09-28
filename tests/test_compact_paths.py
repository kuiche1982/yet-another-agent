import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

# 压缩逻辑已迁到独立模块：
#   config.CONTEXT_WINDOW / AUTO_COMPACT_ENABLED / SYSTEM_PROMPT -> swe_agent.config
#   compact.estimate_tokens / maybe_auto_compact / get_auto_compact_threshold / _auto_failures
#   supervisor.execute_action (compact 动作为 meta 类动作)
from swe_agent import config as C, compact
from swe_agent.supervisor import execute_action


def _llm_alive(host="127.0.0.1", port=8000, timeout=1.0):
    """快速探测本地 Executor 服务是否在线（maybe_auto_compact / compact 动作需要真实模型）。"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    except Exception:
        return False
    finally:
        s.close()


def main():
    """验证 自动(auto) 与 手动(manual/compact 动作) 两条压缩路径。"""
    code_blob = "class Foo:\n    def bar(self, x):\n        # 填充行\n        return x + 1\n" * 50

    def big_messages():
        msgs = [{"role": "system", "content": C.SYSTEM_PROMPT}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"步骤{i}: 实现功能\n{code_blob}"})
            msgs.append({"role": "assistant", "content": f'{{"action":"write_file","path":"f{i}.py","content":"{code_blob}"}}'})
        return msgs

    # 保存并恢复被改写的全局配置/状态，避免污染其它测试
    saved_cw = C.CONTEXT_WINDOW
    saved_ace = C.AUTO_COMPACT_ENABLED
    saved_af = compact._auto_failures
    try:
        # ---- 1) 自动路径：maybe_auto_compact 在超过阈值时压缩 ----
        print("===== 自动压缩 (auto) =====")
        C.CONTEXT_WINDOW = 20000  # 阈值 = 20000 - 13000 = 7000
        msgs = big_messages()
        compact._auto_failures = 0
        tok_before = compact.estimate_tokens(msgs)
        print("压缩前估算 tokens ≈", tok_before, "| 阈值:", compact.get_auto_compact_threshold())
        if not _llm_alive():
            pytest.skip("需要本地 LLM 服务（127.0.0.1:8000）；estimate_tokens 阈值逻辑可静态验证，"
                        "但 maybe_auto_compact 需真实压缩。")
        out = compact.maybe_auto_compact(msgs)
        tok_after = compact.estimate_tokens(out)
        print("压缩后估算 tokens ≈", tok_after, "| 节省:", tok_before - tok_after)
        assert tok_after < tok_before, "自动压缩应减小上下文"
        assert any("对话摘要" in m.get("content", "") for m in out), "自动压缩后应含摘要"
        print("✅ 自动路径 OK\n")

        # ---- 2) 手动路径：模型输出 compact 动作就地压缩 ----
        print("===== 手动压缩 (compact 动作) =====")
        msgs2 = big_messages()
        C.AUTO_COMPACT_ENABLED = False  # 关掉自动，单独验证手动
        res = execute_action({"action": "compact", "instructions": "保留 f0.py 的结构"}, msgs2)
        print("动作返回:", res)
        print("压缩后消息数:", len(msgs2), "| 首条角色:", msgs2[0]["role"])
        assert len(msgs2) == 2
        assert "compact_success" in res
        assert any("对话摘要" in m.get("content", "") for m in msgs2)
        print("✅ 手动路径 OK")
    finally:
        C.CONTEXT_WINDOW = saved_cw
        C.AUTO_COMPACT_ENABLED = saved_ace
        compact._auto_failures = saved_af


if __name__ == "__main__":
    main()


def test_main():
    main()
