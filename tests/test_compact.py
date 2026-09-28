import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

# 压缩逻辑已迁到 swe_agent.compact：estimate_tokens / compact_conversation
from swe_agent.compact import estimate_tokens, compact_conversation


def _llm_alive(host="127.0.0.1", port=8000, timeout=1.0):
    """快速探测本地 Executor 服务是否在线（compact_conversation 需要真实模型）。"""
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
    # 1) 估算函数 sanity（纯启发式，无需模型，始终可测）
    short = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "世界"}]
    n_short = estimate_tokens(short)
    print("[estimate] 短对话 tokens ≈", n_short)

    long_text = "def f(x):\n    return x*2\n" * 200
    big = [{"role": "system", "content": "你是助手"},
           {"role": "user", "content": long_text},
           {"role": "assistant", "content": long_text}]
    n_big = estimate_tokens(big)
    print("[estimate] 长对话(含大段代码) tokens ≈", n_big)
    assert n_big > n_short * 3, "估算应随内容增长"

    # 2) 真实压缩：需要本地 LLM 服务（127.0.0.1:8000）
    if not _llm_alive():
        pytest.skip("需要本地 LLM 服务（127.0.0.1:8000）才能跑 compact_conversation；"
                    "estimate_tokens 已断言通过。")

    code_blob = "def f(x):\n    # 注释行用于填充上下文\n    return x * 2 + 1\n" * 60
    conv = [{"role": "system", "content": "你是一个代码助手"}]
    for i in range(8):
        conv.append({"role": "user", "content": f"请实现第 {i} 个功能：\n{code_blob}"})
        conv.append({"role": "assistant", "content": f'{{"action":"write_file","path":"m{i}.py","content":"{code_blob}"}}'})
        conv.append({"role": "user", "content": f"【工具执行结果】\nwrite_success: m{i}.py ({len(code_blob)} 字节)"})
    pre = estimate_tokens(conv)
    print("[compact] 压缩前 tokens ≈", pre)
    new_msgs, stats = compact_conversation(conv, is_auto=False)
    post = estimate_tokens(new_msgs)
    print("[compact] stats:", stats)
    print("[compact] 压缩后消息数:", len(new_msgs), "首条角色:", new_msgs[0]["role"])
    print("[compact] 摘要片段:\n", new_msgs[-1]["content"][:300])
    assert new_msgs[0]["role"] == "system"
    assert len(new_msgs) == 2  # system + 摘要
    assert "对话摘要" in new_msgs[-1]["content"]
    assert post < pre, "压缩后应更短"
    print("\n✅ compact 工具测试通过")


if __name__ == "__main__":
    main()


def test_main():
    main()
