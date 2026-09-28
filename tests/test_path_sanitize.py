"""实测：文件工具的报错回执不得泄漏真实绝对 WORKSPACE 路径。

根因（2026-09-02）：模型 monologue 出现 `/abs/.../agent_sandbox/.../conway/src/life.py`，
追查为 harness 工具报错分支的 Python 异常消息内嵌了 `str(WORKSPACE)` 绝对前缀，
直接回灌后弱模型照抄成绝对路径、触发 cwd thrash。修复：所有用户可见报错经
`_sanitize` 把 `str(WORKSPACE)` 替换为相对锚点 `./`。本测试锁定该不变量。

为避免真实磁盘/brokered 沙箱限制，全程用假 WORKSPACE + stub Path 的 FS 方法，
让异常消息精确复现"内嵌绝对路径"的泄漏场景（纯字符串操作，无真实 IO）。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import swe_agent.tools as T  # noqa: E402

FAKE_WS = Path("/abs/ws")


def _raise_not_found(self, *a, **k):
    # 复现 Python FileNotFoundError 消息内嵌绝对路径：str(e) 含 str(self)
    raise FileNotFoundError(2, "No such file or directory", str(self))


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(T, "WORKSPACE", FAKE_WS)
    monkeypatch.setattr(T, "GLOBAL_STATE", {"generated_files": []})
    # 所有 FS 操作改为抛 FileNotFoundError（消息内嵌绝对路径），
    # 既避开真实磁盘/沙箱限制，又精确复现"异常消息含 str(WORKSPACE)"的泄漏场景
    monkeypatch.setattr(T.Path, "read_text", _raise_not_found)
    monkeypatch.setattr(T.Path, "write_text", _raise_not_found)
    monkeypatch.setattr(T.Path, "mkdir", lambda self, *a, **k: None)
    yield


def test_sanitize_replaces_workspace_prefix():
    msg = f"boom at {T.WORKSPACE}/src/x.py: no such file"
    assert T._sanitize(msg) == "boom at ./src/x.py: no such file"
    assert str(T.WORKSPACE) not in T._sanitize(msg)


def test_read_file_error_sanitizes(patched):
    out = T.read_file("nonexistent.py")
    assert str(FAKE_WS) not in out, f"泄漏了绝对路径: {out}"
    assert "./" in out
    assert out.startswith("read_error:")


def test_edit_file_read_error_sanitizes(patched):
    # old_string != "" 走「精确替换」分支，文件不存在 → 无法读取分支
    out = T.edit_file("nonexistent.py", old_string="x", new_string="y")
    assert str(FAKE_WS) not in out, f"泄漏了绝对路径: {out}"
    assert "./" in out
    assert out.startswith("edit_error:")


def test_write_verify_error_sanitizes(patched):
    err = T._write_verify(FAKE_WS / "sub" / "file.py", "content")
    assert err is not None, "应当返回错误说明"
    assert str(FAKE_WS) not in err, f"泄漏了绝对路径: {err}"
    assert "./" in err


def test_write_file_error_sanitizes(patched):
    # write_file 报错回执不得泄漏真实绝对 WORKSPACE 路径。
    # 注：单次写入行数限制已迁出门控 guard（BEFORE_TOOL_CALL 的 _write_size_guard），
    # 此处仅验证 write_error 包装层把绝对路径脱敏为相对锚点 ./（与 read/edit 测试同一不变量）。
    out = T.write_file("src/x.py", "def f():\n    pass\n")
    assert str(FAKE_WS) not in out, f"泄漏了绝对路径: {out}"
    assert "./" in out
    assert out.startswith("write_error:")
