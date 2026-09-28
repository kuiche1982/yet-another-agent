#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/lfm_sidecar.py —— LFM 副驾：上下文管理 + 大内容压缩（第 2 样交付物）

设计定位（对齐 forge 教训 + 用户 3 大交付物）：
- 它是【无状态】的小模型副驾，只用 liquid/lfm2.5-1.2b（本机 LM Studio），
  绝不作为任何主 agent 角色（主角色全部走 qwen）。
- 两个职责：
    1) 上下文管理：对话自动/手动压缩（委托 compact.py，但强制 SIDECAR_MODEL，
       不再误用 qwen 主角色）——参考 claude-code autoCompact，超限时把长对话压成摘要。
    2) 大内容压缩：工具结果 / 文件读取落地进主上下文之前，先摘要，避免噪声撑爆上下文。
- 【失败一律 fail-open】：任何 LFM 调用失败都保留原文，绝不静默丢数据。

这是「把约束下沉到 BUILD 层、模型只做推理」原则的体现：压缩是工程能力，不是 prompt 技巧。
"""

import re

import swe_agent.compact as _compact
import swe_agent.config as C
import swe_agent.models as M
from swe_agent.log import logger

# —— 抽取式截断（保真、确定性，弱模型友好）——

# 源码特征：含多个 def/class 定义（但排除 pytest 输出，后者也会打印测试源码行）
_SRC_RE = re.compile(r'(?:^|\n)\s*(?:def|class)\s+[A-Za-z_]\w*\s*\(')
_PYTEST_RE = re.compile(r'(?:collected\s+\d+|test session starts|passed|failed in|\bFAILED\b)',
                        re.MULTILINE)
# 忠实闸门：原文中的关键 token 必须出现在摘要里，否则视为不可信摘要
_KEY_TOKEN_RE = re.compile(r'([A-Za-z_][\w./-]*\.py|\b\w+Error\b|\b\w+Exception\b|Traceback|AssertionError)',
                           re.MULTILINE)


def _is_source_like(text: str) -> bool:
    """粗略判断是否为源码工件（executor 正在编辑的文件）。源码绝不压缩——executor 需基于真实代码自修。"""
    has_defs = len(_SRC_RE.findall(text)) >= 2
    looks_like_test = bool(_PYTEST_RE.search(text))
    return has_defs and not looks_like_test


def _extractive_truncate(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail + 80:
        return text
    omitted = len(text) - head - tail
    return text[:head] + f"\n...[中间省略 {omitted} 字符]...\n" + text[-tail:]


def _key_tokens(text: str):
    return set(_KEY_TOKEN_RE.findall(text))


def _faithful(summary: str, original: str) -> bool:
    """忠实闸门：摘要必须保留原文关键 token（.py 路径 / Error / Exception / Traceback）。"""
    if not summary:
        return False
    need = _key_tokens(original)
    if not need:
        return True  # 原文无关键 token（纯散文），放行
    have = _key_tokens(summary)
    return need <= have  # 所有关键 token 都在摘要里


# LFM 压缩副驾的系统提示：针对「单条工具/命令输出」的【逐字抽取】任务，
# 显式界定 KEEP（保留什么）/ DROP（丢弃什么）/ OUTPUT（输出格式）。
#
# ⚠️ 对象区分（避免误用 Claude-cli 的 compact prompt）：
#   - Claude 的 getCompactPrompt 对象是【整段对话 messages】（多轮 user/assistant/tool），
#     产出【允许有损】的语义摘要，消费方是强模型、对 paraphrase 鲁棒。
#   - 本副驾 compress_content 的对象是【单条工具输出 blob】（如 pytest 失败 dump），
#     executor 要的是 test_x.py:23、断言消息这类【原样字符串】去自修 bug，
#     一旦改写/有损即丢信息。故二者 prompt 不可互借——Claude 那套 <analysis>/STEP
#     结构化是为「对话摘要」设计的，拿单条错误信息喂进去属于对象错配（翻车非 STEP 之过）。
#   这里用平铺 KEEP/DROP 列表（匹配我们的抽取对象，弱模型能稳定产出【部分】锚点），
#   标 [REQUIRED] 强制项；真正的保真由 BUILD 层保证：忠实闸门 _faithful + 生产默认
#   extractive（不调模型）。
_SIDECAR_SYSTEM = (
    "You are a stateless compression copilot. Given a tool/shell output, extract and reproduce "
    "VERBATIM only the spans a developer needs to fix a failure. Copy EXACTLY (punctuation, "
    "case, quotes, whitespace). NEVER translate, rephrase, or add analysis such as 'missing "
    "logic' or 'no implementation'.\n"
    "\n"
    "KEEP (verbatim, one span per line):\n"
    "  - [REQUIRED] File path + line for EVERY failure:  test_game_of_life.py:23, src/life.py:10\n"
    "  - Exception type + full message:  AssertionError: <msg>, TypeError: <msg>\n"
    "  - The FAILED test name and the exact failing assertion line:  assert ... == ..., \"...\"\n"
    "  - Function/class names + signatures when they are the failure site:  def step(self):\n"
    "  - Numeric results:  '1 failed, 2 passed', exit codes, returned values\n"
    "\n"
    "DROP (do NOT echo):\n"
    "  - Env/version boilerplate: 'platform darwin', 'pytest-8.3.2', 'rootdir:', 'collected N items'\n"
    "  - PASSED test lines (collapse to e.g. '2 passed')\n"
    "  - Blank lines, padding, repeated filler, full source of non-failing code\n"
    "\n"
    "OUTPUT: only the extracted spans, one per line, no headers, no prose, no commentary. "
    "If you omit any file path or error message, the extraction is invalid."
)

# one-shot 范例：用与真实任务【不同】的场景（test_calc.py / divide），
# 逼模型学「抽取模式」而非照抄范例里的字面量（避免回声）。
_SIDECAR_EXAMPLE_USER = (
    "Compress this pytest output:\n"
    "============================= test session starts ==============================\n"
    "platform linux -- Python 3.12.0, pytest-8.2.0, pluggy-1.5.0\n"
    "rootdir: /work\n"
    "collected 2 items\n\n"
    "test_calc.py .F                                              [100%]\n\n"
    "________________________________ test_divide ________________________________\n\n"
    "    def test_divide():\n"
    "        r = divide(1, 0)\n"
    ">       assert r == 5, \"division-by-zero-should-raise\"\n"
    "E       AssertionError: division-by-zero-should-raise\n"
    "test_calc.py:14: AssertionError\n\n"
    "=========================== short test summary info ============================\n"
    "FAILED test_calc.py::test_divide - AssertionError: division-by-zero-should-raise\n"
    "1 failed, 1 passed in 0.05s\n"
)
_SIDECAR_EXAMPLE_ASSISTANT = (
    "FAILED test_calc.py::test_divide\n"
    "test_calc.py:14: AssertionError: division-by-zero-should-raise\n"
    'assert r == 5, "division-by-zero-should-raise"\n'
    "1 passed, 1 failed"
)


def _lfm_condense(text: str, kind: str) -> str:
    """可选 LFM 二次浓缩：抽取式 prompt + one-shot 范例 + 忠实闸门。
    失败/不忠实则返回空串（交给调用方回退抽取式）。"""
    mt = C.CONTENT_COMPRESS_MAX_TOKENS
    msgs = [
        {"role": "system", "content": _SIDECAR_SYSTEM},
        {"role": "user", "content": _SIDECAR_EXAMPLE_USER},
        {"role": "assistant", "content": _SIDECAR_EXAMPLE_ASSISTANT},
        {"role": "user", "content":
            f"Now compress this {kind} the same way (keep verbatim, drop boilerplate):\n\n{text}"},
    ]
    try:
        out = M.chat_text_messages(msgs, role="planner",
                                  model_override=C.SIDECAR_MODEL, max_tokens=mt)
    except Exception as e:
        logger.info('%s', f'[sidecar] LFM 二次浓缩失败（{e}），回退抽取式。')
        return ""
    if not out or out.startswith("llm_error"):
        return ""
    summary = out.strip()
    # 忠实闸门：关键 token 不在摘要里 → 视为不可信，回退
    if not _faithful(summary, text):
        return ""
    return summary


def compress_content(text: str, kind: str = "tool output") -> tuple:
    """把大体积文本压缩。返回 (文本或摘要, 是否发生了压缩)。

    设计原则：压缩是工程能力，不是 prompt 技巧。1.2B 弱模型做抽象概括不可靠
    （臆测 / 中英互译 / 编造计数），故默认 extractive 抽取式截断（保真、确定性）。
    - 关闭 / 空 / 未超阈值：返回原文（was_compressed=False）。
    - 源码类内容：绝不压缩，原样返回（executor 需基于真实代码自修）。
    - extractive（默认）：head+tail 抽取式截断，保留头部上下文与尾部错误块。
    - lfm：抽象概括（受忠实闸门约束，关键 token 缺失则回退原文）。
    - hybrid：先抽取式，若仍超预算且开启 LFM 二次浓缩，再对截断结果做 LFM 抽取式浓缩。
    任何压缩失败都 fail-open 返回原文，绝不静默丢数据。
    """
    if not C.SIDECAR_ENABLED or not text or len(text) <= C.CONTENT_COMPRESS_THRESHOLD:
        return text, False
    # 源码工件（executor 正在编辑的文件）绝不压缩
    if _is_source_like(text):
        return text, False

    strategy = C.CONTENT_COMPRESS_STRATEGY
    if strategy == "lfm":
        summary = _lfm_condense(text, kind)
        if summary and len(summary) < len(text):
            return f"[LFM 压缩摘要 · 原文 {len(text)} 字符]\n{summary}", True
        return text, False

    # extractive（默认）或 hybrid 的第一阶段
    trunc = _extractive_truncate(text, C.CONTENT_COMPRESS_HEAD, C.CONTENT_COMPRESS_TAIL)
    if len(trunc) >= len(text):
        return text, False
    compressed_text = f"[抽取式压缩 · 原文 {len(text)} 字符]\n{trunc}"

    if strategy == "hybrid" and C.CONTENT_COMPRESS_LFM_PASS:
        summary = _lfm_condense(trunc, kind)
        if summary and len(summary) < len(trunc):
            return f"[LFM 二次浓缩 · 原文 {len(text)} 字符]\n{summary}", True
    return compressed_text, True


def auto_compact(messages) -> list:
    """上下文管理：超阈值时自动压缩对话，强制走 LFM 副驾模型。"""
    if not C.SIDECAR_ENABLED:
        return _compact.maybe_auto_compact(messages)
    return _compact.maybe_auto_compact(messages, model_override=C.SIDECAR_MODEL)


def manual_compact(messages, instructions: str = "") -> tuple:
    """手动 /compact 命令：压缩对话，强制走 LFM 副驾模型。返回 (new_messages, stats)。"""
    if not C.SIDECAR_ENABLED:
        return _compact.compact_conversation(messages, instructions=instructions, is_auto=False)
    return _compact.compact_conversation(messages, instructions=instructions, is_auto=False,
                                         model_override=C.SIDECAR_MODEL)
