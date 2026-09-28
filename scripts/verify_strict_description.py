"""验证 LM Studio + qwen 是否把 json_schema 字段 description 下发给模型（修订版：A/B 对照）。

上一版缺陷：只在 description 放指令、无 system prompt 对照，无法区分
「description 没下发」与「模型在 json_schema 下无视语义指令」，判定还用 AND 误判。

本版设计：
- 用完全特异、无语义的随机 token 作目标值，模型不可能自发产出。
- CONTROL：token 指令只在 system prompt，description 留空。
- TREATMENT：token 指令只在 description，system prompt 留空。
- 两者 schema 结构 / 用户消息 / strict 写法完全一致，仅指令位置不同。
判定：code_word 是否精确等于 TARGET（精确匹配，无歧义）。
若 CONTROL 命中而 TREATMENT 全部未命中 -> description 没下发（或被无视）。
"""
import json
import os
import sys
import time
import urllib.request

# 本机 localhost 直连，禁用任何代理（避免 sandbox-c 等代理接管）
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))

import openai
from swe_agent.log import logger

BASE = "http://localhost:1234/v1"
MGMT = "http://localhost:1234/api/v1"
KEY = "lm-studio"
MODEL = "qwen2.5.1-coder-7b-instruct"
TARGET = "ZXQ9_PURPLE_FROG"  # 完全特异 token，模型不会自发产出
INSTR = f"code_word 必须恰好是字符串 '{TARGET}'，不要输出其他任何内容。"

client = openai.OpenAI(base_url=BASE, api_key=KEY)


def ensure_loaded():
    try:
        with urllib.request.urlopen(f"{MGMT}/models", timeout=5) as r:
            data = json.load(r)
        loaded = [m.get("id") for m in data.get("data", []) if m.get("loaded_instances")]
    except Exception as e:
        logger.error('%s %s', '探测模型列表失败:', e)
        return False
    if MODEL in loaded:
        logger.info('%s', f'[ensure] {MODEL} 已加载')
        return True
    logger.info('%s', f'[ensure] {MODEL} 未加载，正在加载...')
    req = urllib.request.Request(
        f"{MGMT}/models/load",
        data=json.dumps({"model": MODEL}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            json.load(r)
    except Exception as e:
        logger.error('%s %s', '加载请求失败:', e)
        return False
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{MGMT}/models", timeout=5) as r:
                data = json.load(r)
            loaded = [m.get("id") for m in data.get("data", []) if m.get("loaded_instances")]
            if MODEL in loaded:
                logger.info('%s', f'[ensure] {MODEL} 加载完成')
                return True
        except Exception:
            pass
        time.sleep(2)
    logger.error('%s', '[ensure] 加载超时')
    return False


def make_rf(strict_val, desc_text):
    props = {"code_word": {"type": "string"}}
    if desc_text:
        props["code_word"]["description"] = desc_text
    body = {
        "type": "object",
        "properties": props,
        "required": ["code_word"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "verify_desc", "strict": strict_val, "schema": body},
    }


def run(system_content, strict_val, desc_text, label):
    rf = make_rf(strict_val, desc_text)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": "请按要求返回。"},
            ],
            response_format=rf,
            max_tokens=300,
            temperature=0,
        )
        raw = resp.choices[0].message.content
    except Exception as e:
        return label, "ERROR", str(e)[:200]
    try:
        obj = json.loads(raw)
        got = obj.get("code_word", "")
    except Exception:
        return label, raw, "JSON解析失败"
    hit = got == TARGET
    return label, raw, f"code_word={got!r} ==TARGET={hit}"


if __name__ == "__main__":
    logger.info('%s', '=== 验证 LM Studio 是否下发 json_schema description（对照版）===')
    logger.info('%s %s %s', 'TARGET（模型不可自发产出）:', TARGET, '\n')
    if not ensure_loaded():
        logger.info('%s', '模型未能加载，退出')
        sys.exit(1)
    for sv, lbl in [
        (True, "strict=True(bool)"),
        ("true", 'strict="true"(字符串)'),
        (False, "strict=False"),
    ]:
        logger.info('%s', f'\n##### strict = {lbl} #####')
        label, raw, verdict = run(INSTR, sv, None, f"control[{lbl}]")
        logger.info('%s', f'[{label} | 指令在 system prompt]\n  raw:{raw}\n  {verdict}')
        label, raw, verdict = run("", sv, INSTR, f"treat[{lbl}]")
        logger.info('%s', f'[{label} | 指令仅在 description]\n  raw:{raw}\n  {verdict}')
