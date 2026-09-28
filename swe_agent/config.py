#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_agent/config.py —— 全局配置与常量（从原 demo.py 抽取，保持取值一致）

注意路径处理：本模块位于仓库的 swe_agent/ 子包内，因此所有「相对仓库根」的
路径（模型权重目录、.venv、sessions 等）都必须以 REPO_ROOT 为基准，
禁止使用 Path(__file__).resolve().parent（那会指向 swe_agent/ 而非仓库根）。
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ====================== 路径基准 ======================
# 仓库根目录（swe_agent/ 的父目录）。所有模型权重软链、.venv、sessions 都以它为基准。
REPO_ROOT = Path(__file__).resolve().parent.parent
PWD_DIR = Path("./").resolve()

# ====================== .env 支持 ======================
# 在项目根目录放 .env（含 GLM_API_TOKEN 等密钥），启动时自动注入环境变量。
# .env 不应提交进 git（已加入 .gitignore）。load_dotenv 不会覆盖已存在的真实环境变量。
try:
    from dotenv import load_dotenv
    _dotenv_file = REPO_ROOT / ".env"
    if _dotenv_file.exists():
        load_dotenv(_dotenv_file)
except Exception:
    # python-dotenv 未安装时静默跳过（仍可依赖真实环境变量工作）
    pass

# ====================== 配置 ======================
# ---- GLM / zhipu 远程 Provider（Plan-B 分层架构的远程接入方之一） ----
# 密钥统一从环境变量 / .env 读取（不再硬编码默认值）。缺失时为空串，相关工具会优雅报错。
GLM_API_TOKEN = os.environ.get("GLM_API_TOKEN", "")
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
# 注意模型选型：glm-4.1v-thinking-flashx 实测【不响应 tools/tool_choice】（纯文本思考输出），
# 导致 Executor 的 agent_action 原生函数调用失效、退化为手写 JSON 文本再靠正则兜底。
# glm-4.7 当前作为 zhipu 默认模型（临时代替 glm-4.5-flash / glm-4.7-flash），实测原生
# 返回 tool_calls（arguments 为合法 JSON）。该值是 zhipu provider 的「默认模型 id」；
# 角色若显式分配了其它 zhipu 模型会以角色为准（如 GLM_MODEL=glm-4.5-flash 可回退）。
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4.7")
# zhipu 文档：GLM-4.7 系列 max_tokens 合法范围 1..131072（最大输出 128K），GLM-4.5 系列最大 96K。
# GLM 思考(reasoning) + 格式化 JSON 输出会让真实 completion token 远大于表面契约（实测一次顶层调用
# ~4416 token 含 prompt，复杂任务逼近 10K 很常见），故对远程大模型放宽到 32768（=96K 的三分之一，
# 不到 GLM-4.7 上限的 1/4），既给思考与 JSON 留足余量、避免静默截断，又远在官方上限内。
GLM_MAX_TOKENS = int(os.environ.get("GLM_MAX_TOKENS", "32768"))
GLM_TIMEOUT = (10, 180)          # (connect, read)：glm-4.5-flash 出契约约 50s，180s 留足余量且能更快暴露异常
GLM_REPLAN_MAX = 2               # 故障路由：最多退回 Planner 重规划次数
# 思考链开关：默认 disabled（关掉 thinking）。理由：glm 思考模型会把 reasoning 混进 content，
# 而 TEXT_JSON executor 依赖从 content 抽 JSON，思考噪声会让模型吐更乱的多候选 JSON（含 非法 \' 转义）。
# 注意 zhipu 文档称 thinking.type 仅 GLM-4.5+ 支持；对 4.1 系若 API 以 4xx 拒绝，传输层会【自动去掉
# thinking 重试一次】并打告警，不会让 run 崩。设为 "enabled" 可重新开启思考。
GLM_THINKING = os.environ.get("GLM_THINKING", "disabled")  # 取值: disabled / enabled / ""(沿用远端默认)

# ---- Zhipu 网络搜索（Web Search API，tools 类）----
# 给大模型用的搜索引擎：意图增强检索、结构化输出、多引擎。作为 web_search 工具的后端。
# 文档：https://docs.bigmodel.cn/api-reference/工具-api/网络搜索
# 端点：POST {GLM_BASE_URL}/web_search（GLM_BASE_URL 默认 https://open.bigmodel.cn/api/paas/v4）
# 鉴权：复用 GLM_API_TOKEN（同一开放平台密钥）。
WEB_SEARCH_ENGINE = os.environ.get("WEB_SEARCH_ENGINE", "search_std")   # search_std/search_pro/search_pro_sogou/search_pro_quark
WEB_SEARCH_COUNT = int(os.environ.get("WEB_SEARCH_COUNT", "10"))         # 返回条数 1-50
WEB_SEARCH_CONTENT_SIZE = os.environ.get("WEB_SEARCH_CONTENT_SIZE", "medium")  # medium=摘要 / high=详细
WEB_SEARCH_RECENCY = os.environ.get("WEB_SEARCH_RECENCY", "noLimit")     # oneDay/oneWeek/oneMonth/oneYear/noLimit
WEB_SEARCH_INTENT = os.environ.get("WEB_SEARCH_INTENT", "false").lower() == "true"  # 是否做搜索意图识别
WEB_SEARCH_TIMEOUT = (10, 60)    # (connect, read)
WEB_SEARCH_FALLBACK = os.environ.get("WEB_SEARCH_FALLBACK", "1") == "1"  # Zhipu 失败/未配置时回退 DuckDuckGo
# 系统提示词：web_search 可用时是否注入「可联网搜索获取更多信息」引导词（默认开启）
WEB_SEARCH_GUIDANCE = os.environ.get("WEB_SEARCH_GUIDANCE", "1") == "1"

# ---- 插件系统 ----
# 是否启用插件加载（默认开启）。设 ENABLE_PLUGINS=0 可整体关闭。
ENABLE_PLUGINS = os.environ.get("ENABLE_PLUGINS", "1") == "1"
# 插件根目录：默认项目内 plugins/（让工程自带插件如 websearch 在 harness 启动时自动加载）；
# 可用 SWE_PLUGINS_ROOT 覆盖为 ~/.claude/plugins 等真实 claude-code 插件目录。
PLUGINS_ROOT = os.environ.get("SWE_PLUGINS_ROOT") or str(REPO_ROOT / "plugins")
# 是否连接插件声明的 MCP 服务器（默认关闭，避免未授权外连）；可用 ENABLE_PLUGIN_MCP=1 开启。
ENABLE_PLUGIN_MCP = os.environ.get("ENABLE_PLUGIN_MCP", "0") == "1"

# ---- 角色 → 模型（catalog id，可被 env 覆盖）----
# 这是「模型可换」的开关：planner/executor 各自从 models.PROVIDERS/MODELS
# 选一个模型，不绑定任何具体 provider。换 GLM → 换本地模型，只是改下面任一 env。
# 空串 = 该角色不挂远程模型（优雅降级：planner→本地自规划）。
# 三层角色默认模型（全部本地 LM Studio qwen 部署，零远程配额）：
#  - planner / analyzer / executor = qwen2.5.1-coder-7b-instruct（NATIVE_TOOLS，本地 LM Studio 承载）
# 所有角色统一走本地 qwen，避免远程 glm 配额/网络依赖（用户 2026-08-31 决策）。
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "lfm2.5-2.6b")
ANALYZER_MODEL = os.environ.get("ANALYZER_MODEL", "lfm2.5-2.6b")
EXECUTOR_MODEL = os.environ.get("EXECUTOR_MODEL", "lfm2.5-2.6b")

# 升级子 agent：弱模型（lfm2.5）主产出不可用（空/无法解析为 JSON）时，升级给强脑做复杂判断。
# 仅在「主模型失败」时触发，非每轮调用。默认指向本地 LM Studio 已加载的
# qwen2.5.1-coder-7b-instruct，保证升级路径也走本地、不依赖远程 glm-4.7 配额/网络。
PLANNER_FALLBACK_MODEL = os.environ.get("PLANNER_FALLBACK_MODEL", "lfm2.5-2.6b")
ANALYZER_FALLBACK_MODEL = os.environ.get("ANALYZER_FALLBACK_MODEL", "lfm2.5-2.6b")

# ---- forge v2 验收链模型（可配置；默认复用 driver，可用更强本地模型覆盖）----
# FORGE_PLANNER_MODEL merged to PLANNER_MODEL（旧名已弃用）；FORGE_TESTER_MODEL renamed to TESTER_MODEL
# 空串 = 复用 EXECUTOR_MODEL；"off" = 关闭该环节。
# 例：FORGE_TESTER_MODEL=liquid/lfm2.5-1.2b 让独立验收用一个更「无关」的小模型做裁判。
# ⚠️ 2026-09-02 修复：此处曾重复赋值 `PLANNER_MODEL = os.environ.get("FORGE_PLANNER_MODEL", …)`，
# 无条件覆盖上方 :87 的 `PLANNER_MODEL = os.environ.get("PLANNER_MODEL", …)`，
# 导致 **PLANNER_MODEL 环境变量被静默吞掉**（只有旧名 FORGE_PLANNER_MODEL 生效）。
# 现已统一读 PLANNER_MODEL——其唯一定义在 :87，此处不再重复赋值（单点定义，禁止再加回来）。
TESTER_MODEL = os.environ.get("TESTER_MODEL", "lfm2.5-2.6b")

# ---- Executor 是否允许派发子智能体（agent 动作）----
# 默认关闭：Executor 的核心职责是「亲自」实现代码并跑测试。若允许它调用 agent，
# 弱模型会把活甩给一个「从头开始、零任务上下文」的子智能体，且常给出空 prompt
# （如「请生成 pytest 测试用例并补全 src/life.py」），毫无价值还浪费轮次。
# 设 EXECUTOR_ALLOW_AGENT=1 可放开（仅顶层执行体生效；plan 子智能体始终可嵌套派发）。
EXECUTOR_ALLOW_AGENT = os.environ.get("EXECUTOR_ALLOW_AGENT", "0") == "1"

# 无人值守模式（batch / 后台 e2e）：True 时 harness 不向用户弹交互式提问，
# 改由 BUILD 层把 ask 工具从下发的工具集摘除（模型根本调不到），
# 故障路由/验收失败自动判 fail 或 continue，绝不卡 stdin。默认 False（交互模式）。
UNATTENDED_MODE = os.environ.get("UNATTENDED_MODE", "0") == "1"

# ---- 完成标准：工作区测试 count>0 且全绿（唯一的成功判定）----
# 无隐藏测试、无交付物检查。模型只需让 WORKSPACE 内的测试全部通过即可结束。

# ---- LM Studio（本地 OpenAI 兼容服务，默认 liquid/lfm2.5-1.2b）----
# 用官方 OpenAI SDK 调用（不手搓 requests），base_url 指向 LM Studio 的 /v1。
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "liquid/lfm2.5-1.2b")
# 单轮生成上限：本地 7B 写代码时整文件塞进 tool-call 的 JSON arguments，2048 会在
# 文件较大时于 JSON 中途截断导致 parse_actions 拿到半截 JSON。提到 8192（qwen 7B 输出
# 8k 没问题，瓶颈是输出 cap 而非上下文）。1.2B analyzer 走 lmstudio_chat_messages 时
# 会把传入 max_tokens 收敛到这里（min），不影响其行为。
LMSTUDIO_MAX_TOKENS = int(os.environ.get("LMSTUDIO_MAX_TOKENS", "8192"))

# 本地 rapid-mlx 服务 + Ling 思考分离已废弃：仅保留 GLM(远程) 与 LM Studio(本机 OpenAI 兼容)
# 两个 provider，全部走标准 OpenAI tool_calls 协议（见 models.py）。

# Agent 写代码的工作沙箱（避免污染仓库根目录）
# 必须是绝对路径：harness 多处用 cwd=str(WORKSPACE) 派生子进程，相对路径会依赖
# 调用方 cwd，导致路径错乱。
# 允许通过 SWE_WORKSPACE 环境变量覆盖（电池给每个任务传独立隔离目录，
# 彻底消除「跨任务/跨运行的工作区文件泄漏」，见 run_e2e_battery.py）。
# WORKSPACE = Path(os.environ.get("SWE_WORKSPACE", str(REPO_ROOT / "agent_sandbox"))).resolve()
WORKSPACE = Path(os.environ.get("SWE_WORKSPACE", str(PWD_DIR))).resolve()

# ====================== 分层 KB 作用域与磁盘布局（2026-09-12 定稿） ======================
# 四层作用域（谁和谁共享）：
#   global  : 跨进程唯一 —— 人手动投放的跨项目文档
#   proj    : 每项目唯一 —— 被操作项目自带的文档（WORKSPACE 下的 KnowledgeBase/ + docs/）
#   code    : 每项目唯一 —— 被操作项目的代码文件
#   session : 每对话唯一 —— 不进 LayeredKB，归 ContextManager 管辖
# 全部状态落 SWE_AGENT_HOME（默认 ~/.swe_agent），与仓库 checkout 解耦：仓库可删可换，
# 索引与缓存仍在；也因此 e2e 清沙盒不会连带铲掉缓存。
SWE_AGENT_HOME = Path(os.environ.get("SWE_AGENT_HOME", str(Path.home() / ".swe_agent"))).resolve()
# 可执行/安装产物目录（deploy.sh 写 bin/swe-agent 于此；唯一需要进 PATH 的目录）
BIN_DIR = SWE_AGENT_HOME / "bin"
# global 层两件套：raw = 人投放文档的源目录；global_kb = 它的派生索引（跨项目共享，一份就够）
GLOBAL_KB_RAW_DIR = SWE_AGENT_HOME / "global_kb_raw"
GLOBAL_KB_DIR = SWE_AGENT_HOME / "global_kb"


def _project_name() -> str:
    """项目作用域键：优先 git 仓库根名，回落 WORKSPACE 目录名。

    为什么优先 git 根：WORKSPACE 默认取启动时的 cwd，在仓库子目录里启动会拿到子目录名，
    同一个项目因此分裂成多份缓存。git 根能把「同一仓库的任意子目录」归一到同一项目名。
    e2e 沙盒没有 .git → 回落目录名（正好是 fizzbuzz / conway 这类稳定任务名）。
    纯函数 + 异常安全：git 缺失 / 非仓库 / 超时 / 任何异常都只回落，绝不让 import 失败。
    """
    ws = str(Path(os.environ.get("SWE_WORKSPACE", str(PWD_DIR))).resolve())
    try:
        out = subprocess.run(["git", "-C", ws, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip()).name
    except Exception:
        pass
    return Path(ws).name or "default"


# 项目作用域键（SWE_PROJECT 可显式覆盖，便于 e2e / 测试指定稳定名）
PROJECT_NAME = os.environ.get("SWE_PROJECT") or _project_name()
# 项目目录：挂 projects/ 之下，与 home 下的全局项（bin/、global_kb*/）分命名空间，
# 杜绝「项目名撞全局目录名」（如某项目恰好叫 bin）。
PROJECT_DIR = SWE_AGENT_HOME / "projects" / PROJECT_NAME
CODE_KB_DIR = PROJECT_DIR / "code_kb"     # code 层持久化（原 RAG_CACHE_DIR/code）
PROJ_KB_DIR = PROJECT_DIR / "proj_kb"     # proj 层持久化（原 RAG_CACHE_DIR/proj）
LOGS_DIR = PROJECT_DIR / "logs"           # 该项目运行日志（harness.log / lmstudio_requests.jsonl / e2e_battery/）

# ---- 分层 KB 的源（raw）----
# 分工口诀：跨项目通用文档 → 手动投放进 GLOBAL_KB_RAW_DIR；
#           仅属于本项目的文档 → 项目自己的 docs/ 或 KnowledgeBase/。
# KB_ROOTS / KB_GLOBAL_ROOTS 用冒号分隔，可 env 覆盖（一旦覆盖，默认值整组失效）。
KB_ROOTS = [Path(p) for p in os.environ.get("KB_ROOTS", "").split(":") if p] or [
    WORKSPACE / "KnowledgeBase", WORKSPACE / "docs",
]
KB_GLOBAL_ROOTS = [Path(p) for p in os.environ.get("KB_GLOBAL_ROOTS", "").split(":") if p] or [
    GLOBAL_KB_RAW_DIR,
]

# 根据当前 python 推导 venv 路径，保证 shell 里调用的 python/pytest 与运行环境一致
VENV_BIN = Path(os.sys.executable).parent

# 跨 run 的 agent 记忆文件（绝对路径，随仓库根 rebasing）
AGENT_MEMORY_PATH = REPO_ROOT / "agent_memory.json"
AGENT_MEMORY_MAX = 40   # 记忆条数上限（supervisor 与 state 共用，单一事实来源在此）

MAX_ITER = 60          # Agent 最大决策步数（安全上限，正常由下面三层循环边界先触顶）

# tester 独立验收最大轮数（看门狗定位：防 executor 自欺，不深验；触顶标 skipped 而非 fail）
# ⚠️ 2026-09-03 实测修正：旧值 2 在数学上不可能完成「读实现 + 读测试 + 交卷」三步——
# e2e(01_fizzbuzz) 实测 iter1 读 src/fizzbuzz.py、iter2 读 test_fizzbuzz.py，配额在读文件时
# 恰好耗尽，第 3 轮才轮得到 finish_verify → 永远触顶 → verify 恒为 skipped，第 4 道闸门形同虚设。
# 调到 4：留出「定位(1) + 读实现/读测试(1~2) + 交卷(1)」的最小可行预算。
MAX_TESTER_ITER = 4

# ---- Analyzer（只读调研阶段）收敛与降级阈值（BUILD 层收口，不进 prompt）----
# 软空响应（models.EMPTY_RESPONSE：server 返 200 但模型既无 tool_calls 也无 content，
# 畸形 tool call 被丢弃）时，【同 prompt 重试只会得到同样的空响应】。旧行为是干耗完
# max_steps 再静默放弃整个主循环 —— e2e(01_fizzbuzz) 里 analyzer 因此被整体旁路，
# 日志上没有任何失败痕迹。现在：连续软空超过此阈值即判定该模型对本任务不可用，
# 换 fallback 模型重跑主循环，而不是原地死等。
ANALYZER_SOFT_EMPTY_LIMIT = int(os.environ.get("ANALYZER_SOFT_EMPTY_LIMIT", "1"))
# 主循环内「模型只吐散文、不肯调工具」的轮次上限：超阈值即结束本轮，不再干耗预算。
ANALYZER_NO_TOOL_ROUNDS = int(os.environ.get("ANALYZER_NO_TOOL_ROUNDS", "2"))
# 连续重复同一组动作（动作指纹相同）的次数上限：达到即要求立即交卷，再重复则结束本轮。
# 实测 ling-3.0-tiny 会对同一组文件连发两轮完全相同的 read_file，旧代码无任何进展检测。
# ⚠️ 统一收敛后由 LoopConfig.repeat_threshold 接管（analyzer 用 ANALYZER_REPEAT_THRESHOLD），
# 此常量仅作为「若未显式传 repeat_threshold 时的兜底值」，不再直接驱动 analyzer 主循环。
ANALYZER_REPEAT_LIMIT = int(os.environ.get("ANALYZER_REPEAT_LIMIT", "2"))
# 主模型不可用时，升级到 fallback 模型重跑【完整主循环】的预算。
# 旧行为：fallback 只发一次「强制 finish_analysis」的孤立调用，一次失手就整体放弃。
ANALYZER_FALLBACK_STEPS = int(os.environ.get("ANALYZER_FALLBACK_STEPS", "3"))
# 统一收敛后新增：analyzer 走 make_agent 统一 loop，以下两个常量直接驱动 LoopConfig，
# 不再每个 role 写自己的重复/停滞计数逻辑（对应「放宽 quota 和 repeat limit」）。
# - ANALYZER_MAX_ITER：analyzer 主循环决策步数上限（quota），较旧 max_steps 放宽，
#   给弱模型更多自愈/纠正空间（如 cd /testbed 失败后能多次重试真实命令）。
ANALYZER_MAX_ITER = int(os.environ.get("ANALYZER_MAX_ITER", "24"))
# - ANALYZER_REPEAT_THRESHOLD：analyzer 的「连续重复动作 → 停滞」阈值（repeat limit），
#   较全局 LOOP_REPEAT_THRESHOLD 放宽；配合「失败不计次数」原则，工具调用失败（cd /testbed
#   等）不推进该计数，避免弱模型被重复/停滞阈值提前截断。
ANALYZER_REPEAT_THRESHOLD = int(os.environ.get("ANALYZER_REPEAT_THRESHOLD", "4"))
# ---- Judge（结构化输出裁判，2026-09-03；取代 F3 长度阈值 + 启发式判断）----
# 用【独立模型】按显式标准评「产出算不算数」，输出 json_schema {result:yes/no/notsure, reason}。
# 独立于角色模型（默认 qwen），避免生产者自判偏差；实测 qwen/Spark/Ling(关思考) 均 10/10 可靠，
# lfm 不适合作 judge（思考链吃光预算、~128s/call、需 temp=1.1、thinking 不能关）。
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen2.5.1-coder-7b-instruct")
# notsure 重试次数上限：保持 temp=0 同 prompt 重试（不升温——实测升温对判定零影响），
# 耗尽则降级为 no 回灌续轮。
# 【硬化警告】temp=0 重试靠 append critique 改上下文而非升温；对强模型（Spark/Ling/qwen）有效，
# 但换弱模型当 judge（如 LFM2.5-350M）时，弱模型对追加 critique 免疫 + temp=0 零方差 → 每轮吐同一偏置 = 真·空转。
# 该场景已被「小模型不能当 judge」结论排除，故当前栈安全；若将来引入弱 judge 模型，必须升温或换强模型。
JUDGE_MAX_RETRY = int(os.environ.get("JUDGE_MAX_RETRY", "3"))
# judge 默认生成参数（按模型名子串覆盖，见 JUDGE_MODEL_OVERRIDES）。env 可覆盖：JUDGE_TEMPERATURE / JUDGE_MAX_TOKENS。
# ⚠️ 弱模型（小参数 LLM）当 judge 时，temp=0 重试会退化成确定性空转（见上方 JUDGE_MAX_RETRY 警告），
# 勿靠升温硬救——直接换大模型（json_schema 约束 + 强模型才是语义 judge 的正确解）。
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "1024"))
# 思考类模型需特殊参数（按模型 id 子串匹配，BUILD 配置，不硬编码全局）：
#   ling → 必须 enable_thinking=false（否则思考吃光预算空输出），temp=0/1024 即可；
#   lfm  → 不支持 thinking off（会挂死），须思考 ON + 高温(1.1) + 大预算(4096)，~128s/call，
#          仅作兜底、不推荐（生产者模型，宜写代码不宜当裁判）。
JUDGE_MODEL_OVERRIDES = {
    "ling": {"temperature": 0.0, "max_tokens": 1024, "thinking_off": True},
    "lfm":  {"temperature": 1.1, "max_tokens": 4096, "thinking_off": False},
}
# analyzer 主循环内「judge 判 no/notsure 后回灌续轮」的轮次上限：超阈值即接受 best-effort 交付，
# 交 fallback/planner，避免弱模型在「judge 不过→补完→仍不过」上无限空转。
ANALYZER_JUDGE_ROUNDS = int(os.environ.get("ANALYZER_JUDGE_ROUNDS", "3"))

# judge 提示词（BUILD 层常量，含「铁律：空/占位符/客套话→必须 no」——实测弱模型不加此铁律会把
# 退化输入误判 yes/notsure，加后 Spark/Ling/qwen 均 10/10；与模型无关，不可省）。
JUDGE_PROMPTS = {
    "analyzer": (
        "你是严格的调研质量审校。判断以下调研结果是否包含有效的调研信息。标准：1. 有概念解释；2. 有需求说明。"
        "【铁律】若内容为空、仅为占位符或客套话（如'已完成''详见上文'但无具体内容）、"
        "或没有实质的概念解释与需求说明，必须回答 no，不得给 yes。只有确实同时包含两者才回答 yes。"
        "只输出结构化结果。\n\n【调研结果】\n{content}"
    ),
    "tester": (
        "你是严格的测试质量审校。判断以下交付是否合格。标准：1. 是否包含有效的单元测试；2. 是否包含有效的实现。"
        "【铁律】若实现文件或测试文件为空、测试数量为0、或只是占位符或客套话，必须回答 no，不得给 yes 或 notsure。"
        "只有确实包含有效非空单测与非空实现才回答 yes。只输出结构化结果。\n\n【交付】\n{content}"
    ),
}

# ======================================================================
# 三层嵌套循环边界（analyzer -> planner -> loop_1(loop_2(loop_3)->lint)->pytest->tester)）
# ----------------------------------------------------------------------
# loop_1：整体重试（pytest / tester 失败，或 lint 多次不过时整轮重来，带失败反馈）
# loop_2：编码重试（同一 attempt 内，lint 失败就重跑 loop_3 编码）
# loop_3：executor 工具调用循环（原生 toolcall 协议，每轮 1 个工具调用，稳定）
MAX_ATTEMPTS = int(os.environ.get("SWE_MAX_ATTEMPTS", "3"))   # loop_1 上限
MAX_ROUNDS = int(os.environ.get("SWE_MAX_ROUNDS", "3"))       # loop_2 上限
MAX_STEPS = int(os.environ.get("SWE_MAX_STEPS", "8"))         # loop_3 单轮上限
MODEL_RETRY = int(os.environ.get("SWE_MODEL_RETRY", "2"))     # 单次模型推理失败（本地模型偶发崩溃）的重试次数
MODEL_BACKOFF = float(os.environ.get("SWE_MODEL_BACKOFF", "2.0"))  # 重试之间的退避秒数（给本地 lmstudio 留出恢复窗口）

# ---- Lint（编码后静态校验，作为 pytest 之前的第 3 道本地闸门）----
LINT_ENABLED = os.environ.get("SWE_LINT_ENABLED", "1") == "1"
LINT_MAX_ISSUES = int(os.environ.get("SWE_LINT_MAX_ISSUES", "40"))

# ======================================================================
# Executor 历史重置（GLOBAL 收口到 BUILD 层）
# ----------------------------------------------------------------------
# 设计：当「任务列表里某个 task 被标记完成」时，下一轮把 executor 的对话历史
# 重建为  [静态 system(命中 prompt 缓存) + 最近 K 轮(连续性) + 更新后的任务列表(指令置底)]。
# 这样：
#   - input 长度从「随轮数线性增长」变成「单任务内滚动 + 任务切换时回落」，整体 ≈ 常数；
#   - system 静态 → 不破 prompt 缓存（任务列表故意放 user 轮，不进 system）；
#   - 单任务内的上下文溢出仍由 maybe_auto_compact 兜底（本开关不替代它）。
# 排序严格用 A：任务列表必须置底，因为 chat 循环里模型永远对「数组最后一条消息」作答。
EXECUTOR_RESET_ON_TASK_DONE = False  # 任务完成时【不】重置 executor 历史——保留编译/测试错误线索供 SelfHeal（Manning《Designing AI Agents》SelfHealLoop 要求错误始终留在上下文；旧逻辑每次写文件清空历史，导致模型修一行就丢全部错误而烧光 30 轮预算）
EXECUTOR_RESET_KEEP_TURNS = 1        # 重置时保留最近 K 轮(assistant动作 + tool结果)作连续性

# ---- 全局「停滞/循环」防护阈值（统一覆盖 shell / write_file / edit_file / read_file 等所有动作）----
LOOP_REPEAT_THRESHOLD = 3   # 同一动作指纹连续出现 ≥N 次 → 判定为「重复循环」
ALLOW_REPLAN = os.environ.get("ALLOW_REPLAN", "0") == "1"  # 是否允许「重规划」：loop-guard 升级第2次的强制重规划 + 故障路由退回 Planner 重规划。默认关闭（"0"），强制 Executor 自行 debug；设 ALLOW_REPLAN=1 恢复。
# 单轮最多【执行】的工具调用数。模型可在一轮内下发多个 tool_call（parallel tool calls），
# harness 全部顺序执行（不再只跑第一个）；超出此上限的调用不执行，但仍回一条 tool 消息
# 说明原因——每个 tool_call_id 都必须有对应结果，否则部分 provider 下一轮直接报 400。
MAX_ACTIONS_PER_RESPONSE = int(os.environ.get("SWE_MAX_ACTIONS_PER_RESPONSE", "5"))
# 多工具轮次的结果预算：单条工具结果截断上限 = max(OUTPUT_BUDGET // 本轮调用数, 此下限)。
# 防止「一轮 N 个调用 × 每个 6000 字」把弱模型上下文直接撑爆（单调用时等于 OUTPUT_BUDGET，行为不变）。
OUTPUT_BUDGET_PER_TOOL_MIN = int(os.environ.get("SWE_TOOL_BUDGET_MIN", "1200"))

# ---- 终态校验「必然失败」防护（用户明确要求：别在必然 fail 的测试上打转）----
# 已接线（2026-09-02，P1）：loop_1 单杠（pytest）连续失败达此阈值 → supervisor._l1_gate
# 置 ctx.metadata["unsolvable"] 并 break 终止本任务，run_agent 据此返回 verdict="unsolvable"
# （而非一路 continue 到 MAX_ATTEMPTS 耗尽 / 烧满 1800s 被 SIGKILL）。signature-agnostic。
VAL_DOOMED_THRESHOLD = 2

# ---- 漂移（磁盘代码 vs Planner 契约）回灌次数上限 ----
# 回灌是「提醒」不是「阻断」：同一 run 内最多回灌 N 次，避免刷屏挤占上下文。
# 计数由 guard 的 drift_injections 限制持有（scope=run，不逐轮重置）。
DRIFT_MAX_INJECTIONS = 3

# 阈值保留于此；运行态计数器(_VAL_SIG_HIST 等)已迁至 harness 模块内自持，
# 由 supervisor 在每轮任务开头重置 harness.* 本体，避免跨任务不清零。

# ---- 「接手既有代码」模式：保护预置测试夹具（oracle）不被模型覆盖 ----
_TAKEOVER_KEYWORDS = (
    "接手", "接管", "中途接手", "现有代码", "已有代码", "已有测试", "现有测试",
    "不要从头重写", "不要重写整个项目", "existing code", "take over", "takeover",
)
_TEST_FILE_RE = re.compile(
    r"(^|/)(test_[^/]*\.py|[^/]*_test\.py|"            # Python
    r"[^/]*\.test\.js|[^/]*\.spec\.js|[^/]*\.test\.ts|"  # JS/TS
    r"[^/]*_test\.go|"                                  # Go
    r"[^/]*Test\.java|Test[^/]*\.java)$"                # Java
)

# ---- 单次写入行数护栏（门控，不在工具内硬拒绝；由 BEFORE_TOOL_CALL guard 执行）----
# 弱模型输出预算小，一次 dump 上百行/上百函数易被截断，而 write_file 是整文件覆盖，
# 反复重发同一大文件形成「截断→重发→上下文膨胀」死循环，故需上限。阈值进 BUILD 层，
# 环境变量可覆盖（保持兼容）。测试文件允许更高（实测 3B 模型写 240 行会被拒绝后退化空输出）。
WEAK_MAX_WRITE_LINES = int(os.environ.get("WEAK_MAX_WRITE_LINES", "500"))
WEAK_MAX_TEST_WRITE_LINES = int(os.environ.get("WEAK_MAX_TEST_WRITE_LINES", "1000"))
STRONG_MAX_WRITE_LINES = int(os.environ.get("STRONG_MAX_WRITE_LINES", "1000"))
STRONG_MAX_TEST_WRITE_LINES = int(os.environ.get("STRONG_MAX_TEST_WRITE_LINES", "2000"))

# ---- 单次读取行数护栏（BEFORE_TOOL_CALL guard：ReadSizeGuard）----
# read_file 单次读取行数上限（不按模型强弱分，统一 100 行）；limit 给了看 limit，
# 没给则读目标文件实际行数。超限 REJECT（回灌 reason、提示用 limit 分批）。
MAX_READ_LINES = int(os.environ.get("MAX_READ_LINES", "100"))

# ---- 可靠性 / 安全 / 持久化配置 ----
OUTPUT_BUDGET = int(os.environ.get("DEMO_OUTPUT_BUDGET", "6000"))
# 是否允许模型在一轮内下发多个 tool_call（OpenAI 的 parallel_tool_calls）。
# 默认放开：强模型（如 Ling）会一次下发多个读操作，harness 已支持全部执行；
# 旧值 False 是弱模型时代的遗留，会让服务端每轮只肯返回一个工具调用。
PARALLEL_TOOL_CALLS = os.environ.get("SWE_PARALLEL_TOOL_CALLS", "1") == "1"
YOLO_MODE = False
MODEL_OVERRIDE = ""
# 会话记录：项目作用域下（一次对话属于「在哪个项目里聊的」）。2026-09-12 起从
# REPO_ROOT/sessions 迁到 <SWE_AGENT_HOME>/projects/<project>/sessions；旧会话不做
# fallback（用户拍板）—— REPO_ROOT/sessions/*.json 不再能 --session 恢复。
SESSIONS_DIR = PROJECT_DIR / "sessions"
SESSION_ID: Optional[str] = None

# LSP 语言服务器命令（init 时自动解析/安装；None 表示改用 PATH 检测）
LSP_CMD: Optional[List[str]] = None
# 项目 venv python（优先用于 pip 安装 pylsp，保证与 rapid-mlx 同环境）
_VENV_DIR = REPO_ROOT / ".venv"
VENV_PY = (_VENV_DIR / "bin" / "python") if _VENV_DIR.exists() else Path(os.sys.executable)

# 运行统计（观测）
STATS: Dict[str, Any] = {
    "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
    "retries": 0, "rounds": 0, "actions": {},
    "glm_calls": 0, "glm_tokens": 0,
    # 历史重置观测：每轮发给模型的 input 字符数（rapid-mlx 常不回 usage，用字符数做 token 代理）
    "input_chars_total": 0,
    "input_chars_per_round": [],
    "reset_events": 0,
}

# 终态校验重试预算（harness 级，不靠提示词）
MAX_VAL_RETRIES = 2
VAL_HARD_CAP = 5

# ====================== 对话压缩配置 ======================
CONTEXT_WINDOW = int(os.environ.get("DEMO_CONTEXT_WINDOW", "32768"))
AUTOCOMPACT_BUFFER = 13000
# 目标模型上下文窗口（token 预算来源）：ContextManager.prepare_messages 按它推导
# RAG 预算 + 历史压缩预算。默认 96000（现代本地/远程模型常见窗口）；可由 env
# MODEL_CONTEXT_LENGTH 覆盖。每模型亦可在 models.MODELS[...]["context_length"] 单独声明，
# 缺省回退到此值（见 models.model_context_length）。
MODEL_CONTEXT_LENGTH = int(os.environ.get("MODEL_CONTEXT_LENGTH", "96000"))
MAX_AUTO_COMPACT_FAILURES = 3
AUTO_COMPACT_ENABLED = True

# ====================== LFM Sidecar（上下文管理 + 大内容压缩） ======================
# 独立的小模型副驾（liquid/lfm2.5-1.2b，本机 LM Studio 承载），【只做无状态压缩/摘要】，
# 不承载任何主 agent 角色（主角色已全部走 qwen）。它负责：
#   1) 对话上下文压缩（compact.py 的自动/手动压缩改走 SIDECAR_MODEL，不再误用 qwen）；
#   2) 大体积工具结果（shell 输出 / 文件读取）落地前先摘要，避免主上下文被噪声撑爆。
# 失败一律 fail-open：压缩失败就保留原文，绝不静默丢数据。
SIDECAR_MODEL = os.environ.get("SIDECAR_MODEL", "liquid/lfm2.5-1.2b")
SIDECAR_ENABLED = os.environ.get("SWE_SIDECAR_ENABLED", "1") == "1"
# 工具结果超过该字符数才触发压缩（短结果不值得摘要）
CONTENT_COMPRESS_THRESHOLD = int(os.environ.get("SWE_CONTENT_COMPRESS_THRESHOLD", "2000"))
# 压缩摘要最多生成 token 数（仅 LFM 二次浓缩路径使用）
CONTENT_COMPRESS_MAX_TOKENS = int(os.environ.get("SWE_CONTENT_COMPRESS_MAX_TOKENS", "400"))
# 大内容压缩【默认策略】= extractive（抽取式截断，保真、确定性），而非 abstractive（LFM 抽象概括）。
# 抽象概括在 1.2B 弱模型上不可靠（会臆测/中英互译/编造计数），故默认走工程化的抽取式截断。
CONTENT_COMPRESS_STRATEGY = os.environ.get("SWE_CONTENT_COMPRESS_STRATEGY", "extractive")  # extractive | lfm | hybrid
# 抽取式截断：保留头部上下文 + 尾部错误块（pytest 失败摘要在末尾）。源码类内容跳过压缩。
CONTENT_COMPRESS_HEAD = int(os.environ.get("SWE_CONTENT_COMPRESS_HEAD", "700"))
CONTENT_COMPRESS_TAIL = int(os.environ.get("SWE_CONTENT_COMPRESS_TAIL", "1400"))
# 是否在抽取式截断之后，再用 LFM 做一次「抽取式」二次浓缩（默认关：弱模型不可靠，且抽取式已足够）。
# 开启时仍受忠实闸门约束——关键 token（.py 路径 / Error / Exception）必须出现在摘要中，否则回退抽取式。
CONTENT_COMPRESS_LFM_PASS = os.environ.get("SWE_SIDECAR_LFM_PASS", "0") == "1"

# —— 对话压缩副驾（BUILD 层换模编排）——
# 实验结论（2026-08-31/09-01）：对话压缩在 1.2B 上不可信（会编造完成态/丢关键锚点），
# 2.6B 是合格的压缩副驾。但本机显存下 2.6B 与主 executor（qwen-7B）【无法同时驻留】，
# 故压缩前需 unload qwen → load 2.6b，压缩后 unload 2.6b → reload qwen。
# 这组换模由 swe_agent/model_swap.py 在 BUILD 层自动编排，失败一律 fail-open 并尽力还原 executor。
SIDECAR_COMPRESS_MODEL = os.environ.get("SIDECAR_COMPRESS_MODEL", "lfm2.5-2.6b")
# —— 全局 load/unload 总开关（GLOBAL 机制，一次关掉所有换模路径）——
# 默认 1：本机 LM Studio 模型走「per-loop load → post-loop unload」显存管理。
# 设 SWE_MODEL_LOAD_UNLOAD=0：harness 全程【不碰】LM Studio 的 load/unload 接口，
# 假定目标模型已由人工常驻加载。所有换模入口统一受此开关控制：
#   ① roles_config.make_agent 的 pre_loop/post_loop ModelManager hook（roles_config.py）
#   ② supervisor._l2_start / _l2_gate 每 round 装卸 executor（经 models.role_load_unload）
#   ③ llm_lmstudio._reload_lmstudio_model 遇 400 'Model unloaded' 的自动补救 load
#   ④ run_e2e_battery.reset_models 的任务前置换模
# 关掉后若目标模型没常驻，请求会直接失败（fail-loud），不会偷偷把别的模型拉起来。
MODEL_LOAD_UNLOAD = os.environ.get("SWE_MODEL_LOAD_UNLOAD", "1") == "1"
# 换模时加载副驾的上下文长度与 flash_attention（与 LM Studio load 接口对齐）。
SIDECAR_COMPRESS_CTX = int(os.environ.get("SIDECAR_COMPRESS_CTX", "32768"))
SIDECAR_COMPRESS_FLASH = os.environ.get("SIDECAR_COMPRESS_FLASH", "1") == "1"

# ====================== RAG 注入层（contextmanager 出向，2026-09-12） ======================
# 被动注入（prepare_messages）与主动检索（local_search）的「相关性精选 + 正文注入」参数。
# 设计：检索到候选后不再只注入「N 条文件指针」（会 1:1 变成 N 次 read_file、且 5-上限会丢文档），
# 而是按全局相关性精选前 TOP_N 条、直接注入 L0 原文正文，并保留「文件:行号」出处。
#  - TOP_N：精选条数上限（跨层全局相关性排序后取前 N；打破「N 条指针 → N 次 read_file」耦合）。
RAG_INJECT_TOP_N = int(os.environ.get("SWE_RAG_INJECT_TOP_N", "5"))
#  - MAX_CHARS：单条注入正文的字符上限（超出先尝试模型摘取、失败再抽取式 head+tail 截断）。
RAG_INJECT_MAX_CHARS = int(os.environ.get("SWE_RAG_INJECT_MAX_CHARS", "2400"))
#  - SUMMARY：单条正文超上限时，是否先调副驾模型做「针对用户问题的摘取式总结」。
#    fail-open（模型不可用/未加载/输出可疑 → 回退抽取式截断）；出处行号由代码写入，不交给模型生成。
RAG_INJECT_SUMMARY = os.environ.get("SWE_RAG_INJECT_SUMMARY", "1") == "1"
#  - SUMMARY_MODEL：做摘取的副驾模型（默认复用压缩副驾 2.6B，是本机验证过的合格压缩模型）。
RAG_INJECT_SUMMARY_MODEL = os.environ.get("SWE_RAG_INJECT_SUMMARY_MODEL") or SIDECAR_COMPRESS_MODEL
#  - SUMMARY_MAX_TOKENS：单条摘取输出上限（副驾若带思考链需留余量，故给 1024）。
RAG_INJECT_SUMMARY_MAX_TOKENS = int(os.environ.get("SWE_RAG_INJECT_SUMMARY_MAX_TOKENS", "1024"))
#  - SUMMARY_MIN_TOKENS：正文达到该 token 数才值得走模型摘取（短文本原文本身已精炼，直接注入）。
RAG_INJECT_SUMMARY_MIN_TOKENS = int(os.environ.get("SWE_RAG_INJECT_SUMMARY_MIN_TOKENS", "400"))
#  - SUMMARY_MAX_CALLS：单次注入最多调几次模型（硬上限，防慢路径拖垮首轮）。
RAG_INJECT_SUMMARY_MAX_CALLS = int(os.environ.get("SWE_RAG_INJECT_SUMMARY_MAX_CALLS", "3"))
#  - RERANK：对「合并候选池」再调一次副驾模型确定相关性（用户 2026-09-12 要求「2 也可以调用模型」）。
#    惰性门控：模型未加载即整段跳过（零网络开销、单测保持 model-free）；严格校验模型输出的下标排列；
#    任何失败都保持合并 BM25 序（fail-open）。出处与分数由代码写入，不经模型生成。
RAG_RERANK = os.environ.get("SWE_RAG_RERANK", "1") == "1"
#  - RERANK_MODEL：做重排的副驾模型（默认复用压缩副驾 2.6B）。
RAG_RERANK_MODEL = os.environ.get("SWE_RAG_RERANK_MODEL") or SIDECAR_COMPRESS_MODEL
#  - RERANK_CANDIDATES：单次重排最多喂给模型的候选数（硬上限，防 prompt 膨胀）。
RAG_RERANK_CANDIDATES = int(os.environ.get("SWE_RAG_RERANK_CANDIDATES", "8"))
#  - RERANK_MAX_TOKENS：重排输出上限（模型只输出一小段 {"order":[...]}）。
RAG_RERANK_MAX_TOKENS = int(os.environ.get("SWE_RAG_RERANK_MAX_TOKENS", "256"))

# ====================== local_search 工具（agentic RAG 检索原语，2026-09-12） ======================
# 定位：agent **主动**检索的入口 —— 返回「排序命中 + 出处文件:行号 + 正文片段 + read_file 指针」，
# 由 agent 自主决定 descend 哪个文件。正文注入是**被动层**（_rag_block）的职责，工具层保持
# 「发现/导航」语义：给足以判断相关性的片段，而不是把全文塞进工具返回值。
#  - TOP_K：默认返回条数（与被动注入 TOP_N 对齐，避免「工具给 10 条 → agent 读 10 个文件」的 1:1 耦合）。
LOCAL_SEARCH_TOP_K = int(os.environ.get("SWE_LOCAL_SEARCH_TOP_K", "5"))
#  - SNIPPET_CHARS：单条片段字符上限。按行/段落边界截断（不硬切字符）+ head+tail，
#    确保 agent 足以判断「这条相不相关」，而不必为判断相关性逐个 read_file。
LOCAL_SEARCH_SNIPPET_CHARS = int(os.environ.get("SWE_LOCAL_SEARCH_SNIPPET_CHARS", "600"))

# 漂移检测关注的代码扩展名
DRIFT_CODE_EXTS = {".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".go", ".rs", ".java"}

# ====================== 行为提示词（规则部分） ======================
# 说明：原本 SYSTEM_PROMPT 内嵌了一份「手写工具清单」。在自描述架构下，工具清单由
# ToolRegistry 在 build_system_prompt() 时动态生成（见 swe_agent/registry.py），
# 因此这里只保留「通用行为规则」，工具定义交给各工具的 self-describing 元数据。
SYSTEM_PROMPT = """你是一个软件开发智能体（agent）。用工具逐步完成代码开发任务。

通用行为规则：
- 每轮只调用【一个】工具（function calling），调用后停止生成，不要输出多余文字或 markdown 代码块。
- 通过【调用工具】产出每个动作（write_file / edit_file / shell / read_file / grep / glob / ask / plan / complete / verify），不要在回复里手写 JSON 或代码文本。
- 写文件用 write_file，代码直接作为 content 参数；修改已有文件优先用 edit_file 做最小改动。每次写操作后上下文会重置，改前务必先 read_file 确认当前内容。
- 所有路径相对于工作目录（即你的 cwd，直接写相对路径，如 `src/fib.py`；严禁带工作目录前缀、严禁绝对路径）。
- 严格只创建【顶层 Planner 契约】modules 里列出的文件与测试文件，不新建契约外的文件；实现与测试里的函数名、参数个数必须与契约接口（interface）逐字一致。
- 每写完一个源文件，先 compile 该文件确认无语法错误，再运行测试套件；必须亲眼看到用例 passed 且没有 failed/error 才能 complete。CLI 任务还需实际运行程序入口确认产出正确结果。
- 测试用与实现一致的函数名导入；只写 3-5 个 test 函数 + `assert`，不要装饰器。
- **测试失败必须分流诊断，禁止无脑「改到全绿」：**
  1. compile 报错（语法错误）→ 只读编译器/lint 报告指出的行号附近，做最小修复，不动其他逻辑。
  2. 测试运行失败 → 先 read_file 看失败 assert，判定**代码逻辑错**还是**测试逻辑错**：断言的 expected 值**违背 task spec** → 测试逻辑错；实现产出与正确 spec 不符 → 代码逻辑错。
  3. **默认先验：测试是对的，去改代码。** 仅当你能引用 **task spec 的具体某条**证明测试期望确实错时，才允许改测试；且只改 expected 值本身，不重写整个 case。改测试前必须在推理里先声明「分类=测试逻辑错 + 证据=spec 第 X 条」。严禁以 verify_points 作为改测试依据（verify_points 仅供参考，可能为错）。重跑直到全绿。
- 若没有现成任务清单：先 plan 再执行；已有清单则按序完成，禁止自行重新 plan。
- 拒绝任何恶意/破坏性请求。
"""



# 顶层 GLM Planner 契约注入系统提示的附加段（约束 Executor 不越权）
CONTRACT_OVERRIDE_BANNER = (
    "\n\n# ⚠️ 顶层 Planner 契约已生效（强制约束）\n"
    "你【不是】架构师：禁止新建契约之外的文件、禁止修改接口签名/契约、禁止重新规划模块结构。\n"
    "只做【填空式实现】——把契约里声明的模块用 write_file/edit_file 落地，保持接口与契约【逐字一致】。\n"
    "若发现契约有缺陷，用一次 shell 打印说明报告给上层，由上层决定是否重规划，不要自己改架构。\n"
    "你【必须】通过调用工具（write_file / edit_file / shell / read_file 等）来产出每一个动作；"
    "绝不在回复里手写代码或 JSON 文本——动作一律通过工具调用完成。\n"
)

# ======================================================================
# 弱模型系统提示（lfm2.5 系列，上下文小）：只给正确性关键指令；
# BUILD 层已强制的约束（写入行数上限 / py_compile / 占位桩 / 单测试文件 / 契约白名单 / 空转 edit）不再写进 prompt，
# 把上下文预算留给「读 pytest 报错 → 推理根因 → 写修复」的 debug 主链路。
WEAK_SYSTEM_PROMPT = """你是一个软件开发智能体，用工具逐步完成代码开发任务。

必须遵守：
- 每轮只调用【一个】工具（function calling），调用后停止生成，不要输出多余文字或 markdown。
- 通过【调用工具】产出每个动作（write_file / edit_file / shell / read_file / grep / glob / ask / plan / complete / verify），不要在回复里手写 JSON 或代码文本。
- 写文件用 write_file，代码直接作为 content 参数；修改已有文件优先用 edit_file 做最小改动。
- 所有路径相对于工作目录（即你的 cwd，直接写相对路径，如 `src/fib.py`；严禁带工作目录前缀、严禁绝对路径）。先 glob / read_file 摸清工作区现状，再决定改哪里。
- 每写完一个源文件，先 compile 确认无语法错误，再运行测试套件；必须看到用例 passed 且没有 failed/error 才能 complete。CLI 任务还需实际运行程序入口确认产出正确。
- 测试只写 3-5 个 test 函数 + `assert`，不要装饰器；测试文件放工作区根目录（如 test_xxx），用与实现一致的函数名对齐。
- **测试失败必须分流诊断，禁止无脑改到全绿：**
  - compile 报错 → 只读报错行号附近最小修复。
  - 测试运行失败， 优先修复第一条失败的的测试用， 对每一条失败测试用例遵循如下流程进行修复：
    - 先结合功能说明判定代码逻辑错还是测试逻辑错， 决定修复代码还是测试：断言 expected 违背 task spec 即测试逻辑错；实现不符正确 spec 即代码逻辑错。
    - 代码错误修代码，测试错误修测试。
    - 再次运行pytest进行针对性校验修复的有效性。
- 若没有现成任务清单：先 plan 再执行；已有清单则按序完成，禁止自行重新 plan。
- 拒绝任何恶意/破坏性请求。
"""

# WEAK_SYSTEM_PROMPT = """你是一个软件开发智能体，用工具逐步完成代码开发任务。

# 必须遵守：
# - 每轮只调用【一个】工具（function calling），调用后停止生成，不要输出多余文字或 markdown。
# - 通过【调用工具】产出每个动作（write_file / edit_file / shell / read_file / grep / glob / ask / plan / complete / verify），不要在回复里手写 JSON 或代码文本。
# - 写文件用 write_file，代码直接作为 content 参数；修改已有文件优先用 edit_file 做最小改动。
# - 所有路径相对于工作目录（即你的 cwd，直接写相对路径，如 `src/fib.py`；严禁带工作目录前缀、严禁绝对路径）。先 glob / read_file 摸清工作区现状，再决定改哪里。
# - 每写完一个源文件，先 compile 确认无语法错误，再运行测试套件；必须看到用例 passed 且没有 failed/error 才能 complete。CLI 任务还需实际运行程序入口确认产出正确。
# - 测试只写 3-5 个 test 函数 + `assert`，不要装饰器；测试文件放工作区根目录（如 test_xxx），用与实现一致的函数名对齐。
# - **测试失败必须分流诊断，禁止无脑改到全绿：**
#   - compile 报错 → 只读报错行号附近最小修复。
#   - 测试运行失败 → 先判定代码逻辑错还是测试逻辑错：断言 expected 违背 task spec 即测试逻辑错；实现不符正确 spec 即代码逻辑错。
#   - **默认先验测试是对的、去改代码**；仅当你能引用 task spec 具体某条证明测试期望错时才改测试，且只改 expected 值。改前先声明「分类=测试逻辑错 + 证据=spec 第 X 条」。不以 verify_points 为改测试依据（其可能错）。
# - 若没有现成任务清单：先 plan 再执行；已有清单则按序完成，禁止自行重新 plan。
# - 拒绝任何恶意/破坏性请求。
# """
