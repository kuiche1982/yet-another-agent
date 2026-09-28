# TDD 闸门 + 必然失败防护 + 包结构自愈（demo.py）

> 目标：用户确认按 TDD 流程改造 harness，并**明确要求「别在必然 fail 的测试上打转」**。
> 复跑「开发TODO list 页面」与「开发命令行版TODO List程序，Python技术栈」后，把根因锁定在「测试层没人管」，
> 故本轮在 harness 内新增三道机制（不再靠提示词），覆盖之前两轮的头号死因。

## 1. 新增机制

### ① TDD 闸门：测试栈必须与技术栈匹配（`_detect_test_stack_mismatch`）
- 判定 `detect_stack()` 得到的 `lang`（python / node / frontend）与**实际存在的测试文件语言**是否一致。
- 前端/Node 项目里若存在 `.py` 测试（如 run-1 的 `tests/test_app.py` 用 pytest import `.js`），判定为**结构性不匹配**。
- 关键修正：即使「合法栈」的语法校验（`node --check`）是绿的，只要存在错栈测试文件就**绝不判成功**——
  否则会复现 run-1 那种「node --check 绿了，但 pytest import .js 的测试被静默忽略」的假成功。
- 命中 → 直接 `return "strategy"`，强制换测试框架并遵循 TDD（先写会失败的测试钉接口→实现变绿），禁止沿用错栈测试原地重试。

### ② 必然失败防护（`_validation_error_signature` + `_VAL_SIG_HIST` / `_VAL_DOOMED_STREAK`）
- 每次校验失败，从输出抽取**根因签名**（优先 `ModuleNotFoundError: No module named 'X'` / `ImportError` / `AttributeError` / `AssertionError` 等）；
  抓不到则用输出尾部归一化 hash 兜底（失败文本变了就算不同根因）。
- 同一根因签名**连续出现 ≥ `VAL_DOOMED_THRESHOLD`(=2) 次** → 判定为「必然失败」，强制 `strategy`，**禁止继续原地重试**。
- 不同根因（如 import 错误解决后出现真实逻辑失败）→ 签名变化 → 计数重置，允许正常推进。
- 这正对应「别在必然 fail 的测试上打转」：相同报错重复 = 方案结构性无法通过，空耗轮次。

### ③ 包结构自愈（`_try_package_selfheal`，TDD 安全网）
- 当 Python 测试因**本地模块未打包**而 `ModuleNotFoundError: No module named 'src'` 时，
  自动补 `src/__init__.py` 与根 `conftest.py`（`sys.path.insert(0, 根目录)`），让 `from src.todo` 可达。
- **强约束**：仅对「磁盘上确实存在该目录」的本地模块生效（缺失实现的不自愈）；每个模块只修一次（`_SELFHEAL_DONE` 去重）；**绝不循环**。
- 自愈成功后返回 `continue` 立即可复验；若复验仍同一根因则走 ② 的必然失败分支。

## 2. 改动位置（demo.py）
- 常量区：新增 `VAL_DOOMED_THRESHOLD=2`、`_VAL_SIG_HIST`、`_VAL_DOOMED_STREAK`、`_SELFHEAL_DONE`。
- `run_agent` 开头重置上述状态（保证每次运行独立）。
- 新增函数：`_detect_test_stack_mismatch` / `_validation_error_signature` / `_try_package_selfheal`。
- `_run_final_validation` 内，在「no tests」检查之后、通过判定之前插入 ①② 两道闸门。
- 规划 subtask 文案补一句 TDD 引导（硬约束仍在 harness，提示仅作引导）。

## 3. 离线验证（`verify_tdd_guard.py`，全部通过）
| 用例 | 模拟场景 | 期望 | 结果 |
|---|---|---|---|
| A | 前端项目 + `tests/test_app.py`(pytest import .js)，`node --check` 绿 | 判测试栈不匹配 → `strategy` | ✅ strategy×N，且**未假成功** |
| B1 | Python `from src.todo` 无打包 | 先自愈 1 次（建 `__init__.py`+`conftest.py`）→ `continue`；若同根因再犯 → `strategy` | ✅ |
| B2 | 自愈后变绿 | `continue` → `break` | ✅ |
| unit | 签名提取 | 同一 import 错误签名稳定；真实逻辑失败得到不同签名 | ✅ |
| unit | 不匹配判定 | A=True / B=False | ✅ |

## 4. 仍待办（用户未要求本轮做，留作下一步）
- ④ 目录型交付物闸门更严（`tests/`、`src/` 等要求真实非空）。
- 前端「无自测」强约束：目前前端仅 `node --check`，不强制 Jest 测试（受网络/jest 安装影响，需谨慎开启）。
- 语义进展判据：已被「必然失败根因签名」部分覆盖（同根因重复=无进展），但「每轮换不同文件却原地打转」仍依赖全局停滞/循环防护兜底。

## 5. 实战场复测
- 已启动「开发命令行版TODO List程序，Python技术栈」实跑（后台任务 TUg8aj），验证新闸门在真实 LLM 循环中是否如期触发（尤其 `No module named 'src'` 的自愈与必然失败防护）。
