# 完成度实测报告（2026-08-28，第 4 次综合评估）

环境：rapid-mlx + Ling-3.0-tiny @127.0.0.1:8000，复用服务；`DEMO_MAX_TOKENS=8192`，`MAX_ITER=30`。
Harness：上一轮已落地的截断防护 + 技术栈感知校验 + 悬空引用检测 + **交付物闸门** + **全局停滞/循环防护**。
本次新跑两个任务，评估「模型在 harness 约束下能否真正把任务交付到位」。

---

## 任务一：开发 TODO list 页面（前端 / HTML+CSS+JS）

- 运行：39m17s，30/30 轮跑满，`STATS` 调用 34 次、动作 25 个（plan×5 / read×1 / shell×8 / write×11）。
- 产物（agent_sandbox 最终态）：
  - `index.html` 897B、`styles.css` 1748B（HTML 的 `href/scr` 引用均落地，**无悬空引用**）
  - `src/app.js` 5189B —— `node --check` 通过（语法有效）
  - `tests/test_app.py` 9731B（pytest，但 `from app import TodoStore` —— 把 **.js 当 .py 导入** → `ModuleNotFoundError`）
  - `tests/test_app.js` 1907B（Jest 风格，无 runner，环境无 jest）
- 关键行为：
  - 截断防护命中 3 次（轮 4/16/23/29 `finish_reason=length`，被拦下重写）。
  - **交付物闸门生效**：写 `index.html` 后任务「搭建文件结构」未立即标记完成，直到 `styles.css / src/app.js / tests/` 都落盘。
  - 全局循环/停滞防护 **未触发**：模型每轮换不同动作（重写不同文件/重新 plan），workspace 签名持续变化 → 不判为「无进展」。属已知小模型「原地打转」盲区。
  - 终态校验：pytest 收集即 `ModuleNotFoundError: No module named 'app'`（exit 2）→ `校验未通过`，**从未打印 `✅ 校验通过`（0 次）**。
- 结论：**未通过（PARTIAL）**。页面骨架 + 有效 JS 语法齐全、引用无悬空，但**测试层完全对不上栈**（JS 项目用 pytest 导入 .js / Jest 无 runner），且多次长文件截断，模型始终无法收敛到通过校验。Harness 正确**拒绝假成功**。

---

## 任务二：开发命令行版 TODO List 程序（Python 技术栈）

- 运行：24m19s，30/30 轮跑满，`STATS` 调用 32 次、动作 27 个（plan×1 / read×3 / shell×10 / task_output×2 / write×11）。
- 产物（agent_sandbox 最终态）：
  - `src/todo.py` 2992B（CLI + 持久化，类 `Todo`，txt 格式存储，**中途被整体重写为新版 API**）
  - `src/tests/test_todo.py` 8385B（pytest，27 个用例，import `from src.todo import ...`）
  - `src/.gitignore`、`src/requirements.txt`
- 关键行为：
  - **全局防护首次实战场触发**：轮 10–12 连续 3 次 `python -m pytest --version` → 轮 12 触发 **repeat 告警**并强制换动作；后续又触发 **stuck（连续 5 轮无进展）** 与 **强制重规划**（共 3 次告警）。防护按设计工作。
  - 截断防护命中 2 次。
  - 终态校验：`pytest -q src/tests/test_todo.py` → `ModuleNotFoundError: No module named 'src'`（exit 2）→ `校验未通过`，**0 次 `✅ 校验通过`**。
- 失败根因（两层）：
  1. **结构性（打包）**：测试在 `src/tests/` 内 `from src.todo import ...`，但 `src/` 既无 `__init__.py`、也无 `conftest.py`/`PYTHONPATH` 注入 → pytest 收集即失败。纯工程疏忽。
  2. **逻辑性（真 bug）**：补齐打包（加 `__init__.py`+`conftest.py`）后重跑，结果 **20 passed / 7 failed** —— 说明即便打包修好，实现仍有实质缺陷：`Todo.__init__` 不接受 `done=` 关键字、load 解析空行计数偏差、main 子进程路径 `python todo.py` 找不到入口等。
- 结论：**未通过（PARTIAL→偏低）**。Python 栈本应是 harness 强项（pytest 校验直接可用），但模型把包结构做错、且实现与测试接口对不齐。Harness 正确拒绝假成功；全局防护在此任务价值显著（救回 3 次死循环/停滞）。

---

## 横向对比与 harness 健康度

| 维度 | 任务一（前端） | 任务二（Python CLI） |
|---|---|---|
| 是否跑满 30 轮 | 是 | 是 |
| 截断防护命中 | 3 次 | 2 次 |
| 交付物闸门 | 生效 | 生效 |
| 全局循环/停滞防护 | 未触发（持续换动作） | **触发 3 次（救回死循环）** |
| 终态 `✅校验通过` | 0 | 0 |
| 假成功 | 无 | 无 |
| 核心问题 | 测试栈选错（pytest 导入 .js） | 包结构错 + 实现/测试接口不符 |
| 代码可用度 | JS 语法有效，页面引用完整 | 逻辑 20/27 用例通过（修打包后） |

**Harness 侧结论（全部符合预期，无回退）**：
- 截断防护、交付物闸门、悬空引用检测、技术栈感知校验、全局停滞/循环防护 **全部生效**。
- 最关键：**两轮均 0 次 `✅校验通过`，无任何旧测试冒充/假成功** —— 之前「Tetris 误报成功」那类致命缺陷已根治。
- 全局防护从「设想」变为「实战场验证有效」（任务二连续 3 次同命令被拦下）。

**模型/任务侧仍是瓶颈（非 harness 缺陷）**：
1. 小模型写大文件易截断（已靠回灌缓解，但仍耗轮次）。
2. 栈/包结构规划弱：前端任务硬套 pytest 导入 .js；Python 任务把 `src/` 当包却无 `__init__.py`/`conftest`。
3. 校验失败后「原地打转型」：换文件名/重写同逻辑却不解决根因；任务一因此耗尽 30 轮。
4. 已知盲区：全局防护对「每轮换不同动作但实质无进展」不敏感（任务一），需更强的「语义进展」判据（如校验误差集合是否缩小）。

---

## 待办（可选，需用户确认后再动手）

- A. 终态校验对「纯前端无自测」的兜底：前端任务若无 JS 测试，至少要求 `node --check` 全部通过 + 可选 jsdom 冒烟；当前已做语法门禁，可补「至少 1 个能跑的 jest/vitest」强约束。
- B. 包结构自愈：校验报 `No module named 'X'` 时，harness 自动补 `conftest.py`（注入 `sys.path`）或提示 `python -m pytest` 的正确 cwd，降低模型打包错误成本。
- C. 语义进展判据：用「校验报错行数 / 失败用例数是否下降」替代纯文件签名，触发 stuck 更早。
- D. 交付物闸门对「目录」的判定更严格：任务二计划写 `tests/` 却落到 `src/tests`，闸门放行过早（模型把 `tests` 误建成文件也被后续动作盖过），可强化目录型交付物的校验。
