# GLM 视觉模型充当 Planner 验证

验证日期：2026-08-30
结论：**`glm-4.1v-thinking-flashx` 与 `glm-4.6v` 都支持 `response_format=json_object`，且都能作为 Planner 使用。**

## 验证方式
1. 直接打 zhipu OpenAI 兼容接口，分别带/不带 `response_format={"type":"json_object"}`，捕获 HTTP 状态 + 原始 content + JSON 解析结果。
2. 走真实 harness 路径：`PLANNER_MODEL=<model>` → `roles._make_plan` → `models.chat_text` → `llm_glm.glm_chat` + `_planner_response_format()`。

## 结果

| 模型 | json_object 支持 | max_tokens 上限 | 作为 Planner |
|------|------------------|-----------------|--------------|
| glm-4.1v-thinking-flashx | ✅ | 16384（超出即 400 code 1210） | ✅ 端到端生成 1 模块/4 任务 |
| glm-4.6v | ✅（思考模型，reasoning_tokens 计入） | ≥32768 | ✅ 端到端生成 1 模块/3 任务 |

两个模型在 `response_format=json_object` 下都稳定返回可解析的 Planner 契约，并通过 `_normalize_plan`（对 step/deliverables/hidden_tests 做 `str()` 兜底，结构容错）。

## 过程中发现并修复的问题
1. **`_planner_response_format()` 旧实现是死代码**：它把 `spec["provider"]`（PROVIDERS 的 key，如 `"zhipu"`）和 transport 字符串（`"remote_openai"`）比较，条件恒为 False → `json_object` 从未真正下发。已改为读取 `M.PROVIDERS[provider]["transport"]` 判断。修复后两个模型都正确收到 `json_object`。
2. **`glm-4.6v` 不在 MODELS 目录**：原来的 `role_spec` 会抛 `ModelUnknown`。已在 `models.py` 新增 catalog 条目（`provider: zhipu, fc: JSON_MODE`）。
3. **`glm-4.1v-thinking-flashx` 的 max_tokens 超限**：harness 默认 `GLM_MAX_TOKENS=32768` 超过其 16384 上限，触发 `400 code 1210`。已在该 catalog 条目单独封顶 `max_tokens=8192`（Planner 只输出紧凑 JSON，足够）。

## 用法
```bash
# 用免费额度的视觉模型当 Planner（Executor 仍用本地/其他模型）
PLANNER_MODEL=glm-4.1v-thinking-flashx python main.py ...
# 或
PLANNER_MODEL=glm-4.6v python main.py ...
```
> 注意：这两个模型缺乏原生 `tool_choice`，**不适合当 Executor**（函数调用会退化成手写 JSON），但 Planner/Reviewer 走 content + json_object，完美契合，且有大量免费 token。

## 附带验证脚本
- `verify_glm_vision_json.py`：裸接口对照测试（带/不带 response_format）
- `verify_glm_vision_planner.py`：真实 harness 路径端到端测试
- `verify_glm41v_maxtok.py`：定位 400 根因为 max_tokens 超限
