# 结构化输出（structured output）统一 plan 格式 —— 定向验证

> 目标：用 `response_format: json_schema` 约束 GLM(planner 大脑) 与 LM Studio 的
> qwen(executor 兜底) 的 `plan` 输出，看能否统一成 harness 期望的扁平格式
> `{"tasks": ["字符串描述", ...]}`（之前 qwen 走 tool calling 把 plan 写成嵌套 dict
> 导致 `_verify_task_deliverables` 对 dict 跑正则崩溃）。
> 本次只做定向验证，**未跑全量 harness**。

## 验证结果（证据）

| 后端 | response_format | 是否生效 | 返回内容形态 | 扁平 `{"tasks":[str]}` |
|------|----------------|----------|--------------|------------------------|
| LM Studio qwen (`qwen2.5.1-coder-7b-instruct`) | `json_schema`(strict+additionalProperties:false) | ✅ 强制生效 | 原始 JSON，无围栏 | ✅ 完美 `{"tasks":[...7项...]}` |
| GLM (`glm-4.7`, `open.bigmodel.cn/api/paas/v4`) | `json_schema`(4 种变体：strict/no-strict/addprop) | ❌ **被忽略** | markdown 围栏 ```` ```json {...} ````，结构任意（有的还嵌套 `plan[].actions`） | ❌ `json.loads` 直接失败 |
| GLM | `json_object` | ✅ 生效 | 原始 JSON，无围栏 | ✅ 用 prompt 约束 `tasks` 键后产出 `{"tasks":[...8项...]}` |

## 结论

1. **LM Studio qwen** 可用 `json_schema`（strict + `additionalProperties:false`）**强制**产出扁平
   `{"tasks":[string]}` —— 结构化输出对它完全可用，能根治之前嵌套 dict 崩溃。
2. **GLM 的 `json_schema` 在本 endpoint/model 不被采纳**（返回 fenced markdown，等于没约束）。
   但 GLM 的 **`json_object` 是被采纳的**，且配合 prompt 里要求 `tasks` 键，能稳定产出
   同样的扁平 `{"tasks":[string]}`。
3. 因此**两者可以统一到同一个输出形状** `{"tasks":[string,...]}`，只是机制不同：
   - GLM → `response_format={"type":"json_object"}` + prompt 约束用 `tasks` 键
   - LM Studio qwen → `response_format={"type":"json_schema",...}`（strict）
4. **harness 侧需要的唯一适配**：解析器要 (a) 先剥掉 ```` ```json ```` 围栏（防 GLM
   json_schema 退化情形），(b) `json.loads` 后读 `tasks` 字段。两后端经此解析后形状一致。

## 与之前崩溃的关系

- 之前崩溃发生在 **executor 走 tool calling 时 qwen 把 plan 写成嵌套 dict**。
- 若 executor 改走 LM Studio `json_schema`（强制扁平），该崩溃被根治。
- GLM 作为 planner 时本来是 `glm_chat()` 路径（content 解析），其 plan 经 `roles.py`
  写入 `GLOBAL_STATE["plan"]`；用 `json_object` + `tasks` 键后格式更稳。
- **注意**：在 faithful 架构（GLM planner + qwen executor）下，executor 的 `plan` 动作被
  `supervisor.py:506` 守卫拒绝（"禁止重新规划"），原本就不会触发嵌套 dict 崩溃；该崩溃是
  `LOCAL_BRAIN=1`（planner 退化成 qwen 自己）时才会踩到。

## 验证脚本（均只读探针，未改 harness 主流程）

- `verify_structured_plan.py` —— 初次双向验证（暴露 GLM 空内容问题）
- `verify_glm_struct.py` —— GLM 4 版 schema 变体 + LM Studio 回测
- `verify_glm_final.py` —— GLM 官方文档风格 json_schema + json_object 带 tasks 键

## 下一步（待用户确认，未执行）

- 在 `models.py`/`roles.py` 的 plan 生成处按后端选择 `json_object`(GLM) vs
  `json_schema`(LM Studio) 的 response_format，并加围栏剥离 + `tasks` 读取的解析器。
- 仍不跑全量，先把新的 plan 解析跑一个定向回放确认。

## 补充：tools + response_format 不能在 GLM 上混用（11:34 用户 curl 触发验证）
- 用户贴出 GLM 官方文档：response_format.type 仅支持 text / json_object（**无 json_schema**），并以 tools + json_object 共存的 curl 为例。
- 定向验证（verify_glm_tools_jsonobject.py）：在 GLM(glm-4.7) 上同时传 tools + response_format=json_object → 返回**空**（has_tool_calls=False, content=''）。即 GLM 不接受 tools 与 json_object 混用，必须二选一。
- 结论对齐官方文档：GLM 结构化输出 = json_object（已验证可产出扁平 {"tasks":[str]}）；json_schema 本就不是 GLM 选项，故此前 json_schema 尝试被忽略属预期。
- 因此统一方案：planner(GLM) 用 json_object 不带 tools；executor 动作用 tool_calling 不带 response_format；两者在 plan 步骤都只产出扁平 {"tasks":[str]}。

## 接入 harness 完成 + 定向验证（11:42 动手）
- 接线：在传输层加 response_format 透传——llm_glm.glm_chat、llm_lmstudio.lmstudio_chat_messages、llm_local.chat/_local_chat 均新增 response_format 参数；models.chat_text 透传；roles._make_plan 按 planner provider 选 response_format（GLM/LM Studio 均 json_object，local 传 None）。
- 约束：response_format 与 tools 互斥（LM Studio 端仅在无 tools 时下发），避免 GLM/LM Studio 在「调工具 vs 返回 JSON」间摇摆/返回空（此前已实测 GLM tools+json_object 返回空）。
- 定向验证（verify_plan_structured.py，真实 _make_plan，非全量跑）：
  - GLM planner：4 tasks，全部合格 dict+desc ✅（89s）
  - LM Studio qwen planner(qwen-lm-tjson→qwen2.5.1-coder-7b-instruct)：7 tasks，全部合格 ✅（68s）
  - 两者都经 _normalize_plan 产出合法契约，task 形状全合格 → 原崩溃根因（tasks 含 dict/str 不一致致正则抛 dict 异常）已根除。
- 统一结论：GLM 与 LM Studio qwen 的 plan 输出经 json_object 结构化输出归一为同一合法契约形状，解析走统一 _extract_json_object + _normalize_plan 管线。
- 注：executor 的 plan 动作(_m_plan)在 faithful 架构下被守卫禁止（GLM planner 已给定），其 dict 归一化防御仍保留；本接线只动 planner 路径，Reviewer 保持自由文本（未传 response_format）。
