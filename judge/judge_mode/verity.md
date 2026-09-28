```
curl -v http://127.0.0.1:1234/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm2.5-2.6b",
    "chat_template_kwargs": { "enable_thinking": false },
    "messages": [{"role":"user","content":"你好"}]
  }'
```

```
curl -v http://127.0.0.1:1234/v1/chat/completions \
  -H "Content‑Type: application/json" \
  -d '{
  "model": "lfm2.5‑2.6b",
  "chat_template_kwargs": { "enable_thinking": false },
  "messages": [{ "role": "user", "content": "你好" }]
}'
```

curl -v http://127.0.0.1:1234/v1/chat/completions \
  -H "Content‑Type: application/json" \
  -d '{
  "model": "lfm2.5-2.6b",
  "messages": [{"role": "user", "content": "Say this is a test!"}],
}'

curl http://localhost:1234/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "chat_template_kwargs": { "enable_thinking": false },
    "model": "lfm2.5-2.6b",
    "system_prompt": "You answer only in rhymes.",
    "input": "What is your favorite color?"
}'

curl http://localhost:1234/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "thinking": false,
    "model": "lfm2.5-2.6b",
    "system_prompt": "You answer only in rhymes.",
    "input": "What is your favorite color?"
}'

先确认 Ling 的精确模型 id，再用它重跑两个探针。
模型 id 是 `Ling-3.0-Tiny`。把两个探针改成读环境变量 `JUDGE_MODEL`，再对 Ling 跑。
Ling 暴露了关键差异，而且探针被默认 120s 超时杀了（Ling 单 call ~35s，是 Spark 的 10 倍）。先报已看到的硬事实：

**Ling-3.0-Tiny 在 temp=0 和 temp=0.5 直接返回空内容（`finish=length`、`raw=''`、解析失败），只有 temp=1.1 才吐出合法 JSON。** 这跟「低温更稳」的直觉相反——Ling 做结构化输出**必须高温**。你定的「默认温度 0」在 Ling 上当裁判会直接空转。

我把探针2（硬 prompt，真实要落地的形态）改成只跑 temp=1.1（Ling 唯一能吐 JSON 的温度），5 个用例跑完，给足超时。
全部为空——Ling 在 temp=1.1 + 硬 prompt 下 5/5 返回空内容（解析失败）。但 probe1 里 temp=1.1 + 软 prompt 的 analyzer-good 那次**明明吐出了合法 JSON**。说明 Ling 是**非确定性翻车**，不是稳定可用。

这是决定性差异：Spark-1.7B 在 json_schema 上其实比 Ling 更稳（12/12 + 10/10）。得区分 Ling 是「系统性坏」还是「偶发抖」——这决定 retry 能否救它。做个聚焦方差测试：analyzer-good 在 temp=1.1 连打 3 次，看几次能出 JSON。
明白了——Ling 带思考链，512 的 `max_tokens` 被 `<think>` 推理吃光，JSON 根本没空间吐出来（所以 `finish=length` + 空内容）。提到 8k 重测。我顺手让解析容错：剥掉 `<think>...</think>`、抓首个 `{` 到末个 `}`。
关键线索——Ling 用 `chat_template_kwargs:{"enable_thinking":false}` 关思考。这比堆 max_tokens 更对：之前空输出/慢是因为它把 token 预算烧在思考上。关掉后低温可能也行了（之前 temp=0 失败也许就是思考吃光预算，不是温度本身）。

先改方差测试加上关思考字段、max_tokens 降回 1024，重跑确认稳+快：
关思考后 Ling 完全变样：**3/3 JSON_OK，且 2–5s 出结果**（之前 60–110s）。`enable_thinking:false` 是 Ling 当 judge 的命门。现在补回全量验证：硬 prompt 的 5 个用例在 temp=0 和 temp=1.1 都跑一遍（确认你「默认温度 0」能否成立），顺带重跑软 prompt 看 hardening 收益。先给两个探针加字段、降 max_tokens。
关思考后 Ling 完全反转：**10/10 全对、2–6s 出结果**，且 temp=0 和 temp=1.1 verdict 一致。证据齐了，跨两个模型都坐实。先把实证结论记进今日日志，再给你完整判读。
两个模型都跑完了，结论干净。

## Ling 翻案：关掉思考就满血

| 配置 | 结果 |
|---|---|
| Ling 默认（带思考）temp=0/0.5 | 返回空内容、`finish=length`、解析失败 ❌ |
| Ling 默认（带思考）temp=1.1 | 偶发能出 JSON，但 **60–110s/次** ❌ |
| **Ling `enable_thinking:false`** | **10/10 全对、2–6s/次** ✅ |

`chat_template_kwargs:{"enable_thinking":false}` 是 Ling 当 judge 的命门——之前空输出和慢都是思考链把 token 预算吃光。关掉后它比 Spark 还快。

## 双模型汇总（thinking-off 后）

| 用例 | Spark-1.7B (硬prompt) | Ling-3.0-Tiny (硬prompt) |
|---|---|---|
| analyzer 有内容 | yes ✓ | yes ✓ |
| analyzer 空 | no ✓ | no ✓ |
| analyzer 客套话 | no ✓ | no ✓ |
| tester 有内容 | yes ✓ | yes ✓ |
| tester 空 | no ✓ | no ✓ |

**两个模型都是 10/10 全对、合 schema、temp=0 与 1.1 verdict 一致。**

## 三条被实证坐实的结论

1. **传输层零风险**：`strict:"true"` 的 json_schema 在 Spark 和 Ling 上都 100% 可解析。我之前担心的「畸形 JSON 崩主循环」不在。
2. **hardened prompt 不可省**：软 prompt 下两模型都把退化输入判成 `notsure`/误判 `yes`；加了「铁律：空/占位符/客套话→必须 no」才全对。**这是落地前提，与模型无关**。
3. **你定的 `notsure→升温(0.5→1.1)` 被证伪**：两模型 temp=0 和 1.1 的 verdict 逐字相同，且 Ling 升温反而更多 `notsure`。**修正为：notsure 保持 temp=0 同 prompt 重试 ×3 → 仍否降级 no 回灌，不升温。**

## 修正后的落地 spec（待你拍板才写代码）

- judge 调用：`temp=0` + `enable_thinking:false`（**按模型**的配置 flag，仅 Ling 类加，Spark/qwen/lfm 不加——这字段不能硬编码全局）。
- judge prompt 强制含「空/占位符/客套话→no」铁律（BUILD 常量）。
- `yes`→采纳；`no`→reason **回灌生产者同一段对话**续轮（exhausted 门禁保留完整上下文，禁 `_base_msgs` 重建，否则 240s 挂死重现）；`notsure`→temp=0 重试 ×3→否当 no。
- tester 保留 BUILD 层客观前置（文件非空 + pytest 收集>0）跳过模型调用；analyzer 靠 hardened prompt。
- 这段**取代 F3 的 `last_content` 长度阈值**，`respect_stop` 保留不动。

证据脚本在 `/tmp`（可复跑，读 `JUDGE_MODEL` 环境变量切模型）。要我按上面 spec 落地代码就说一声——会改 `roles_config.py`(judge 字段) + `roles.py`(analyzer/tester 循环接 judge) + `config.py`(铁律常量) + 双 fake 测试。

（本轮回你验证指令，未改 harness 源码。）
验证完成，证据齐了。Ling 关思考后 10/10 全对、2–6s，跟你点名的 Spark 一样稳。设计可行，但有两处必须按实证修正：**hardened prompt 铁律不可省** + **`notsure` 重试不升温（保持 temp=0）**，以及 Ling 必须带 `enable_thinking:false`（按模型 flag，不硬编码）。

要我按修正 spec 落地代码，回一声即可。
用户授权推进。这是多文件改动，按项目纪律**动手前先 re-read 源码**（脑图仅供参考，代码为唯一基准）。并行读 analyzer 当前实现、`config.py` 常量、`roles_config.py`、以及模型调用路径（确认 `chat_template_kwargs`/`enable_thinking` 怎么传）。
tester 在 `verify.py`（`run_tester`/`verify_gate`），已自带客观验收（读文件+跑测试+evidence）。按我们敲定的 spec，tester 走**客观前置检查**（非空+pytest 收集>0），analyzer 走**模型 judge**。我并行读 remaining 文件：config 常量、roles_config、models 调用路径、verify 的 tester 入口。