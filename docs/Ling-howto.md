~/.lmstudio/models/rapid-mlx/Ling-3.0-tiny-MLX-4bit

pip install rapid-mlx

# 本地推理

rapid-mlx chat Ling-3.0-tiny --prompt "你好"

# 启动兼容OpenAI API服务

rapid-mlx serve Ling-3.0-tiny --port 8000

> 错误原因：**rapid‑mlx 没有 `generate` 子命令**，那是原生 `mlx‑lm` 的命令，我之前记错了。

### 方式1：终端交互式聊天（chat）

```bash
rapid-mlx chat Ling-3.0-tiny
```

进入 REPL，直接输入提问，`/exit` 退出。

### 方式2：启动 OpenAI API 服务（对接 LM Studio 外部API，目标方案）

```bash
rapid-mlx serve Ling-3.0-tiny --port 8000 --gpu-memory-utilization 0.65 \
--kv-cache-quantization --kv-cache-quantization-bits 4 \
--use-paged-cache
--default-max-tokens 2048 \


 --reasoning-parser ling3
 `rapid‑mlx` **还没有内置 `ling3` reasoning‑parser**，这个 parser 是 vLLM/SGLang 才有，rapid‑mlx 暂未适配 Ling‑3.0 的思考块解析。

>
> ❌ 不能加 `--reasoning‑parser ling3`，命令直接报错。
```

- 服务地址：`http://127.0.0.1:8000/v1`
- LM Studio → Load Model → External OpenAI‑compatible API，填入上面地址即可使用该模型。

测试服务是否通，新开终端执行 curl：

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Ling-3.0-tiny",
    "messages":[{"role":"user","content":"你好"}]
  }'
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8001/v1/chat/completions \
-H "Content‑Type: application/json" \
-d '{
"model":"Ling-3.0-tiny",
"messages":[{"role":"user","content":"2+3等于几"}],
"stream":false
}'
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model":"Ling-3.0-tiny",
"enable_thinking": true,
"reasoning_effort":"high",
"messages":[
{"role":"user","content":"一个水池，进水3小时满，出水5小时放空，同时打开进水和出水，多久可以放满水池"}],
"stream":false
}'
```

```json
{
  "id": "chatcmpl-251bd71d",
  "object": "chat.completion",
  "created": 1787875299,
  "model": "Ling-3.0-tiny",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "2+3等于5。" },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 26,
    "completion_tokens": 7,
    "total_tokens": 33,
    "prompt_tokens_details": { "cached_tokens": 10 }
  }
}
```

### 方式3：原生 mlx‑lm 直接跑（不通过 rapid‑mlx）

```bash
python -m mlx_lm.generate --model Ling-3.0-tiny --prompt "你好"
```

### LM Studio 接入外部API简要步骤

1. 保持 `rapid‑mlx serve` 终端不要关闭
2. LM Studio → 左侧 `Load Model` → `External OpenAI‑compatible API`
3. API Base URL：`http://localhost:8000/v1`
4. API key 随便填字符串，点击 Connect，就可以对话。

> 注意：Ling‑3.0‑tiny 需要 `--reasoning‑parser ling3`，不然思考块解析异常。
> 自检命令：`rapid‑mlx doctor`，检查环境依赖是否齐全。

需要我给你一段可直接复制的 Python OpenAI 客户端测试代码吗？


cd ~/kuiwork/workdir2/litertlm && .venv/bin/python demo.py "开发 TODO list 网页（前端），从零开始：用 HTML + CSS + JavaScript 实现，产出 index.html、styles.css、app.js，并在 tests/ 下写前端测试。功能：添加待办、标记完成/取消完成、删除待办。完成后用 node --check 校验 JS 语法并尝试运行前端测试。" > eval_gc2_web_r5.log 2>&1



cd ~/kuiwork/workdir2/litertlm && ./.venv/bin/rapid-mlx serve ~/kuiwork/workdir2/litertlm/Ling-3.0-tiny --port 8000 --served-model-name Ling-3.0-tiny --gpu-memory-utilization 0.5 --kv-cache-dtype int8 --resident-memory-limit-gb 11
