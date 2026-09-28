from rapid_ml import Model
from swe_agent.log import logger

model = Model("../models/Ling-3.0-tiny", reasoning_parser="ling3")

resp = model.chat(
    messages=[{"role":"user","content":"你的问题"}],
    max_tokens=512,
    stream=False
)

logger.info('%s %s', '思考：', resp.reasoning)
logger.info('%s %s', '答案：', resp.content)
