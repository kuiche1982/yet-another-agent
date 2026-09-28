 
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import json

local_model_path = "~/kuiwork/workdir2/litertlm/models/LFM2.5-350M-MLX-4bit"

model, tokenizer = load(
    local_model_path,
    tokenizer_config={"trust_remote_code": True}
)

 # Adjusted sampling for deterministic validation tasks
sampler = make_sampler(
      temp=0.3,          # Lower temperature for consistency
      top_p=0.9,         # Slightly lower than before
      top_k=10,
      min_p=0.2,
  )
response = generate(
    model,
    tokenizer,
    prompt='''Is the following string a valid phone number? 13521871956''',
    max_tokens=50,
    sampler=sampler,
    verbose=False
)
print(response)





