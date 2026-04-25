import json
from pathlib import Path

p = Path("/scratch/xla2767/hold2/models/qwen3_8b_miner/tokenizer_config.json")
data = json.loads(p.read_text())

data["eos_token"] = "<|im_end|>"
data["pad_token"] = "<|im_end|>"
data["padding_side"] = "left"
data["extra_special_tokens"] = {}

p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
print(f"patched {p}")