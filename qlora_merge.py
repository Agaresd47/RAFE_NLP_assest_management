from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "/projects/p32908/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507",
    max_seq_length = 32768,
    dtype          = torch.bfloat16,
    load_in_4bit   = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
)

from safetensors.torch import load_file
from peft import set_peft_model_state_dict
weights = load_file("/scratch/xla2767/models/nlp_combined_new/checkpoint-65/adapter_model.safetensors")
set_peft_model_state_dict(model, weights)

model.save_pretrained_merged(
    "/scratch/xla2767/models/try_delete_latter3",
    tokenizer,
    save_method="merged_16bit",
)
print("Merge 完成！")

# 验证
from transformers import AutoModelForCausalLM, AutoTokenizer

test_model = AutoModelForCausalLM.from_pretrained(
    "/scratch/xla2767/models/try_delete_latter",
    dtype=torch.bfloat16,
    device_map="auto",
)
test_tokenizer = AutoTokenizer.from_pretrained("/scratch/xla2767/models/try_delete_latter")
test_model.eval()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 1+1?"}
]
text = test_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = test_tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = test_model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)

response = test_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"输出: {response}")