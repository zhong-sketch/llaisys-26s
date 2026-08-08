import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = r"D:\LLAISYS\models\DeepSeek-R1-Distill-Qwen-1.5B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float32,
)

model.eval()

prompt = "请简单介绍一下人工智能。"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
