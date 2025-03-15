import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fine_tune import MODEL_TO_OUTPUT

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LOADED_LORA_DIR = MODEL_TO_OUTPUT[MODEL_NAME]


# 2. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Some OPT tokenizers may not define a pad_token by default
# so we set it to the eos token to avoid errors
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Load the original baseline model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 4. (Optional) Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 5. Prepare a prompt
prompt = "Explain the difference between a list and a tuple in Python."

# 6. Tokenize and move inputs to the same device as the model
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 7. Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

# 8. Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"=== Baseline {MODEL_NAME} Response ===")
print(response)


# 1. Load base model
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 2. Load LoRA weights
model = PeftModel.from_pretrained(base_model, LOADED_LORA_DIR)

# 3. Move model to GPU
model = model.to("cuda")

# 4. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(LOADED_LORA_DIR, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(prompt, return_tensors="pt")

# 5. Move inputs to GPU
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 6. Generate
outputs = model.generate(**inputs, max_new_tokens=100)
print(f"=== LoRA {MODEL_NAME} Response ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

