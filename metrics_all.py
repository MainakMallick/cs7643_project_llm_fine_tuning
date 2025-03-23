import os
import json
import torch
import datasets
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import types
import doctest

# === CONFIG ===
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODELS = {
    "Base": None,

    "IA3": "IA3-deepseek-ai-1-5b-finetuned"
}
EVAL_LIMIT = 10
MAX_TOKENS = 256

# === LOAD MODEL ===
def load_model(base_model, adapter_path=None):
    print(f"\n🔄 Loading model: {adapter_path or 'Base'}")

    if adapter_path is None:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# === CLEANUP GPU ===
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# === GENERATE CODE ===
def generate_code(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === EVALUATE HUMANEVAL ===
def evaluate_humaneval(model, tokenizer, data, model_name):
    results = []
    passed = 0

    for i, example in enumerate(tqdm(data, desc=f"Evaluating {model_name}")):
        prompt = example["prompt"]
        reference = example.get("canonical_solution", "")
        pred = generate_code(model, tokenizer, prompt)

        try:
            code = "from typing import *\n" + pred.split("\nif __name__")[0]
            local_env = {}
            exec(code, local_env)

            test_pass = False
            for name, obj in local_env.items():
                if isinstance(obj, types.FunctionType):
                    doctest.run_docstring_examples(obj, local_env, name=name, verbose=False)
                    passed += 1
                    test_pass = True
                    break
        except Exception:
            test_pass = False

        results.append({
            "task_id": example.get("task_id", f"task_{i}"),
            "prompt": prompt,
            "reference_solution": reference,
            "predicted_code": pred,
            "pass": test_pass
        })

    accuracy = passed / len(results) if results else 0.0
    return accuracy, results

# === MAIN ===
if __name__ == "__main__":
    dataset = datasets.load_dataset("openai_humaneval", split="test")
    examples = [x for x in dataset if "prompt" in x][:EVAL_LIMIT]

    all_results = {}

    for name, adapter_path in MODELS.items():
        model, tokenizer = load_model(BASE_MODEL, adapter_path)
        acc, details = evaluate_humaneval(model, tokenizer, examples, name)
        all_results[name] = {
            "accuracy": acc,
            "details": details
        }
        print(f"\n✅ {name} accuracy: {acc:.4f}")
        cleanup()

    with open("humaneval_debug_outputs.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n📄 Saved results to humaneval_debug_outputs.json")
